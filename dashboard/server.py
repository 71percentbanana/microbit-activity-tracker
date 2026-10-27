"""
Live dashboard server for both micro:bit projects: the gesture classifier (../gesture) and
the activity tracker (project root). Reads the micro:bit over USB Serial and streams
everything to dashboard.html in the browser.

Gestures view: records labelled gestures (Record mode), classifies them with the gesture
Random Forest (Predict mode) and sends the result back to the micro:bit LEDs.
Activity view: classifies the continuous stream every 0.5 s with the activity tracker's
Random Forest and shows the activity on the LEDs.

Usage:
  python server.py               # auto-detect the micro:bit
  python server.py --port COM5   # use a specific serial port
  python server.py --demo        # simulated gestures, no hardware needed
"""

import os
import sys
import json
import time
import re
import queue
import shutil
import argparse
import threading
import webbrowser
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import joblib

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_DIR = os.path.join(os.path.dirname(BASE_DIR), "gesture")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")  # remembers who is recording on this laptop
sys.path.insert(0, GESTURE_DIR)

import train as trainer
import simulator
import activity
from gesture_features import (GESTURES, GESTURE_NAMES, GESTURE_SHORT, GESTURE_GROUPS, LED_CODES, MIN_SAMPLES,
                              extract_features, add_custom_gesture, remove_custom_gesture)

# Predictions below this probability are reported as "not sure"
CONFIDENCE_THRESHOLD = 0.3
# Suggested number of recordings per gesture
TARGET_PER_GESTURE = 30


def find_microbit_port():
    """Attempt to automatically detect connected micro:bit COM port."""
    if not HAS_SERIAL:
        return None
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        mfg = (p.manufacturer or "").lower()
        # micro:bit uses USB vendor ID 0x0D28 (Arm DAPLink); older drivers also name it mbed/micro:bit
        if p.vid == 0x0D28 or "micro:bit" in desc or "mbed" in desc or "micro:bit" in mfg:
            return p.device
    return None


class Hub:
    """Shared state between the micro:bit link, the HTTP handlers and the browser clients."""

    def __init__(self, data_dir: str, model_dir: str, demo: bool):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.demo = demo
        self.lock = threading.Lock()
        self.clients = []
        self.view = 'gesture'  # 'gesture' or 'activity'
        self.mode = 'predict'
        self.label = GESTURES[0]
        self.link = None
        self.link_status = {'connected': False, 'port': None, 'error': None}
        self.history = deque(maxlen=12)
        self.saved_paths = []  # recordings saved this session, for undo
        self.person = 'sim' if demo else self.load_settings().get('person', '')
        self.program = None  # 'gesture' or 'activity_recorder', detected from the serial data
        self.board_model = False  # True once the micro:bit sends its own predictions (PRED lines)
        self.board_prediction = None  # the micro:bit's answer for the gesture that is just ending
        self.training = False
        self.model = None
        self.metrics = None
        self.load_model()
        self.activity = activity.ActivityTracker(self.broadcast)
        self.activity_eval = None
        self.activity_eval_error = None

    # ---- browser clients (Server-Sent Events) ----
    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=2000)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def broadcast(self, event: str, data):
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode('utf-8')
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    # ---- state ----
    @staticmethod
    def load_settings() -> dict:
        try:
            with open(SETTINGS_PATH) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def counts(self, person: str = None) -> dict:
        """Recordings per gesture for one person (default: whoever is recording now)."""
        person_dir = os.path.join(self.data_dir, person if person is not None else self.person)
        counts = {}
        for g in GESTURES:
            gesture_dir = os.path.join(person_dir, g)
            counts[g] = len([n for n in os.listdir(gesture_dir) if n.endswith('.csv')]) if os.path.isdir(gesture_dir) else 0
        return counts

    def team(self) -> dict:
        """Total recordings per person, for everyone in data_dir."""
        people = sorted(d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))) if os.path.isdir(self.data_dir) else []
        return {p: sum(self.counts(p).values()) for p in people}

    def state(self) -> dict:
        return {
            'view': self.view,
            'mode': self.mode,
            'label': self.label,
            'gestures': [{'id': g, 'name': GESTURE_NAMES[g], 'short': GESTURE_SHORT[g], 'group': GESTURE_GROUPS[g]} for g in GESTURES],
            'person': self.person,
            'counts': self.counts() if self.person else {g: 0 for g in GESTURES},
            'team': self.team(),
            'target': TARGET_PER_GESTURE,
            'link': self.link_status,
            'program': self.program,
            'board_model': self.board_model,
            'onboard': self.load_onboard_info(),
            'demo': self.demo,
            'has_model': self.model is not None,
            'metrics': self.metrics,
            'history': list(self.history),
            'training': self.training,
            'can_undo': bool(self.saved_paths),
            'activity': {**self.activity.state(), 'eval': self.activity_eval, 'eval_error': self.activity_eval_error},
        }

    def set_link_status(self, connected: bool, port=None, error=None):
        self.link_status = {'connected': connected, 'port': port, 'error': error}
        self.broadcast('link', self.link_status)

    def load_model(self):
        model_path = os.path.join(self.model_dir, "gesture_rf.joblib")
        metrics_path = os.path.join(self.model_dir, "gesture_metrics.json")
        if os.path.exists(model_path) and os.path.exists(metrics_path):
            self.model = joblib.load(model_path)
            with open(metrics_path) as f:
                self.metrics = json.load(f)

    def send_to_microbit(self, text: str):
        if self.link is not None:
            self.link.write_line(text)

    # ---- events from the micro:bit link ----
    def on_sample(self, x: float, y: float, z: float, recording: bool):
        self.broadcast('sample', {'x': x, 'y': y, 'z': z, 'rec': recording})
        changed = self.activity.on_sample(time.time(), x, y, z)
        if changed and self.view == 'activity':
            self.send_to_microbit("ACT:" + changed.upper())

    def on_device_stats(self, sample_count: float, strength: float, level: float, average: float, peak: float):
        """Statistics computed on the micro:bit by the activity recording program."""
        self.broadcast('device', {'samples': int(sample_count), 'strength': strength, 'level': level,
                                  'average': round(average, 1), 'peak': peak})

    def set_program(self, program: str):
        """Which MakeCode program the micro:bit is running, told apart by its line format."""
        if program == self.program:
            return
        self.program = program
        if program == 'activity_recorder':
            self.broadcast('notice', {'level': 'info', 'text': 'The micro:bit is running the activity recording program. Use the Activity tab: '
                                      'press A on the micro:bit to start streaming, B to stop. Gestures need gesture_classifier.hex.'})
        self.broadcast('state', self.state())

    def on_recording(self, samples: list, duration: float, truth: str = None):
        """Called when button A is released with the samples recorded while it was held."""
        if self.view != 'gesture':
            return
        rec = np.array(samples, dtype=np.float64)
        if len(rec) < MIN_SAMPLES:
            self.broadcast('notice', {'level': 'warn', 'text': 'Recording too short - hold A for the whole gesture.'})
            return
        if self.mode == 'record':
            self.save_recording(rec)
        else:
            self.predict(rec, duration, truth)

    @staticmethod
    def load_onboard_info():
        """Details of the model last exported to the micro:bit (gesture/microbit/onboard_model.json)."""
        try:
            with open(os.path.join(GESTURE_DIR, "microbit", "onboard_model.json")) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def on_board_prediction(self, index: int, percent: int):
        """The micro:bit recognised the gesture itself; it sends this just before END."""
        info = self.load_onboard_info()
        classes = info['classes'] if info else []
        gesture = classes[index] if 0 <= index < len(classes) else None
        self.board_prediction = {
            'gesture': gesture,
            'name': GESTURE_NAMES.get(gesture, gesture or f'class {index}'),
            'confidence': percent / 100,
        }
        if not self.board_model:
            self.board_model = True
            self.broadcast('notice', {'level': 'good', 'text': 'The micro:bit is recognising gestures by itself (on-board model). '
                                      "The dashboard shows its answer next to the laptop's."})
            self.broadcast('state', self.state())

    def save_recording(self, rec: np.ndarray):
        self.board_prediction = None  # not needed while recording training data
        if not self.person:
            self.broadcast('notice', {'level': 'warn', 'text': 'Type your name in "Recording as" first - recordings are saved per person.'})
            return
        gesture_dir = os.path.join(self.data_dir, self.person, self.label)
        os.makedirs(gesture_dir, exist_ok=True)
        path = os.path.join(gesture_dir, f"{self.label}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.csv")
        np.savetxt(path, rec, fmt='%d', delimiter=',', header='x,y,z', comments='')
        self.saved_paths.append(path)
        self.send_to_microbit("SAVED")
        self.broadcast('saved', {'label': self.label, 'n_samples': len(rec), 'counts': self.counts(), 'team': self.team(), 'can_undo': True})

    def predict(self, rec: np.ndarray, duration: float, truth: str = None):
        if self.model is None:
            self.broadcast('notice', {'level': 'warn', 'text': 'No trained model yet - record gestures, then Train.'})
            return
        probs = self.model.predict_proba([extract_features(rec)])[0]
        classes = list(self.model.classes_)
        best = int(np.argmax(probs))
        gesture, confidence = classes[best], float(probs[best])
        confident = confidence >= CONFIDENCE_THRESHOLD
        if not self.board_model:
            self.send_to_microbit(LED_CODES.get(gesture, 'CUSTOM') if confident else "?")
        result = {
            'gesture': gesture,
            'name': GESTURE_NAMES.get(gesture, gesture),
            'confidence': confidence,
            'confident': confident,
            'threshold': CONFIDENCE_THRESHOLD,
            'probs': {c: float(p) for c, p in zip(classes, probs)},
            'n_samples': len(rec),
            'duration_s': round(duration, 2),
            'time': time.strftime('%H:%M:%S'),
            'truth': truth,
            'board': self.board_prediction,
        }
        self.board_prediction = None
        self.history.appendleft(result)
        self.broadcast('prediction', result)

    # ---- actions from the dashboard ----
    def set_view(self, view: str):
        if view in ('gesture', 'activity'):
            self.view = view
            if view == 'activity' and self.activity.current:
                self.send_to_microbit("ACT:" + self.activity.current.upper())
            self.broadcast('state', self.state())

    def start_activity_eval(self):
        """Evaluate the activity model in the background (cached after the first run)."""
        def run():
            try:
                self.activity_eval = activity.evaluate()
            except Exception as e:
                self.activity_eval_error = str(e)
            self.broadcast('state', self.state())
        if self.activity.model is not None:
            threading.Thread(target=run, daemon=True).start()

    def set_mode(self, mode: str):
        if mode in ('predict', 'record'):
            self.mode = mode
            self.broadcast('state', self.state())

    def set_label(self, label: str):
        if label in GESTURES:
            self.label = label
            self.broadcast('state', self.state())

    def undo(self):
        while self.saved_paths:
            path = self.saved_paths.pop()
            if os.path.exists(path):
                os.remove(path)
                self.broadcast('notice', {'level': 'info', 'text': f'Deleted last recording ({os.path.basename(path)}).'})
                break
        self.broadcast('state', self.state())

    def add_gesture(self, name: str):
        try:
            gid = add_custom_gesture(name)
        except ValueError as e:
            self.broadcast('notice', {'level': 'warn', 'text': str(e)})
            return
        self.label = gid
        self.mode = 'record'
        self.broadcast('notice', {'level': 'good', 'text': f'Added "{GESTURE_NAMES[gid]}". Hold A to record it, about {TARGET_PER_GESTURE} times, then Train.'})
        self.broadcast('state', self.state())

    def remove_gesture(self, gid: str):
        if not gid.startswith('c_') or gid not in GESTURES:
            return
        name = GESTURE_NAMES[gid]
        remove_custom_gesture(gid)
        if self.label == gid:
            self.label = GESTURES[0]
        self.broadcast('notice', {'level': 'info', 'text': f'Removed "{name}". Its recordings are kept and come back if you add it again. Train to update the model.'})
        self.broadcast('state', self.state())

    def set_person(self, name: str):
        """Who is recording on this laptop. Saved as a folder name, so keep it simple."""
        slug = re.sub(r'[^a-z0-9_-]', '', str(name or '').strip().lower().replace(' ', '_'))[:30]
        self.person = slug
        if not self.demo:
            with open(SETTINGS_PATH, 'w') as f:
                json.dump({'person': slug}, f)
        self.saved_paths = []
        self.broadcast('state', self.state())

    def reset(self, label: str):
        """Move the current person's recordings of one gesture (or 'all') to <data_dir>_deleted/<timestamp>/<person>/."""
        gestures = GESTURES if label == 'all' else [label] if label in GESTURES else []
        if not self.person:
            gestures = []
        trash = os.path.join(self.data_dir + "_deleted", time.strftime('%Y%m%d_%H%M%S'), self.person)
        moved = 0
        for g in gestures:
            src = os.path.join(self.data_dir, self.person, g)
            files = [n for n in os.listdir(src) if n.endswith('.csv')] if os.path.isdir(src) else []
            if files:
                os.makedirs(os.path.join(trash, g), exist_ok=True)
            for name in files:
                shutil.move(os.path.join(src, name), os.path.join(trash, g, name))
                moved += 1
        self.saved_paths = [p for p in self.saved_paths if os.path.exists(p)]
        what = f"all of {self.person}'s gestures" if label == 'all' else f"{self.person}'s {GESTURE_NAMES.get(label, label)}"
        self.broadcast('notice', {'level': 'info', 'text': f'Reset {what}: moved {moved} recordings to {os.path.relpath(trash, os.path.dirname(GESTURE_DIR))}.'})
        self.broadcast('state', self.state())

    def start_training(self) -> bool:
        with self.lock:
            if self.training:
                return False
            self.training = True
        self.broadcast('training', {})
        threading.Thread(target=self._train, daemon=True).start()
        return True

    def _train(self):
        try:
            trainer.train(self.data_dir, self.model_dir, quiet=True)
            self.load_model()
            self.broadcast('notice', {'level': 'good', 'text': f"Model trained - test accuracy {self.metrics['accuracy'] * 100:.1f}%."})
        except Exception as e:
            self.broadcast('notice', {'level': 'error', 'text': f'Training failed: {e}'})
        finally:
            self.training = False
            self.broadcast('state', self.state())


class SerialLink(threading.Thread):
    """Reads the micro:bit serial stream, reconnecting automatically if it is unplugged."""

    def __init__(self, hub: Hub, port: str, baud: int):
        super().__init__(daemon=True)
        self.hub = hub
        self.port = port
        self.baud = baud
        self.ser = None
        self.write_lock = threading.Lock()

    def run(self):
        if not HAS_SERIAL:
            self.hub.set_link_status(False, None, 'pyserial is not installed (pip install pyserial)')
            return
        while True:
            port = self.port or find_microbit_port()
            if not port:
                self.hub.set_link_status(False, None, 'No micro:bit found - plug it in via USB')
                time.sleep(2)
                continue
            try:
                with serial.Serial(port, baudrate=self.baud, timeout=1.0) as ser:
                    self.ser = ser
                    ser.reset_input_buffer()
                    self.hub.set_link_status(True, port)
                    self.read_loop(ser)
            except (serial.SerialException, OSError) as e:
                self.hub.set_link_status(False, port, str(e))
            finally:
                self.ser = None
            time.sleep(2)

    def read_loop(self, ser):
        recording = None  # list of samples while button A is held
        started = 0.0
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            if line.startswith('PRED:'):
                try:
                    index, percent = (int(v) for v in line[5:].split(','))
                except ValueError:
                    continue
                self.hub.on_board_prediction(index, percent)
                continue
            if line == 'START':
                recording = []
                started = time.time()
                continue
            if line == 'END':
                if recording is not None:
                    self.hub.on_recording(recording, time.time() - started)
                recording = None
                continue
            parts = line.split(',')
            if len(parts) == 8:
                # Activity recording program (microbit/makecode_program.js), sent while recording:
                # sampleCount,x,y,z,strength,activityLevel,averageActivity,peakActivity
                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    continue
                self.hub.on_device_stats(*values[:1], *values[4:])
                self.hub.set_program('activity_recorder')
                parts = parts[1:4]
            elif len(parts) == 3:
                self.hub.set_program('gesture')
            else:
                continue
            try:
                x, y, z = (float(p) for p in parts)
            except ValueError:
                continue
            if recording is not None:
                recording.append((x, y, z))
            self.hub.on_sample(x, y, z, recording is not None)

    def write_line(self, text: str):
        with self.write_lock:
            if self.ser is not None:
                try:
                    self.ser.write((text + '\n').encode('utf-8'))
                except (serial.SerialException, OSError):
                    pass


class DemoLink(threading.Thread):
    """Simulated micro:bit: idle readings with a synthetic gesture every couple of seconds."""

    def __init__(self, hub: Hub):
        super().__init__(daemon=True)
        self.hub = hub

    def run(self):
        rng = np.random.default_rng()
        self.hub.set_link_status(True, 'DEMO')
        while True:
            for _ in range(int(rng.integers(35, 60))):
                self.hub.on_sample(*simulator.idle_sample(rng), False)
                time.sleep(simulator.SAMPLE_PERIOD)
            gesture = self.hub.label if self.hub.mode == 'record' else str(rng.choice(simulator.SIMULATED))
            if gesture not in simulator.SIMULATED:
                continue  # digits aren't simulated
            rec = simulator.generate(gesture, rng)
            for x, y, z in rec:
                self.hub.on_sample(x, y, z, True)
                time.sleep(simulator.SAMPLE_PERIOD)
            self.hub.on_recording(rec.tolist(), len(rec) * simulator.SAMPLE_PERIOD, truth=gesture)

    def write_line(self, text: str):
        pass


class QuietServer(ThreadingHTTPServer):
    """Doesn't print a traceback when a browser tab closes mid-request."""

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    hub: Hub = None

    def log_message(self, format, *args):
        pass  # keep the console quiet

    def send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(DASHBOARD_PATH, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/state':
            self.send_json(self.hub.state())
        elif self.path == '/events':
            self.stream_events()
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except json.JSONDecodeError:
            body = {}
        if self.path == '/api/mode':
            self.hub.set_mode(body.get('mode'))
        elif self.path == '/api/label':
            self.hub.set_label(body.get('label'))
        elif self.path == '/api/view':
            self.hub.set_view(body.get('view'))
        elif self.path == '/api/gestures/add':
            self.hub.add_gesture(body.get('name'))
        elif self.path == '/api/gestures/remove':
            self.hub.remove_gesture(str(body.get('id', '')))
        elif self.path == '/api/person':
            self.hub.set_person(body.get('person'))
        elif self.path == '/api/reset':
            self.hub.reset(body.get('label'))
        elif self.path == '/api/undo':
            self.hub.undo()
        elif self.path == '/api/train':
            if not self.hub.start_training():
                self.send_json({'ok': False, 'error': 'Already training'}, 409)
                return
        else:
            self.send_error(404)
            return
        self.send_json({'ok': True})

    def stream_events(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        q = self.hub.subscribe()
        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    msg = b": ping\n\n"  # keep the connection alive
                self.wfile.write(msg)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            self.hub.unsubscribe(q)


def main():
    parser = argparse.ArgumentParser(description="Live dashboard for the micro:bit gesture classifier and activity tracker.")
    parser.add_argument('--port', type=str, default=None, help="Micro:bit Serial COM port (e.g. COM3). Auto-detected if omitted.")
    parser.add_argument('--baud', type=int, default=115200, help="Serial baud rate (default 115200).")
    parser.add_argument('--demo', action='store_true', help="Simulate gestures without hardware (uses data/sim and models/sim).")
    parser.add_argument('--http_port', type=int, default=8000, help="Dashboard port (default 8000).")
    parser.add_argument('--no_browser', action='store_true', help="Don't open the dashboard in a browser.")
    args = parser.parse_args()

    if args.demo:
        data_dir = os.path.join(GESTURE_DIR, "data", "sim")
        model_dir = os.path.join(GESTURE_DIR, "models", "sim")
        if not os.path.isdir(data_dir):
            print("Demo mode: generating simulated recordings and training a demo model...")
            simulator.seed_dataset(data_dir)
            trainer.train(data_dir, model_dir, quiet=True)
    else:
        data_dir = trainer.DATA_DIR
        model_dir = trainer.MODEL_DIR

    hub = Hub(data_dir, model_dir, args.demo)
    hub.start_activity_eval()
    hub.link = DemoLink(hub) if args.demo else SerialLink(hub, args.port, args.baud)
    hub.link.start()

    Handler.hub = hub
    server = QuietServer(('127.0.0.1', args.http_port), Handler)
    server.daemon_threads = True
    url = f"http://localhost:{args.http_port}"
    print(f"micro:bit dashboard running at {url}" + (" (DEMO - simulated data)" if args.demo else ""))
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
        server.server_close()


if __name__ == '__main__':
    main()
