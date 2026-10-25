"""
Real-time micro:bit Activity Recognition Receiver & Classifier.
Reads 3-axis accelerometer data from micro:bit over USB Serial, scales milli-g to m/s²,
maintains a sliding window buffer, and performs real-time Random Forest inference.
"""

import os
import sys
import time
import argparse
from collections import deque
import numpy as np
import joblib

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

from features import extract_window_features

# Conversion constant: micro:bit milli-g to m/s² (1000 mg = 1g = 9.80665 m/s²)
MG_TO_MS2 = 9.80665 / 1000.0

def find_microbit_port():
    """Attempt to automatically detect connected micro:bit COM port."""
    if not HAS_SERIAL:
        return None
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        mfg = (p.manufacturer or "").lower()
        # micro:bit uses USB vendor ID 0x0D28 (Arm DAPLink); older drivers also name it mbed/micro:bit
        if p.vid == 0x0D28 or "micro:bit" in desc or "mbed" in desc or "micro:bit" in mfg:
            return p.device
    return None

def run_simulation(rf, label_encoder, window_size=40):
    """
    Run simulated accelerometer streaming from test patterns to verify pipeline.
    """
    print("\n--- RUNNING IN SIMULATION MODE (No micro:bit connected) ---")
    print("Generating simulated activity motion patterns...\n")

    patterns = [
        ("Sitting (stationary resting on desk)", lambda t: (np.random.normal(0, 20), np.random.normal(0, 20), np.random.normal(1000, 30))),
        ("Walking (rhythmic low-freq swing)", lambda t: (
            np.sin(t * 3) * 600 + np.random.normal(0, 80),
            np.cos(t * 3) * 900 + np.random.normal(0, 100),
            np.random.normal(300, 150)
        )),
        ("Jogging (high-intensity oscillation)", lambda t: (
            np.sin(t * 7) * 1600 + np.random.normal(0, 250),
            np.cos(t * 7) * 1900 + np.random.normal(0, 300),
            np.random.normal(600, 400)
        )),
        ("Standing (vertical still)", lambda t: (np.random.normal(0, 25), np.random.normal(980, 40), np.random.normal(100, 30))),
    ]

    buffer = deque(maxlen=window_size)

    for pattern_name, generator in patterns:
        print(f"\n>> Simulating physical activity: {pattern_name} for 4 seconds...")
        for step in range(80):  # 80 samples @ 20Hz = 4 seconds
            t = step * 0.05
            x_mg, y_mg, z_mg = generator(t)
            
            # Convert mg to m/s²
            x_ms2 = x_mg * MG_TO_MS2
            y_ms2 = y_mg * MG_TO_MS2
            z_ms2 = z_mg * MG_TO_MS2
            
            buffer.append([x_ms2, y_ms2, z_ms2])

            # Classify every 10 samples (every 0.5s once window is full)
            if len(buffer) == window_size and step % 10 == 0:
                window_arr = np.array(buffer, dtype=np.float32)
                feats = extract_window_features(window_arr)
                probs = rf.predict_proba([feats])[0]
                pred_idx = np.argmax(probs)
                activity = label_encoder.classes_[pred_idx]
                confidence = probs[pred_idx] * 100
                print(f"   [Prediction] -> {activity.upper():<12} ({confidence:5.1f}% confidence) | Accel: ({x_ms2:5.2f}, {y_ms2:5.2f}, {z_ms2:5.2f}) m/s²")
            time.sleep(0.02)  # fast simulation display

    print("\nSimulation test finished successfully.")

def run_serial_stream(port, baudrate, rf, label_encoder, window_size=40, step_eval=10):
    """
    Listen to live serial stream from physical BBC micro:bit and predict activity.
    """
    print(f"\nOpening serial connection on {port} at {baudrate} baud...")
    ser = serial.Serial(port, baudrate=baudrate, timeout=1.0)
    time.sleep(2.0)  # Allow micro:bit serial connection to stabilize
    ser.reset_input_buffer()
    print("Connected! Listening for micro:bit accelerometer packets (x,y,z in mg)...\n")

    buffer = deque(maxlen=window_size)
    sample_counter = 0

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            # Clean line and handle potential MakeCode formats (e.g., '100,200,300' or 'x:100, y:200, z:300')
            cleaned = line.replace('x:', '').replace('y:', '').replace('z:', '').replace(' ', '')
            parts = cleaned.split(',')
            if len(parts) == 8:
                # Recording program format: 'sampleCount,x,y,z,strength,activityLevel,averageActivity,peakActivity'
                parts = parts[1:4]
            elif len(parts) != 3:
                continue

            try:
                # micro:bit raw output is in milli-g
                x_mg = float(parts[0])
                y_mg = float(parts[1])
                z_mg = float(parts[2])

                # Convert to m/s² to match WISDM dataset scaling
                x_ms2 = x_mg * MG_TO_MS2
                y_ms2 = y_mg * MG_TO_MS2
                z_ms2 = z_mg * MG_TO_MS2

                buffer.append([x_ms2, y_ms2, z_ms2])
                sample_counter += 1

                # Once window has filled, evaluate prediction every `step_eval` samples
                if len(buffer) == window_size and (sample_counter % step_eval == 0):
                    window_arr = np.array(buffer, dtype=np.float32)
                    feats = extract_window_features(window_arr)
                    probs = rf.predict_proba([feats])[0]
                    pred_idx = np.argmax(probs)
                    activity = label_encoder.classes_[pred_idx]
                    confidence = probs[pred_idx] * 100

                    # Clean formatted terminal output
                    timestamp = time.strftime('%H:%M:%S')
                    print(f"[{timestamp}] Activity: {activity.upper():<11} "
                          f"| Confidence: {confidence:5.1f}% "
                          f"| Accelerometer: X={x_ms2:5.2f}, Y={y_ms2:5.2f}, Z={z_ms2:5.2f} m/s²")

            except ValueError:
                continue

    except KeyboardInterrupt:
        print("\nStopping real-time inference.")
    finally:
        ser.close()
        print("Serial port closed.")

def main():
    parser = argparse.ArgumentParser(description="Real-time HAR inference with BBC micro:bit.")
    parser.add_argument('--model_dir', type=str, default="models", help="Directory containing trained model artifacts.")
    parser.add_argument('--port', type=str, default=None, help="Micro:bit Serial COM port (e.g. COM3). Auto-detected if omitted.")
    parser.add_argument('--baud', type=int, default=115200, help="Serial baud rate (default 115200).")
    parser.add_argument('--simulate', action='store_true', help="Run simulated activity patterns without hardware.")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "random_forest_har.joblib")
    encoder_path = os.path.join(args.model_dir, "label_encoder.joblib")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print(f"Error: Model files not found in {args.model_dir}. Please run train_model.py first.")
        sys.exit(1)

    print(f"Loading trained Random Forest model from {model_path}...")
    rf = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print(f"Loaded model successfully. Detected classes: {list(label_encoder.classes_)}")

    if args.simulate:
        run_simulation(rf, label_encoder)
        return

    # Auto-detect or use specified port
    port = args.port
    if not port:
        port = find_microbit_port()

    if not port:
        print("\n[!] No BBC micro:bit detected on USB serial ports.")
        print("Tip: You can specify a COM port directly with --port COMx, or run with --simulate to test.")
        choice = input("Would you like to run the simulation test instead? [Y/n]: ").strip().lower()
        if choice in ('', 'y', 'yes'):
            run_simulation(rf, label_encoder)
        else:
            print("Exiting.")
        return

    run_serial_stream(port, args.baud, rf, label_encoder)

if __name__ == '__main__':
    main()
