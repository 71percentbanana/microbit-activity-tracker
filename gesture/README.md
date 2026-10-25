# BBC micro:bit Gesture Classifier (Random Forest)

Recognises hand gestures from the **BBC micro:bit**'s 3-axis $(X, Y, Z)$ accelerometer using a **Random Forest Classifier** trained on your own recordings, with a live web dashboard for recording, training and prediction.

This project is separate from the activity tracker in the parent folder and shares no code or data with it. Both projects share one live dashboard in [`../dashboard`](../dashboard), which has a **Gestures** and an **Activity** view.

---

## ✋ Gestures (21)
**Movements**
- `Swipe left` / `Swipe right` — sweep the board sideways
- `Up` / `Down` — lift the board up quickly / push it down
- `Push` / `Pull` — push the board forward, away from you / pull it back towards you
- `Circle` — draw a circle in the air

**Shakes, tilts & taps**
- `Shake` — shake the board rapidly back and forth
- `Tilt left` / `Tilt right` — tip the board onto its left / right edge
- `Tap` / `Double tap` — tap the board once / twice against your other palm

**Numbers 1–9** — write the digit in the air in front of you, as if on a whiteboard

**Custom gestures** — click **+ Add gesture** on the dashboard, name it (for example *Zigzag*), record ~30, then **Train model**. Custom gestures are saved in [`custom_gestures.json`](custom_gestures.json), so teammates get them with `git pull`; the micro:bit shows a star when it recognises one. **Remove** takes a custom gesture off the list but keeps its recordings, which come back if you add it again.

Start every gesture with the board flat, face up (LEDs facing the ceiling), the same way every time.

The micro:bit LEDs show the result: arrows for swipes, up/down and tilts, a small / big square for push / pull, one / two bars for tap / double tap, and the digit itself for numbers.

---

## 📁 Project Structure

```
gesture/
├── microbit/
│   ├── gesture_makecode.js   # MakeCode program: streams x,y,z, hold A to record, shows result on LEDs
│   └── gesture_classifier.hex # Same program, compiled - drag onto the MICROBIT drive to flash
├── data/raw/<person>/<gesture>/*.csv  # Recordings, one folder per person (one CSV per gesture performance)
├── models/                   # Created by training (not in git - click Train model after pulling)
│   ├── gesture_rf.joblib     # Trained Random Forest model
│   ├── gesture_metrics.json  # Accuracy, precision, recall, F1, confusion matrix
│   └── confusion_matrix.png  # Confusion matrix plot for the report
├── gesture_features.py       # Feature extraction (statistics + time-normalised shape)
├── train.py                  # Trains and evaluates the model
├── simulator.py              # Synthetic gestures for --demo mode
└── requirements.txt
```

---

## ⚙️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Flash the micro:bit
Drag [`microbit/gesture_classifier.hex`](microbit/gesture_classifier.hex) onto the **MICROBIT** drive, or build it in MakeCode:
1. Open [makecode.microbit.org](https://makecode.microbit.org/) and create a **New Project**.
2. Switch to the **JavaScript** tab at the top.
3. Paste the contents of [`microbit/gesture_makecode.js`](microbit/gesture_makecode.js) (MakeCode converts it to blocks if you switch back).
4. Click **Download** to flash the `.hex` file onto your micro:bit. It shows `READY` when it starts.

### 3. Start the Dashboard
Close every MakeCode tab (or click **Disconnect** in MakeCode) and any other program using the micro:bit's COM port — two programs reading the same micro:bit each get only part of the data. Then run:
```bash
cd ../dashboard
python server.py
```
The dashboard opens at http://localhost:8000 and auto-detects the micro:bit (or use `--port COM7`).

### 4. Record Training Data
1. In the **Gestures** view, type your first name in **Recording as** (recordings are saved to `data/raw/<your name>/`).
2. Click **Record** and pick a gesture (keys `[` and `]` move to the previous / next one).
3. **Hold button A**, perform the gesture, **release A**. The micro:bit shows ✓ and the recording is saved.
4. Repeat ~30 times per gesture. Use **Undo last recording** for mistakes.

### 5. Train
Click **Train model** on the dashboard, or run:
```bash
python train.py
```
This prints test accuracy, cross-validation accuracy and — once two or more people have recorded — the accuracy on each person when the model is trained without them. It saves the model, metrics and `models/confusion_matrix.png`.

### 6. Predict
Click **Predict**, hold A, do a gesture, release. The dashboard shows the gesture, its confidence and the probability of every class, and the micro:bit shows an arrow / icon. Predictions below 30% confidence show `?` (set by `CONFIDENCE_THRESHOLD` in `dashboard/server.py`).

### Demo Without Hardware
```bash
cd ../dashboard
python server.py --demo
```
Uses simulated gestures (kept in `data/sim/` and `models/sim/`, never mixed with real recordings). Demo accuracy is meaningless — use it only to try the dashboard.

---

## 👥 Recording as a Teammate

Everyone records on their own laptop and shares recordings through GitHub. Each person's recordings live in their own folder, so nobody overwrites anyone else's.

**Setup (once)**
1. Install [Python 3.10+](https://www.python.org/downloads/) (tick *Add Python to PATH*) and [Git](https://git-scm.com/downloads).
2. Get the project and install the libraries:
   ```bash
   git clone https://github.com/71percentbanana/microbit-activity-tracker.git
   cd microbit-activity-tracker
   pip install -r requirements.txt
   ```
3. Plug in the micro:bit. It already has the program; if it doesn't show `READY` when plugged in, drag [`microbit/gesture_classifier.hex`](microbit/gesture_classifier.hex) onto the **MICROBIT** drive.

**Record**
1. Close any MakeCode tabs, then run `python dashboard/server.py` (opens http://localhost:8000).
2. Type your first name in **Recording as**.
3. Hold the micro:bit **the same way as everyone else on the team** — the LED display is flipped for the team's grip, so the arrows and numbers should read the right way up for you. Recordings made with a different grip won't match.
4. Record ~30 of each gesture, then click **Train model** to see your results.

**Share**
```bash
git pull
git add gesture/data/raw/<your name>
git commit -m "Add <your name>'s gesture recordings"
git push
```
Anyone can then `git pull`, start the dashboard and click **Train model**: the evaluation shows **Accuracy on a new person** and a **By person** table — the model trained without you, tested on your recordings.

---

## 🧠 How It Works

- **Sampling:** the micro:bit sends `x,y,z` (milli-g) about 20–25 times per second, plus `START` / `END` when button A is pressed and released, so each recording contains exactly one gesture.
- **Features (96 per recording):** length; mean, std, min, max, range and the time position of the min/max for X, Y, Z and magnitude; zero-crossing rate per axis; and each axis resampled to 16 evenly spaced points. The resampled shape captures the *order* of movements, which is what separates left from right and up from down.
- **Model:** `RandomForestClassifier` (200 trees, balanced class weights).
- **Evaluation:** stratified 80/20 split by recording, plus k-fold cross-validation over all recordings. With recordings from two or more people, each person is also left out in turn (trained on everyone else, tested on them) — the realistic accuracy for someone new. The deployed model is then refit on every recording.
