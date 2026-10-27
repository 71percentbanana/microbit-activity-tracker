# BBC micro:bit Human Activity Recognition (HAR) using Random Forest

> This repository holds two separate micro:bit machine-learning projects that share one live dashboard:
> - **Activity tracker** (this folder) — recognises walking, jogging, stairs, sitting and standing.
> - **Gesture classifier** ([`gesture/`](gesture)) — recognises hand gestures trained on your own recordings.
> - **Dashboard** ([`dashboard/`](dashboard)) — `python dashboard/server.py` opens a web page with a **Gestures** and an **Activity** view.

This project trains a **Random Forest Classifier** on the **WISDM Accelerometer Dataset** and provides real-time activity recognition from the 3-axis $(X, Y, Z)$ accelerometer of a **BBC micro:bit**.

---

## 🏃 Detected Activities
- `Walking`
- `Jogging`
- `Upstairs`
- `Downstairs`
- `Sitting`
- `Standing`

---

## 📁 Project Structure

```
ai_project/
├── data/
│   └── features.csv               # Extracted sliding window statistical features
├── models/
│   ├── random_forest_har.joblib   # Trained Random Forest model
│   ├── label_encoder.joblib       # Activity class label encoder
│   ├── model_metadata.json        # Hyperparameters, sampling rate, feature names
│   ├── confusion_matrix.png       # Test set evaluation confusion matrix
│   └── feature_importances.png    # Top feature importances visualization
├── microbit/
│   └── microbit_sender.py         # MicroPython script for BBC micro:bit
├── features.py                    # Feature extraction module (107 statistical, spectral and rhythm features)
├── prepare_data.py                # Preprocesses WISDM raw data into sliding windows
├── train_model.py                 # Trains and evaluates Random Forest model
├── realtime_predict.py            # Serial receiver & real-time live predictor
└── requirements.txt               # Dependencies
```

---

## ⚙️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data & Extract Features
Reads the WISDM dataset, parses continuous streams, creates 2-second windows (40 samples @ 20 Hz, 50% overlap), and extracts 107 features per window: 28 time-domain statistics plus percentiles, skewness, kurtosis, jerk, frequency-spectrum bands, autocorrelation (step period) and axis correlations:
```bash
python prepare_data.py --data_path "C:\Users\alanv\Downloads\WISDM_ar_latest\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt"
```

### 3. Train Random Forest Model
Trains the model on extracted features and outputs metrics + plots:
```bash
python train_model.py
```

Results: **95.7%** on a random 80/20 split of windows and **85.1%** on 8 of the 36 WISDM participants held out of training entirely (76.3% with the 28 basic statistics alone).

To compare the Random Forest with eight other classifiers (Extra Trees, gradient boosting, decision tree, SVM, k-NN, MLP, logistic regression, naive Bayes) on both the activity and gesture tasks, and to sweep the size of the on-board gesture model:
```bash
python compare_models.py --exclude c_b
```
`--exclude c_b` leaves out custom gesture B, which was recorded after the on-board model in the report was exported. Results go to `models/model_comparison.json` and `models/model_comparison.png`.

### 4. Flash micro:bit
You can use either **Microsoft MakeCode** or the **Python Editor**:

#### Option A: Microsoft MakeCode (Recommended)
1. Open [makecode.microbit.org](https://makecode.microbit.org/) and create a **New Project**.
2. Switch to the **JavaScript** tab at the top.
3. Paste the contents of [`microbit/makecode_program.js`](file:///c:/Users/alanv/ai_project/microbit/makecode_program.js).
4. Click **Download** to flash the `.hex` file onto your micro:bit.

#### Option B: MicroPython
1. Open [python.microbit.org](https://python.microbit.org/).
2. Paste the contents of [`microbit/microbit_sender.py`](file:///c:/Users/alanv/ai_project/microbit/microbit_sender.py).
3. Click **Send to micro:bit**.

### 5. Run Real-time Inference
Connect your micro:bit via USB and run:
```bash
python realtime_predict.py
```
*(The script auto-detects the micro:bit serial COM port and performs real-time classification at 20 Hz!)*

To test without hardware plugged in:
```bash
python realtime_predict.py --simulate
```

---

## 📊 Live Dashboard (Activity + Gestures)

Flash [`gesture/microbit/gesture_classifier.hex`](gesture/microbit/gesture_classifier.hex) (drag it onto the **MICROBIT** drive), close any MakeCode tabs, then run:
```bash
python dashboard/server.py
```
The dashboard opens at http://localhost:8000. In the **Activity** view, the last 2 seconds of the micro:bit's readings are classified with this folder's Random Forest twice a second. The view shows the current activity, a 5-minute timeline, time spent in each activity, and the model's evaluation, including accuracy on people the model never saw. The micro:bit LEDs show the activity as well. Wear the micro:bit upright in a front trouser pocket, as the WISDM phones were.
