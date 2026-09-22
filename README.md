# BBC micro:bit Human Activity Recognition (HAR) using Random Forest

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
├── features.py                    # Feature extraction module (28 statistical features)
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
Reads the WISDM dataset, parses continuous streams, creates 2-second windows (40 samples @ 20 Hz, 50% overlap), and extracts time-domain statistical features:
```bash
python prepare_data.py --data_path "C:\Users\alanv\Downloads\WISDM_ar_latest\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt"
```

### 3. Train Random Forest Model
Trains the model on extracted features and outputs metrics + plots:
```bash
python train_model.py
```

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
