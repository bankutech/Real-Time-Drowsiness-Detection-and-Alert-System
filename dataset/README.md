# 📊 Dataset & Physiological Feature Engineering

This directory contains the raw and preprocessed physiological feature datasets used to train, evaluate, and benchmark all Machine Learning, Markovian, and Regression models in the **Real-Time Driver Drowsiness Detection and Alert System**.

---

## 📁 Directory Structure

```
dataset/
├── driver_drowsiness_dataset.csv   # Raw baseline dataset (4,001 frames across all 4 driver states)
└── cleaned_features.csv            # Cleaned, imputed, outlier-capped & standardized dataset
```

---

## 🧬 Feature Dictionary & Physiological Metrics

The dataset captures 11 primary physiological, ocular, facial, and head-pose telemetry features extracted across time windows:

| Feature Name | Type | Physical Range / Unit | Physiological Description & Importance |
| :--- | :---: | :---: | :--- |
| `ear` | `float64` | $[0.0, 0.5]$ | **Eye Aspect Ratio**: Geometric ratio between vertical eye landmark distances and horizontal eye width. Rapidly drops during eyelid droop and eye closures. |
| `mar` | `float64` | $[0.0, 1.2]$ | **Mouth Aspect Ratio**: Ratio of vertical lip opening distance to horizontal mouth width. Spikes during yawning maneuvers. |
| `blink_duration`| `float64` | $[0.05, 3.0]\text{ s}$ | Duration in seconds of continuous eyelid closure. Normal blinks $\approx 0.1-0.3\text{s}$; microsleeps $\ge 0.5-1.5\text{s}$. |
| `blink_rate` | `float64` | $[0, 60]\text{ bpm}$ | Number of completed blinks per minute calculated over a rolling temporal window. |
| `yawn_freq` | `float64` | $[0.0, 2.0]\text{ yawns/min}$ | Rolling frequency of detected yawn events exceeding the dynamic MAR threshold ($\ge 0.60$). |
| `eye_closure_dur` | `float64` | $[0.0, 5.0]\text{ s}$ | Maximum continuous duration of eye closure in the latest rolling observation buffer. |
| `head_pitch` | `float64` | $[-90^\circ, +90^\circ]$ | Nodding angle of the head (sagittal plane) computed via OpenCV `solvePnP` against canonical 3D facial points. Indicates nodding off. |
| `head_yaw` | `float64` | $[-90^\circ, +90^\circ]$ | Lateral turning angle of the head (transverse plane). Captures distraction or driver disorientation. |
| `head_roll` | `float64` | $[-90^\circ, +90^\circ]$ | Tilting angle of the head (coronal plane). Indicates loss of neck muscle tone during severe fatigue. |
| `face_angle` | `float64` | $[0^\circ, 90^\circ]$ | Composite magnitude of 3D angular head deviation from forward road-facing alignment: $\sqrt{\text{pitch}^2 + \text{yaw}^2 + \text{roll}^2}$. |
| `perclos` | `float64` | $[0.0, 1.0]$ | **Percentage of Eye Closure**: The gold standard in automotive safety. Measures proportion of time eyes are $\ge 80\%$ closed over a 60-frame ($2.0\text{s}$) window. |
| `fatigue_score` | `float64` | $[0.0, 100.0]$ | Continuous ground-truth driver fatigue index derived from multimodal physiological strain. Target variable for continuous regression. |
| `state` | `string` | Categorical | Target classification label: `Alert` (0), `Slightly Drowsy` (1), `Drowsy` (2), `Sleep` (3). |

---

## 🏷️ Class Taxonomy & Distribution

| Class ID | State Label | Sample Count | Percentage | Physiological State Profile |
| :---: | :--- | :---: | :---: | :--- |
| `0` | **Alert** | 1,401 | 35.0% | $EAR > 0.25$, $PERCLOS < 0.15$, upright head pose, normal blink rate ($10-20\text{ bpm}$). |
| `1` | **Slightly Drowsy** | 1,000 | 25.0% | $0.20 \le EAR \le 0.25$, elevated blink rate ($20-35\text{ bpm}$), mild head inclination. |
| `2` | **Drowsy** | 1,000 | 25.0% | $EAR < 0.20$, frequent yawning ($MAR > 0.60$), PERCLOS $> 0.35$, head drooping ($pitch > 15^\circ$). |
| `3` | **Sleep** | 600 | 15.0% | $EAR \le 0.12$, continuous eye closure $> 1.5\text{s}$, PERCLOS $> 0.70$, severe head slump. |
| **Total** | | **4,001** | **100.0%** | **Stratified $80/20$ Split $\rightarrow$ 3,200 Train / 801 Test** |

---

## ⚙️ Data Preprocessing & Cleaning Pipeline

The raw data is processed through `src/preprocessing.py` implementing Unit 1 rigorous standards:

1. **Missing Value Imputation**: Median strategy across numerical columns via `sklearn.impute.SimpleImputer` to prevent distribution skewing.
2. **Sensor Outlier Capping**: Robust Interquartile Range (IQR) thresholding:
   $$\text{IQR} = Q_3 - Q_1, \quad [x_{\text{lower}}, x_{\text{upper}}] = [Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$$
3. **Stratified Splitting**: 80/20 Train-Test split retaining identical class proportions across both partitions (`random_state=42`).
4. **Standard Feature Scaling**: Zero-mean, unit-variance standardization:
   $$z = \frac{x - \mu}{\sigma}$$
   (Scaler fitted exclusively on training set to prevent data leakage).

---

## 📈 Exploratory Data Analysis (EDA)

All exploratory statistical charts, correlation heatmaps, feature histograms, and boxplots generated from this dataset are saved in [`outputs/eda/`](../outputs/eda/).
