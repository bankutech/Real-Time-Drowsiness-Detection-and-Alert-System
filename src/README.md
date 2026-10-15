# 🛠️ Source Architecture & Module Documentation

This directory contains the modular Python source code for the **Real-Time Driver Drowsiness Detection and Alert System**.

---

## 🏛️ Architectural Overview & Data Flow

```
                                  [ Video Frame / Camera 0 ]
                                               │
                                               ▼
                              ┌───────────────────────────────────┐
                              │     feature_extraction.py         │
                              │  MediaPipe 478 Dense FaceMesh     │
                              │  EAR, MAR, PERCLOS, solvePnP Pose │
                              └─────────────────┬─────────────────┘
                                                │ (11-D Raw Features)
                                                ▼
                              ┌───────────────────────────────────┐
                              │        preprocessing.py           │
                              │  Median Imputation & Scaler       │
                              └─────────────────┬─────────────────┘
                                                │ (11-D Scaled Vector)
                                                ▼
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         │                                      │                                      │
         ▼                                      ▼                                      ▼
┌──────────────────┐                  ┌────────────────────┐                 ┌──────────────────┐
│linear_regression │                  │ bayesian_logistic  │                 │  ensemble.py     │
│Continuous Score  │                  │  svm_classifier    │                 │  random_forest   │
│[0.0, 100.0]      │                  │  decision_tree     │                 │  adaboost        │
└────────┬─────────┘                  └─────────┬──────────┘                 └────────┬─────────┘
         │                                      │                                     │
         │ Fatigue Index                        │ Discrete Class Emission Probabilities
         │                                      ▼
         │                            ┌────────────────────┐
         │                            │      hmm.py        │
         │                            │  Bayesian Forward  │
         │                            │  Belief Filter     │
         │                            └─────────┬──────────┘
         │                                      │ Filtered State ID & Smoothed Belief
         └───────────────────┬──────────────────┘
                             ▼
              ┌─────────────────────────────┐
              │    realtime_detection.py    │
              │  - Head-Up Display (HUD)    │
              │  - 3-Tier Alert Dispatch    │
              │    (alert_system.py)        │
              └──────────────┬──────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│   OpenCV Window / CLI   │       │   Web Dashboard (app.py)│
│       (main.py)         │       │   MJPEG & Telemetry API │
└─────────────────────────┘       └─────────────────────────┘
```

---

## 📂 Source File Directory & Descriptions

### 1. Core Framework & Utilities
- **[`config.py`](config.py)**: Central configuration module. Defines project directory paths, classification taxonomy, UI BGR color palettes, MediaPipe landmark indices, alerting thresholds, and HMM hyperparameters.
- **[`utils.py`](utils.py)**: System utility layer. Implements thread-safe logging, JSON/CSV I/O, mathematical helpers, model serialization wrappers (`save_model`, `load_model`), and sound synthesis utilities.

### 2. Unit 1: Preprocessing, EDA & Feature Extraction
- **[`preprocessing.py`](preprocessing.py)**: Implements missing data imputation (`SimpleImputer`), IQR sensor noise capping, stratified 80/20 train/test splitting, and `StandardScaler` normalization.
- **[`statistics_analysis.py`](statistics_analysis.py)**: Generates comprehensive exploratory data analysis figures, including class balance histograms, correlation heatmaps, feature boxplots, and summary statistics CSV.
- **[`feature_extraction.py`](feature_extraction.py)**: Real-time computer vision engine powered by MediaPipe FaceLandmarker. Extracts 478 dense 3D landmarks, calculates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), rolling 60-frame PERCLOS, and 3D Head Pose (Pitch, Yaw, Roll via OpenCV `solvePnP`).

### 3. Unit 2: Regression, Bayesian Logistic & Support Vector Machines
- **[`linear_regression.py`](linear_regression.py)**: Trains Ridge and Lasso L1/L2 regularized linear models to predict a continuous Fatigue Index score $[0, 100]$. Generates residual distribution plots and Q-Q normality charts.
- **[`bayesian_logistic.py`](bayesian_logistic.py)**: Implements Multi-Class Bayesian Logistic Regression with Gaussian weight priors $\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ and MAP optimization. Outputs calibrated posterior probabilities and Shannon entropy uncertainty scores.
- **[`svm_classifier.py`](svm_classifier.py)**: Implements Linear and Non-Linear RBF Kernel Support Vector Machines. Generates 2D decision boundary contour projections.

### 4. Unit 3: Dimensionality Reduction & Unsupervised Clustering
- **[`pca.py`](pca.py)**: Computes Principal Component Analysis, Scree plots with cumulative variance curves, and 2D/3D subspace projections.
- **[`kmeans.py`](kmeans.py)**: Implements K-Means clustering with dual-axis Inertia (Elbow method) and Silhouette coefficient optimization.
- **[`gmm.py`](gmm.py)**: Implements Gaussian Mixture Models with Expectation-Maximization (EM) optimization, AIC/BIC hyperparameter selection, and covariance error ellipse visualizations.
- **[`hierarchical.py`](hierarchical.py)**: Implements Agglomerative Hierarchical clustering with Ward linkage, dendrogram visualization, and cophenetic correlation scoring.

### 5. Unit 4: Pure NumPy Hidden Markov Models (Temporal Modeling)
- **[`hmm.py`](hmm.py)**: Implements a first-order Hidden Markov Model written from scratch using pure NumPy. Features:
  - Transition matrix $A$ with high self-persistence ($P \ge 0.88$).
  - Stationary prior distribution $\pi$ via Markov eigenvector solving.
  - Viterbi dynamic programming for optimal sequence decoding.
  - Streaming Bayesian Forward belief tracking for real-time video debouncing.

### 6. Unit 5: Tree-Based Architectures & Ensemble Learning
- **[`decision_tree.py`](decision_tree.py)**: Implements CART Decision Tree with Gini impurity splitting, cost-complexity pruning, and high-resolution tree visualization.
- **[`random_forest.py`](random_forest.py)**: Implements Random Forest Ensemble with Out-of-Bag (OOB) error monitoring and Mean Decrease Impurity (MDI) feature ranking.
- **[`adaboost.py`](adaboost.py)**: Implements AdaBoost with stagewise exponential loss decay visualization and dynamic sample weight adaptation.
- **[`ensemble.py`](ensemble.py)**: Implements Soft Voting Probability Aggregators and Stacking Ensemble meta-classifiers.

### 7. Evaluation, Alerting & Live Execution
- **[`evaluation.py`](evaluation.py)**: Unified benchmarking engine. Compares all 8 classifiers across Accuracy, Macro F1, Precision, Recall, ROC AUC, per-frame latency, and FPS throughput. Exports unified comparison tables and multi-model ROC charts.
- **[`alert_system.py`](alert_system.py)**: 3-tier escalation engine (Level 0: Safe, Level 1: Warning, Level 2: Critical Alarm). Synthesizes pure sine audio tones (1000 Hz / 2500 Hz) with sub-10ms response time and logs timestamped events to CSV/JSON.
- **[`realtime_detection.py`](realtime_detection.py)**: Master real-time pipeline integrating webcam video capture, landmark extraction, multi-model inference, HMM smoothing, and rich Head-Up Display (HUD) rendering.
