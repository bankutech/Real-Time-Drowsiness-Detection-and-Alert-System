# 🧪 Test Suite & Verification Framework

This directory contains a complete, automated verification and unit testing suite covering all 9 developmental phases and 5 syllabus curriculum units of the **Real-Time Driver Drowsiness Detection and Alert System**.

---

## 📋 Test Modules Directory

| Test File | Phase / Syllabus Unit | Scope & Verified Subsystems |
| :--- | :---: | :--- |
| `run_all_tests.py` | **Master Runner** | Orchestrates and executes all phase test scripts sequentially with formatted pass/fail summaries. |
| `test_phase1.py` | **Unit 1: Preprocessing & EDA** | Verifies missing value median imputation, IQR outlier capping, stratified 80/20 train/test splitting, StandardScaler zero-mean normalization, and summary statistics generation. |
| `test_phase2.py` | **Unit 1: Facial Landmarks** | Tests MediaPipe 478 Dense FaceMesh initialization, EAR computation, MAR computation, PERCLOS 60-frame rolling window buffer, and solvePnP 3D head pose estimation. |
| `test_phase3.py` | **Unit 2: Regressors & Bayes/SVM** | Verifies Ridge/Lasso continuous fatigue scoring, Bayesian Logistic MAP posteriors and entropy uncertainty, and Linear / RBF Support Vector Machines. |
| `test_phase4.py` | **Unit 3: Unsupervised & PCA** | Tests PCA variance retention ($\ge 95\%$), K-Means clustering and Silhouette scoring, Gaussian Mixture Model EM convergence and AIC/BIC selection, and Hierarchical Agglomerative clustering with dendrogram generation. |
| `test_phase5.py` | **Unit 4: Pure NumPy HMM** | Validates transition matrix row stochasticity, stationary prior distributions $\pi$, Viterbi dynamic programming decoding, and streaming real-time Bayesian forward belief tracking. |
| `test_phase6.py` | **Unit 5: Trees & Ensembles** | Tests Cost-Complexity pruned Decision Trees, Random Forest OOB error tracking, AdaBoost stagewise boosting error decay, Soft Voting, and Stacking Ensemble meta-learner. |
| `test_phase7.py` | **Phase 7: Multi-Model Benchmark** | Runs all 8 classification architectures over identical stratified test samples, computing Accuracy, Macro F1, Precision, Recall, ROC AUC, per-frame latency (ms), and FPS throughput. |
| `test_phase8.py` | **Phase 8: Realtime Detection & Alert** | Validates 3-tier alert escalation logic, temporal blink debouncing, pure sine audio tone synthesizer dispatch (1000 Hz / 2500 Hz), HUD telemetry rendering, and audit log exports. |
| `test_phase9.py` | **Phase 9: CLI & Web Dashboard** | Tests command-line interface arguments, subcommands, REST endpoints (`/api/telemetry`, `/api/models`, `/api/benchmark`), and MJPEG video streaming server. |

---

## 🚀 Running the Tests

### 1. Run Complete Test Suite
To execute all 9 test phases sequentially with aggregated diagnostics:

```bash
python tests/run_all_tests.py
```

### 2. Run via Standard Python `unittest`
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Run Individual Phase Tests
```bash
python tests/test_phase1.py  # Unit 1 Preprocessing
python tests/test_phase2.py  # Unit 1 MediaPipe CV
python tests/test_phase3.py  # Unit 2 Regression, Bayes & SVM
python tests/test_phase4.py  # Unit 3 PCA & Clustering
python tests/test_phase5.py  # Unit 4 HMM & Viterbi
python tests/test_phase6.py  # Unit 5 Trees & Ensembles
python tests/test_phase7.py  # Phase 7 Benchmarking
python tests/test_phase8.py  # Phase 8 Alert & Realtime
python tests/test_phase9.py  # Phase 9 CLI & Web App
```

---

## ✅ Quality Standards & Assertions

Every test module enforces strict tolerances:
- Numerical assertions check $L_2$ error bounds $< 10^{-4}$ where analytical solutions exist.
- Data partitions are verified for zero target leakage between training and testing sets.
- Sub-millisecond latency bounds ($< 1.0\text{ ms}$) are enforced on single-frame inference pipelines.
