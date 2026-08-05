# 🤖 Machine Learning Models & Serialized Artifacts Registry

This directory contains serialized binary model weights, transformation pipelines, and deep learning landmark assets powering the **Real-Time Driver Drowsiness Detection and Alert System**.

---

## 📦 Model Artifacts Catalog

All models are serialized using `joblib` (or MediaPipe binary graph format) and can be loaded directly in sub-millisecond time.

| File Name | Size | Architecture / Algorithm | Syllabus Unit | Input Shape | Purpose & Functionality |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `face_landmarker.task` | 3.76 MB | MediaPipe Dense Mesh (478 3D Points) | Unit 1 | `(H, W, 3)` RGB Frame | Sub-millisecond landmark extraction for eyes, lips, iris, and rigid head pose. |
| `imputer.joblib` | 0.25 KB | Median Imputer (`SimpleImputer`) | Unit 1 | `(N, 11)` Raw Vector | Fills missing sensor telemetry using historical training medians. |
| `scaler.joblib` | 0.83 KB | StandardScaler ($z = \frac{x-\mu}{\sigma}$) | Unit 1 | `(N, 11)` Raw Vector | Standardizes all 11 features to zero mean and unit variance. |
| `fatigue_regressor.joblib` | 0.96 KB | Ridge / Lasso Linear Regressor | Unit 2 | `(N, 11)` Scaled | Predicts a continuous Fatigue Index score $\in [0, 100]$. |
| `bayesian_logistic.joblib` | 1.93 KB | Bayesian Logistic Regression (MAP) | Unit 2 | `(N, 11)` Scaled | Computes exact class posteriors $P(y=k \mid \mathbf{x})$ and uncertainty entropy. |
| `svm_linear.joblib` | 11.4 KB | Linear Support Vector Machine | Unit 2 | `(N, 11)` Scaled | Fast maximum-margin hyperplane classifier. |
| `svm_rbf.joblib` | 21.9 KB | Non-Linear SVM (RBF Kernel) | Unit 2 | `(N, 11)` Scaled | High-dimensional non-linear decision boundary mapping. |
| `pca.joblib` | 2.78 KB | Principal Component Analysis ($k=5$) | Unit 3 | `(N, 11)` Scaled | Dimensionality reduction capturing $> 95\%$ cumulative variance. |
| `kmeans.joblib` | 14.7 KB | K-Means Clustering ($k=4$) | Unit 3 | `(N, 11)` Scaled | Unsupervised driver cluster partitioning with silhouette validation. |
| `gmm.joblib` | 13.3 KB | Gaussian Mixture Model (EM, 4 Components) | Unit 3 | `(N, 11)` Scaled | Soft probabilistic cluster assignment with AIC/BIC model selection. |
| `hierarchical.joblib` | 125.8 KB | Agglomerative Clustering (Ward Linkage) | Unit 3 | `(N, 11)` Scaled | Hierarchical tree clustering and distance dendrogram mapping. |
| `hmm.joblib` | 1.43 KB | Pure NumPy Hidden Markov Model ($N=4$) | Unit 4 | `(T, 4)` Probabilities | Temporal state transition tracking, Viterbi dynamic decoding, and jitter debouncing. |
| `decision_tree.joblib` | 4.68 KB | Cost-Complexity Pruned Decision Tree | Unit 5 | `(N, 11)` Scaled | Interpretable rule-based hierarchical splitting (Max Depth: 6). |
| `random_forest.joblib` | 424.9 KB | Random Forest Ensemble ($B=100$) | Unit 5 | `(N, 11)` Scaled | High-accuracy bootstrap aggregation with Out-of-Bag (OOB) monitoring. |
| `adaboost.joblib` | 29.8 KB | AdaBoost with SAMME.R ($M=50$) | Unit 5 | `(N, 11)` Scaled | Sequential boosting on hard-to-classify borderline drowsiness instances. |
| `ensemble_voting.joblib` | 283.2 KB | Soft Voting Classifier | Unit 5 | `(N, 11)` Scaled | Weighted probability aggregation across Bayes, SVM, RF, and AdaBoost. |
| `ensemble_stacking.joblib`| 284.3 KB | Stacking Meta-Learner | Unit 5 | `(N, 11)` Scaled | Meta-classifier trained on out-of-fold base model predictions. |

---

## ⚡ Quick Model Loading Example

```python
import joblib
import numpy as np

# 1. Load Preprocessors
scaler = joblib.load("models/scaler.joblib")
imputer = joblib.load("models/imputer.joblib")

# 2. Load Desired Classifier (e.g., Random Forest or Stacking Ensemble)
model = joblib.load("models/random_forest.joblib")

# 3. Load Temporal Filter
hmm = joblib.load("models/hmm.joblib")

# 4. Predict on a new 11-D feature vector
raw_features = np.array([[0.18, 0.65, 0.45, 28.0, 0.4, 0.6, 12.0, 3.0, 1.0, 12.4, 0.38]])
scaled_features = scaler.transform(imputer.transform(raw_features))

# Base classification
probabilities = model.predict_proba(scaled_features)[0]

# Temporal HMM belief update
filtered_state_id, state_distribution = hmm.update_belief(probabilities)
print(f"Detected State: {['Alert', 'Slightly Drowsy', 'Drowsy', 'Sleep'][filtered_state_id]}")
```

---

## 🔄 Retraining & Exporting

To retrain and regenerate all models from scratch:

```bash
python main.py --mode train
```
