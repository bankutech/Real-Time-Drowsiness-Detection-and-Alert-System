# Model Performance & Real-Time Benchmark Comparison

| Model | Accuracy (%) | Precision (%) | Recall (%) | Macro F1 (%) | ROC AUC | Latency (ms) | Throughput (FPS) | Size (KB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bayesian Logistic | 24.42 | 19.61 | 18.32 | 12.24 | 0.4857 | 0.0006 | 1613321.9 | 1.9 |
| SVM (RBF) | 29.78 | 11.96 | 22.33 | 12.06 | 0.5103 | 0.0557 | 17948.5 | 28.2 |
| Stacking Ensemble | 24.09 | 27.76 | 18.07 | 11.35 | 0.4994 | 0.0908 | 11015.5 | 330.4 |
| SVM (Linear) | 21.36 | 8.22 | 16.02 | 10.87 | 0.4733 | 0.0091 | 110120.3 | 13.0 |
| Decision Tree | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.0005 | 1989752.3 | 4.6 |
| Random Forest | 0.0 | 0.0 | 0.0 | 0.0 | 0.4742 | 0.0245 | 40850.9 | 480.8 |
| AdaBoost | 0.0 | 0.0 | 0.0 | 0.0 | 0.5153 | 0.019 | 52568.8 | 29.1 |
