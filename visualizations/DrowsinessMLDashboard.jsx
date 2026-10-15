import React, { useState, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, LineChart, Line, PieChart, Pie, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend
} from "recharts";

const DASHBOARD_DATA = {"leaderboard": [{"Model": "Bayesian Logistic", "Accuracy (%)": 100.0, "Precision (%)": 100.0, "Recall (%)": 100.0, "Macro F1 (%)": 100.0, "ROC AUC": 1.0, "Latency (ms)": 0.0001, "Throughput (FPS)": 8774274.4, "Size (KB)": 1.9}, {"Model": "SVM (Linear)", "Accuracy (%)": 100.0, "Precision (%)": 100.0, "Recall (%)": 100.0, "Macro F1 (%)": 100.0, "ROC AUC": 1.0, "Latency (ms)": 0.0029, "Throughput (FPS)": 340185.1, "Size (KB)": 12.9}, {"Model": "Stacking Ensemble", "Accuracy (%)": 100.0, "Precision (%)": 100.0, "Recall (%)": 100.0, "Macro F1 (%)": 100.0, "ROC AUC": 1.0, "Latency (ms)": 0.0207, "Throughput (FPS)": 48318.8, "Size (KB)": 326.0}, {"Model": "Random Forest", "Accuracy (%)": 100.0, "Precision (%)": 100.0, "Recall (%)": 100.0, "Macro F1 (%)": 100.0, "ROC AUC": 1.0, "Latency (ms)": 0.0085, "Throughput (FPS)": 117641.5, "Size (KB)": 480.8}, {"Model": "SVM (RBF)", "Accuracy (%)": 99.88, "Precision (%)": 99.91, "Recall (%)": 99.79, "Macro F1 (%)": 99.85, "ROC AUC": 1.0, "Latency (ms)": 0.008, "Throughput (FPS)": 124456.8, "Size (KB)": 28.2}, {"Model": "Decision Tree", "Accuracy (%)": 99.13, "Precision (%)": 99.21, "Recall (%)": 99.2, "Macro F1 (%)": 99.2, "ROC AUC": 0.9999, "Latency (ms)": 0.0001, "Throughput (FPS)": 8309670.0, "Size (KB)": 4.6}, {"Model": "AdaBoost", "Accuracy (%)": 97.63, "Precision (%)": 97.81, "Recall (%)": 98.19, "Macro F1 (%)": 97.93, "ROC AUC": 0.9973, "Latency (ms)": 0.0058, "Throughput (FPS)": 173029.5, "Size (KB)": 29.1}, {"Model": "HMM (Viterbi)", "Accuracy (%)": 96.5, "Precision (%)": 96.75, "Recall (%)": 96.69, "Macro F1 (%)": 96.71, "ROC AUC": 1.0, "Latency (ms)": 0.0094, "Throughput (FPS)": 106322.0, "Size (KB)": 1.5}], "classDist": {"counts": {"Alert": 1401, "Slightly Drowsy": 1000, "Drowsy": 1000, "Sleep": 600}, "percentages": {"Alert": 35.02, "Slightly Drowsy": 24.99, "Drowsy": 24.99, "Sleep": 15.0}, "total_samples": 4001}, "stateStats": [{"state": "Alert", "ear": 0.319, "mar": 0.282, "perclos": 0.041, "fatigue_score": 11.773, "blink_rate": 14.025, "yawn_freq": 0.055, "head_pitch": 2.058, "head_yaw": 0.077, "head_roll": -0.001}, {"state": "Slightly Drowsy", "ear": 0.251, "mar": 0.358, "perclos": 0.178, "fatigue_score": 40.589, "blink_rate": 22.008, "yawn_freq": 0.199, "head_pitch": 7.0, "head_yaw": 3.925, "head_roll": 3.1}, {"state": "Drowsy", "ear": 0.202, "mar": 0.542, "perclos": 0.433, "fatigue_score": 75.716, "blink_rate": 27.922, "yawn_freq": 0.545, "head_pitch": 17.747, "head_yaw": 9.706, "head_roll": 8.073}, {"state": "Sleep", "ear": 0.132, "mar": 0.383, "perclos": 0.835, "fatigue_score": 99.665, "blink_rate": 6.32, "yawn_freq": 0.152, "head_pitch": 29.86, "head_yaw": 13.552, "head_roll": 14.813}], "scatter": [{"ear": 0.2953, "mar": 0.2748, "perclos": 0.0291, "fatigue_score": 7.1602, "state": "Alert"}, {"ear": 0.3377, "mar": 0.2977, "perclos": 0.0496, "fatigue_score": 15.1479, "state": "Alert"}, {"ear": 0.3455, "mar": 0.3531, "perclos": 0.0401, "fatigue_score": 7.5792, "state": "Alert"}, {"ear": 0.3468, "mar": 0.3627, "perclos": 0.0539, "fatigue_score": 19.94, "state": "Alert"}, {"ear": 0.3233, "mar": 0.1759, "perclos": 0.0522, "fatigue_score": 10.4943, "state": "Alert"}, {"ear": 0.3115, "mar": 0.3158, "perclos": 0.0484, "fatigue_score": 16.4596, "state": "Alert"}, {"ear": 0.3326, "mar": 0.2818, "perclos": 0.0279, "fatigue_score": 6.9608, "state": "Alert"}, {"ear": 0.3072, "mar": 0.2831, "perclos": 0.0319, "fatigue_score": 12.018, "state": "Alert"}, {"ear": 0.2707, "mar": 0.2029, "perclos": 0.0556, "fatigue_score": 18.5865, "state": "Alert"}, {"ear": 0.3089, "mar": 0.2501, "perclos": 0.0467, "fatigue_score": 15.5675, "state": "Alert"}, {"ear": 0.3284, "mar": 0.261, "perclos": 0.0393, "fatigue_score": 10.5154, "state": "Alert"}, {"ear": 0.3746, "mar": 0.3, "perclos": 0.0778, "fatigue_score": 9.5774, "state": "Alert"}, {"ear": 0.3444, "mar": 0.2404, "perclos": 0.068, "fatigue_score": 10.6704, "state": "Alert"}, {"ear": 0.326, "mar": 0.3558, "perclos": 0.0834, "fatigue_score": 12.6817, "state": "Alert"}, {"ear": 0.3525, "mar": 0.3136, "perclos": 0.0634, "fatigue_score": 7.748, "state": "Alert"}, {"ear": 0.33, "mar": 0.2858, "perclos": 0.0441, "fatigue_score": 5.1425, "state": "Alert"}, {"ear": 0.3007, "mar": 0.2663, "perclos": 0.018, "fatigue_score": 9.715, "state": "Alert"}, {"ear": 0.3205, "mar": 0.2575, "perclos": 0.0697, "fatigue_score": 11.5446, "state": "Alert"}, {"ear": 0.3508, "mar": 0.339, "perclos": 0.0571, "fatigue_score": 11.9998, "state": "Alert"}, {"ear": 0.3221, "mar": 0.2476, "perclos": 0.0408, "fatigue_score": 11.4805, "state": "Alert"}, {"ear": 0.3129, "mar": 0.2849, "perclos": 0.0293, "fatigue_score": 16.724, "state": "Alert"}, {"ear": 0.3041, "mar": 0.278, "perclos": 0.0132, "fatigue_score": 4.092, "state": "Alert"}, {"ear": 0.2989, "mar": 0.3046, "perclos": 0.0467, "fatigue_score": 19.8661, "state": "Alert"}, {"ear": 0.281, "mar": 0.3189, "perclos": 0.0544, "fatigue_score": 13.5204, "state": "Alert"}, {"ear": 0.2872, "mar": 0.3311, "perclos": 0.0273, "fatigue_score": 6.5337, "state": "Alert"}, {"ear": 0.3106, "mar": 0.2911, "perclos": 0.0591, "fatigue_score": 9.8618, "state": "Alert"}, {"ear": 0.3396, "mar": 0.2601, "perclos": 0.0414, "fatigue_score": 9.5697, "state": "Alert"}, {"ear": 0.3485, "mar": 0.3195, "perclos": 0.0713, "fatigue_score": 15.5176, "state": "Alert"}, {"ear": 0.2958, "mar": 0.2776, "perclos": 0.0308, "fatigue_score": 14.4456, "state": "Alert"}, {"ear": 0.3204, "mar": 0.2924, "perclos": 0.0735, "fatigue_score": 15.2654, "state": "Alert"}, {"ear": 0.3155, "mar": 0.2573, "perclos": 0.0006, "fatigue_score": 14.3935, "state": "Alert"}, {"ear": 0.3064, "mar": 0.2551, "perclos": 0.0331, "fatigue_score": 8.767, "state": "Alert"}, {"ear": 0.3099, "mar": 0.228, "perclos": 0.0314, "fatigue_score": 10.0707, "state": "Alert"}, {"ear": 0.3119, "mar": 0.284, "perclos": 0.0685, "fatigue_score": 6.6541, "state": "Alert"}, {"ear": 0.3406, "mar": 0.2494, "perclos": 0.0457, "fatigue_score": 10.2136, "state": "Alert"}, {"ear": 0.2935, "mar": 0.2937, "perclos": 0.0301, "fatigue_score": 11.4422, "state": "Alert"}, {"ear": 0.3232, "mar": 0.2721, "perclos": 0.035, "fatigue_score": 14.0601, "state": "Alert"}, {"ear": 0.2975, "mar": 0.3023, "perclos": 0.0297, "fatigue_score": 1.6401, "state": "Alert"}, {"ear": 0.3861, "mar": 0.3009, "perclos": 0.0619, "fatigue_score": 13.8365, "state": "Alert"}, {"ear": 0.3146, "mar": 0.269, "perclos": 0.0195, "fatigue_score": 9.1583, "state": "Alert"}, {"ear": 0.2915, "mar": 0.2761, "perclos": 0.0, "fatigue_score": 9.6797, "state": "Alert"}, {"ear": 0.3298, "mar": 0.309, "perclos": 0.0274, "fatigue_score": 12.4499, "state": "Alert"}, {"ear": 0.3449, "mar": 0.2961, "perclos": 0.0409, "fatigue_score": 7.4052, "state": "Alert"}, {"ear": 0.2459, "mar": 0.2814, "perclos": 0.038, "fatigue_score": 13.1917, "state": "Alert"}, {"ear": 0.343, "mar": 0.3098, "perclos": 0.0759, "fatigue_score": 17.1891, "state": "Alert"}, {"ear": 0.2881, "mar": 0.3167, "perclos": 0.0415, "fatigue_score": 20.4958, "state": "Alert"}, {"ear": 0.358, "mar": 0.3149, "perclos": 0.0307, "fatigue_score": 10.513, "state": "Alert"}, {"ear": 0.2459, "mar": 0.2635, "perclos": 0.0274, "fatigue_score": 9.5377, "state": "Alert"}, {"ear": 0.3493, "mar": 0.2767, "perclos": 0.0373, "fatigue_score": 10.5057, "state": "Alert"}, {"ear": 0.3546, "mar": 0.2329, "perclos": 0.0384, "fatigue_score": 16.0435, "state": "Alert"}, {"ear": 0.2848, "mar": 0.2625, "perclos": 0.0168, "fatigue_score": 15.602, "state": "Alert"}, {"ear": 0.3401, "mar": 0.2787, "perclos": 0.0571, "fatigue_score": 14.5426, "state": "Alert"}, {"ear": 0.3062, "mar": 0.2584, "perclos": 0.0421, "fatigue_score": 14.7793, "state": "Alert"}, {"ear": 0.2973, "mar": 0.339, "perclos": 0.0432, "fatigue_score": 6.222, "state": "Alert"}, {"ear": 0.3487, "mar": 0.292, "perclos": 0.0361, "fatigue_score": 17.6693, "state": "Alert"}, {"ear": 0.2658, "mar": 0.2382, "perclos": 0.0453, "fatigue_score": 7.1666, "state": "Alert"}, {"ear": 0.3539, "mar": 0.2775, "perclos": 0.0, "fatigue_score": 10.1946, "state": "Alert"}, {"ear": 0.2808, "mar": 0.296, "perclos": 0.0392, "fatigue_score": 8.757, "state": "Alert"}, {"ear": 0.3201, "mar": 0.2938, "perclos": 0.0472, "fatigue_score": 12.5018, "state": "Alert"}, {"ear": 0.3179, "mar": 0.2273, "perclos": 0.0025, "fatigue_score": 13.5656, "state": "Alert"}, {"ear": 0.3452, "mar": 0.3245, "perclos": 0.0177, "fatigue_score": 12.37, "state": "Alert"}, {"ear": 0.332, "mar": 0.1881, "perclos": 0.0426, "fatigue_score": 8.4424, "state": "Alert"}, {"ear": 0.3188, "mar": 0.2736, "perclos": 0.0465, "fatigue_score": 9.6106, "state": "Alert"}, {"ear": 0.2906, "mar": 0.3314, "perclos": 0.1861, "fatigue_score": 13.4598, "state": "Alert"}, {"ear": 0.2637, "mar": 0.2918, "perclos": 0.0599, "fatigue_score": 10.4989, "state": "Alert"}, {"ear": 0.3391, "mar": 0.2269, "perclos": 0.0718, "fatigue_score": 0.3135, "state": "Alert"}, {"ear": 0.3643, "mar": 0.3301, "perclos": 0.0245, "fatigue_score": 11.9164, "state": "Alert"}, {"ear": 0.3155, "mar": 0.2763, "perclos": 0.0614, "fatigue_score": 15.1982, "state": "Alert"}, {"ear": 0.326, "mar": 0.2348, "perclos": 0.0401, "fatigue_score": 6.0974, "state": "Alert"}, {"ear": 0.3258, "mar": 0.2529, "perclos": 0.0195, "fatigue_score": 16.3023, "state": "Alert"}, {"ear": 0.3586, "mar": 0.3148, "perclos": 0.0699, "fatigue_score": 17.6275, "state": "Alert"}, {"ear": 0.2985, "mar": 0.3296, "perclos": 0.0804, "fatigue_score": 11.7004, "state": "Alert"}, {"ear": 0.3499, "mar": 0.2585, "perclos": 0.055, "fatigue_score": 19.0909, "state": "Alert"}, {"ear": 0.3376, "mar": 0.2342, "perclos": 0.0499, "fatigue_score": 15.327, "state": "Alert"}, {"ear": 0.3023, "mar": 0.3052, "perclos": 0.0534, "fatigue_score": 5.3446, "state": "Alert"}, {"ear": 0.2951, "mar": 0.2534, "perclos": 0.0599, "fatigue_score": 13.6581, "state": "Alert"}, {"ear": 0.325, "mar": 0.2064, "perclos": 0.0261, "fatigue_score": 11.4725, "state": "Alert"}, {"ear": 0.279, "mar": 0.2709, "perclos": 0.0185, "fatigue_score": 14.1088, "state": "Alert"}, {"ear": 0.3209, "mar": 0.2575, "perclos": 0.0332, "fatigue_score": 0.0, "state": "Alert"}, {"ear": 0.3147, "mar": 0.2552, "perclos": 0.0348, "fatigue_score": 11.1836, "state": "Alert"}, {"ear": 0.3503, "mar": 0.2655, "perclos": 0.1861, "fatigue_score": 2.8534, "state": "Alert"}, {"ear": 0.3559, "mar": 0.3114, "perclos": 0.0079, "fatigue_score": 17.0516, "state": "Alert"}, {"ear": 0.2754, "mar": 0.198, "perclos": 0.0451, "fatigue_score": 7.6057, "state": "Alert"}, {"ear": 0.3407, "mar": 0.2764, "perclos": 0.0289, "fatigue_score": 11.0603, "state": "Alert"}, {"ear": 0.3351, "mar": 0.2656, "perclos": 0.0416, "fatigue_score": 9.5128, "state": "Alert"}, {"ear": 0.298, "mar": 0.2835, "perclos": 0.0458, "fatigue_score": 6.0222, "state": "Alert"}, {"ear": 0.2825, "mar": 0.2814, "perclos": 0.0092, "fatigue_score": 10.4539, "state": "Alert"}, {"ear": 0.3333, "mar": 0.31, "perclos": 0.0125, "fatigue_score": 14.6603, "state": "Alert"}, {"ear": 0.2938, "mar": 0.2966, "perclos": 0.0499, "fatigue_score": 18.1573, "state": "Alert"}, {"ear": 0.3203, "mar": 0.2968, "perclos": 0.0258, "fatigue_score": 9.618, "state": "Alert"}, {"ear": 0.1848, "mar": 0.4786, "perclos": 0.3859, "fatigue_score": 79.1414, "state": "Drowsy"}, {"ear": 0.2008, "mar": 0.6586, "perclos": 0.4144, "fatigue_score": 83.6466, "state": "Drowsy"}, {"ear": 0.2342, "mar": 0.5473, "perclos": 0.4315, "fatigue_score": 74.8965, "state": "Drowsy"}, {"ear": 0.1825, "mar": 0.6295, "perclos": 0.6057, "fatigue_score": 72.0015, "state": "Drowsy"}, {"ear": 0.1631, "mar": 0.4359, "perclos": 0.483, "fatigue_score": 74.8116, "state": "Drowsy"}, {"ear": 0.213, "mar": 0.4384, "perclos": 0.4859, "fatigue_score": 70.5914, "state": "Drowsy"}, {"ear": 0.2169, "mar": 0.5651, "perclos": 0.4479, "fatigue_score": 80.9755, "state": "Drowsy"}, {"ear": 0.199, "mar": 0.5952, "perclos": 0.6593, "fatigue_score": 78.3087, "state": "Drowsy"}, {"ear": 0.1865, "mar": 0.6427, "perclos": 0.3555, "fatigue_score": 76.2033, "state": "Drowsy"}, {"ear": 0.1895, "mar": 0.6258, "perclos": 0.4865, "fatigue_score": 78.352, "state": "Drowsy"}, {"ear": 0.2169, "mar": 0.3931, "perclos": 0.5024, "fatigue_score": 67.7338, "state": "Drowsy"}, {"ear": 0.1884, "mar": 0.564, "perclos": 0.4618, "fatigue_score": 77.6884, "state": "Drowsy"}, {"ear": 0.1692, "mar": 0.4765, "perclos": 0.4171, "fatigue_score": 77.4694, "state": "Drowsy"}, {"ear": 0.1889, "mar": 0.4685, "perclos": 0.4276, "fatigue_score": 79.712, "state": "Drowsy"}, {"ear": 0.2221, "mar": 0.4281, "perclos": 0.3569, "fatigue_score": 70.4222, "state": "Drowsy"}, {"ear": 0.1983, "mar": 0.4654, "perclos": 0.5394, "fatigue_score": 78.7161, "state": "Drowsy"}, {"ear": 0.2287, "mar": 0.6091, "perclos": 0.3301, "fatigue_score": 60.8428, "state": "Drowsy"}, {"ear": 0.2139, "mar": 0.5034, "perclos": 0.4482, "fatigue_score": 79.5046, "state": "Drowsy"}, {"ear": 0.1882, "mar": 0.4348, "perclos": 0.4661, "fatigue_score": 70.5593, "state": "Drowsy"}, {"ear": 0.2091, "mar": 0.4679, "perclos": 0.4278, "fatigue_score": 66.8156, "state": "Drowsy"}, {"ear": 0.1763, "mar": 0.4889, "perclos": 0.5098, "fatigue_score": 72.5421, "state": "Drowsy"}, {"ear": 0.1973, "mar": 0.6416, "perclos": 0.3882, "fatigue_score": 75.5295, "state": "Drowsy"}, {"ear": 0.1905, "mar": 0.5249, "perclos": 0.4489, "fatigue_score": 71.4959, "state": "Drowsy"}, {"ear": 0.1634, "mar": 0.4124, "perclos": 0.4659, "fatigue_score": 71.419, "state": "Drowsy"}, {"ear": 0.2048, "mar": 0.3458, "perclos": 0.4037, "fatigue_score": 86.2158, "state": "Drowsy"}, {"ear": 0.2156, "mar": 0.7511, "perclos": 0.4063, "fatigue_score": 67.1058, "state": "Drowsy"}, {"ear": 0.1784, "mar": 0.6631, "perclos": 0.4013, "fatigue_score": 64.4645, "state": "Drowsy"}, {"ear": 0.1821, "mar": 0.5101, "perclos": 0.4042, "fatigue_score": 77.0816, "state": "Drowsy"}, {"ear": 0.2062, "mar": 0.4545, "perclos": 0.3554, "fatigue_score": 78.5979, "state": "Drowsy"}, {"ear": 0.2459, "mar": 0.665, "perclos": 0.3667, "fatigue_score": 75.4874, "state": "Drowsy"}, {"ear": 0.1782, "mar": 0.4634, "perclos": 0.4917, "fatigue_score": 72.1543, "state": "Drowsy"}, {"ear": 0.2083, "mar": 0.5584, "perclos": 0.5133, "fatigue_score": 72.7847, "state": "Drowsy"}, {"ear": 0.1838, "mar": 0.4417, "perclos": 0.4637, "fatigue_score": 82.7541, "state": "Drowsy"}, {"ear": 0.1987, "mar": 0.7206, "perclos": 0.3, "fatigue_score": 77.0021, "state": "Drowsy"}, {"ear": 0.1962, "mar": 0.477, "perclos": 0.1861, "fatigue_score": 81.484, "state": "Drowsy"}, {"ear": 0.1746, "mar": 0.5253, "perclos": 0.4122, "fatigue_score": 78.736, "state": "Drowsy"}, {"ear": 0.1959, "mar": 0.5088, "perclos": 0.5468, "fatigue_score": 73.2753, "state": "Drowsy"}, {"ear": 0.1919, "mar": 0.5177, "perclos": 0.5665, "fatigue_score": 85.2531, "state": "Drowsy"}, {"ear": 0.2075, "mar": 0.4562, "perclos": 0.3311, "fatigue_score": 77.3479, "state": "Drowsy"}, {"ear": 0.2091, "mar": 0.5653, "perclos": 0.5077, "fatigue_score": 81.1779, "state": "Drowsy"}, {"ear": 0.2176, "mar": 0.7637, "perclos": 0.4487, "fatigue_score": 74.2793, "state": "Drowsy"}, {"ear": 0.2063, "mar": 0.6379, "perclos": 0.54, "fatigue_score": 86.7027, "state": "Drowsy"}, {"ear": 0.1985, "mar": 0.39, "perclos": 0.4504, "fatigue_score": 89.678, "state": "Drowsy"}, {"ear": 0.1882, "mar": 0.6181, "perclos": 0.5039, "fatigue_score": 73.6866, "state": "Drowsy"}, {"ear": 0.1984, "mar": 0.44, "perclos": 0.426, "fatigue_score": 72.1257, "state": "Drowsy"}, {"ear": 0.1797, "mar": 0.4864, "perclos": 0.5144, "fatigue_score": 71.9056, "state": "Drowsy"}, {"ear": 0.1859, "mar": 0.559, "perclos": 0.5272, "fatigue_score": 69.1598, "state": "Drowsy"}, {"ear": 0.2306, "mar": 0.5624, "perclos": 0.3536, "fatigue_score": 78.0108, "state": "Drowsy"}, {"ear": 0.1726, "mar": 0.506, "perclos": 0.4454, "fatigue_score": 62.1756, "state": "Drowsy"}, {"ear": 0.194, "mar": 0.4197, "perclos": 0.3102, "fatigue_score": 80.9185, "state": "Drowsy"}, {"ear": 0.2061, "mar": 0.5137, "perclos": 0.3533, "fatigue_score": 83.6311, "state": "Drowsy"}, {"ear": 0.2149, "mar": 0.343, "perclos": 0.383, "fatigue_score": 73.0506, "state": "Drowsy"}, {"ear": 0.2459, "mar": 0.6598, "perclos": 0.3423, "fatigue_score": 78.8131, "state": "Drowsy"}, {"ear": 0.1912, "mar": 0.7126, "perclos": 0.3119, "fatigue_score": 85.0125, "state": "Drowsy"}, {"ear": 0.1949, "mar": 0.5555, "perclos": 0.3825, "fatigue_score": 75.0239, "state": "Drowsy"}, {"ear": 0.1876, "mar": 0.5183, "perclos": 0.5392, "fatigue_score": 69.138, "state": "Drowsy"}, {"ear": 0.1991, "mar": 0.4906, "perclos": 0.3996, "fatigue_score": 74.1682, "state": "Drowsy"}, {"ear": 0.1423, "mar": 0.6643, "perclos": 0.5105, "fatigue_score": 75.1461, "state": "Drowsy"}, {"ear": 0.167, "mar": 0.7196, "perclos": 0.3828, "fatigue_score": 58.727, "state": "Drowsy"}, {"ear": 0.2132, "mar": 0.5032, "perclos": 0.4039, "fatigue_score": 73.5481, "state": "Drowsy"}, {"ear": 0.1913, "mar": 0.5589, "perclos": 0.4184, "fatigue_score": 82.0181, "state": "Drowsy"}, {"ear": 0.2131, "mar": 0.7728, "perclos": 0.533, "fatigue_score": 71.4613, "state": "Drowsy"}, {"ear": 0.2265, "mar": 0.6392, "perclos": 0.5374, "fatigue_score": 82.5687, "state": "Drowsy"}, {"ear": 0.2016, "mar": 0.4976, "perclos": 0.5332, "fatigue_score": 77.2041, "state": "Drowsy"}, {"ear": 0.1936, "mar": 0.5711, "perclos": 0.4806, "fatigue_score": 80.8512, "state": "Drowsy"}, {"ear": 0.2256, "mar": 0.5988, "perclos": 0.5903, "fatigue_score": 85.6834, "state": "Drowsy"}, {"ear": 0.192, "mar": 0.4578, "perclos": 0.6066, "fatigue_score": 83.7926, "state": "Drowsy"}, {"ear": 0.1758, "mar": 0.6503, "perclos": 0.5048, "fatigue_score": 79.0401, "state": "Drowsy"}, {"ear": 0.1814, "mar": 0.5311, "perclos": 0.4273, "fatigue_score": 72.8591, "state": "Drowsy"}, {"ear": 0.1916, "mar": 0.5845, "perclos": 0.4368, "fatigue_score": 72.5331, "state": "Drowsy"}, {"ear": 0.1634, "mar": 0.6785, "perclos": 0.3622, "fatigue_score": 72.6191, "state": "Drowsy"}, {"ear": 0.2271, "mar": 0.5025, "perclos": 0.5442, "fatigue_score": 72.2178, "state": "Drowsy"}, {"ear": 0.1836, "mar": 0.5474, "perclos": 0.5381, "fatigue_score": 82.1449, "state": "Drowsy"}, {"ear": 0.2033, "mar": 0.501, "perclos": 0.3463, "fatigue_score": 67.0045, "state": "Drowsy"}, {"ear": 0.2176, "mar": 0.6233, "perclos": 0.498, "fatigue_score": 90.0471, "state": "Drowsy"}, {"ear": 0.2281, "mar": 0.664, "perclos": 0.4313, "fatigue_score": 78.872, "state": "Drowsy"}, {"ear": 0.2007, "mar": 0.4501, "perclos": 0.3817, "fatigue_score": 79.0809, "state": "Drowsy"}, {"ear": 0.2039, "mar": 0.6254, "perclos": 0.4104, "fatigue_score": 78.8139, "state": "Drowsy"}, {"ear": 0.1843, "mar": 0.7184, "perclos": 0.4293, "fatigue_score": 72.2897, "state": "Drowsy"}, {"ear": 0.21, "mar": 0.594, "perclos": 0.4731, "fatigue_score": 64.642, "state": "Drowsy"}, {"ear": 0.2, "mar": 0.671, "perclos": 0.4084, "fatigue_score": 74.6436, "state": "Drowsy"}, {"ear": 0.2042, "mar": 0.5298, "perclos": 0.3726, "fatigue_score": 70.9431, "state": "Drowsy"}, {"ear": 0.1599, "mar": 0.6204, "perclos": 0.3492, "fatigue_score": 79.0398, "state": "Drowsy"}, {"ear": 0.2013, "mar": 0.5779, "perclos": 0.5242, "fatigue_score": 73.7312, "state": "Drowsy"}, {"ear": 0.18, "mar": 0.3631, "perclos": 0.322, "fatigue_score": 91.6276, "state": "Drowsy"}, {"ear": 0.1582, "mar": 0.5685, "perclos": 0.4266, "fatigue_score": 84.6627, "state": "Drowsy"}, {"ear": 0.232, "mar": 0.5961, "perclos": 0.4988, "fatigue_score": 77.3621, "state": "Drowsy"}, {"ear": 0.2364, "mar": 0.6288, "perclos": 0.5611, "fatigue_score": 75.3883, "state": "Drowsy"}, {"ear": 0.2157, "mar": 0.6717, "perclos": 0.4277, "fatigue_score": 79.7585, "state": "Drowsy"}, {"ear": 0.2222, "mar": 0.5028, "perclos": 0.3728, "fatigue_score": 79.9936, "state": "Drowsy"}, {"ear": 0.1594, "mar": 0.3194, "perclos": 0.8944, "fatigue_score": 96.7546, "state": "Sleep"}, {"ear": 0.1221, "mar": 0.4491, "perclos": 0.9309, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.0961, "mar": 0.3264, "perclos": 0.8111, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1005, "mar": 0.2811, "perclos": 0.8246, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1385, "mar": 0.2579, "perclos": 0.8515, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1254, "mar": 0.3785, "perclos": 0.9676, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1569, "mar": 0.3674, "perclos": 0.8875, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1169, "mar": 0.4206, "perclos": 0.8036, "fatigue_score": 99.8899, "state": "Sleep"}, {"ear": 0.1122, "mar": 0.4296, "perclos": 0.8222, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1421, "mar": 0.4514, "perclos": 0.8732, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1313, "mar": 0.351, "perclos": 0.7966, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.166, "mar": 0.3812, "perclos": 0.9361, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.125, "mar": 0.3785, "perclos": 0.6973, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1251, "mar": 0.3663, "perclos": 0.824, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1181, "mar": 0.4289, "perclos": 0.1861, "fatigue_score": 98.7682, "state": "Sleep"}, {"ear": 0.1461, "mar": 0.401, "perclos": 0.7856, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1366, "mar": 0.3872, "perclos": 0.7583, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1182, "mar": 0.3192, "perclos": 0.8721, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1223, "mar": 0.3704, "perclos": 0.8721, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1414, "mar": 0.3387, "perclos": 0.8199, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1135, "mar": 0.4213, "perclos": 0.7952, "fatigue_score": 99.2284, "state": "Sleep"}, {"ear": 0.1557, "mar": 0.3752, "perclos": 0.8436, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1304, "mar": 0.4491, "perclos": 0.9752, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1295, "mar": 0.3913, "perclos": 0.8632, "fatigue_score": 96.5523, "state": "Sleep"}, {"ear": 0.1315, "mar": 0.3752, "perclos": 0.882, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1238, "mar": 0.3851, "perclos": 0.7825, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1184, "mar": 0.3124, "perclos": 0.8426, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1194, "mar": 0.3433, "perclos": 0.7844, "fatigue_score": 93.1916, "state": "Sleep"}, {"ear": 0.1257, "mar": 0.3871, "perclos": 0.9589, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1537, "mar": 0.3576, "perclos": 0.985, "fatigue_score": 99.5604, "state": "Sleep"}, {"ear": 0.2459, "mar": 0.4725, "perclos": 0.7286, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1145, "mar": 0.3968, "perclos": 0.9265, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1215, "mar": 0.4121, "perclos": 0.6948, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1224, "mar": 0.3723, "perclos": 0.8847, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1235, "mar": 0.4066, "perclos": 0.7318, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1325, "mar": 0.4343, "perclos": 0.6994, "fatigue_score": 97.7978, "state": "Sleep"}, {"ear": 0.13, "mar": 0.3489, "perclos": 0.8834, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1392, "mar": 0.4228, "perclos": 0.9263, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1448, "mar": 0.4213, "perclos": 0.8671, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1239, "mar": 0.3543, "perclos": 1.0, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1415, "mar": 0.4428, "perclos": 0.8774, "fatigue_score": 98.5083, "state": "Sleep"}, {"ear": 0.1208, "mar": 0.3565, "perclos": 0.9051, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1231, "mar": 0.4377, "perclos": 0.9182, "fatigue_score": 99.9032, "state": "Sleep"}, {"ear": 0.1377, "mar": 0.4288, "perclos": 0.7981, "fatigue_score": 97.2941, "state": "Sleep"}, {"ear": 0.1595, "mar": 0.3859, "perclos": 0.7524, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1325, "mar": 0.4466, "perclos": 0.8323, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1391, "mar": 0.4121, "perclos": 0.9066, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1248, "mar": 0.2749, "perclos": 0.8086, "fatigue_score": 98.333, "state": "Sleep"}, {"ear": 0.1293, "mar": 0.3893, "perclos": 0.9105, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1196, "mar": 0.2884, "perclos": 0.7382, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1233, "mar": 0.4155, "perclos": 0.7817, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1371, "mar": 0.4074, "perclos": 0.9051, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1315, "mar": 0.4364, "perclos": 0.9227, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1093, "mar": 0.412, "perclos": 0.6525, "fatigue_score": 98.7308, "state": "Sleep"}, {"ear": 0.1401, "mar": 0.395, "perclos": 0.8814, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1485, "mar": 0.4172, "perclos": 0.8112, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1417, "mar": 0.3179, "perclos": 0.8461, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1375, "mar": 0.3332, "perclos": 0.8902, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.148, "mar": 0.351, "perclos": 0.7093, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1411, "mar": 0.3286, "perclos": 0.7101, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1445, "mar": 0.387, "perclos": 0.7942, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1353, "mar": 0.4019, "perclos": 0.9318, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1169, "mar": 0.3291, "perclos": 0.8526, "fatigue_score": 91.8926, "state": "Sleep"}, {"ear": 0.1234, "mar": 0.4239, "perclos": 0.9057, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1359, "mar": 0.387, "perclos": 0.9977, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1104, "mar": 0.3696, "perclos": 0.7207, "fatigue_score": 97.4923, "state": "Sleep"}, {"ear": 0.1028, "mar": 0.3058, "perclos": 0.7838, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1365, "mar": 0.295, "perclos": 0.9386, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1384, "mar": 0.4659, "perclos": 0.8229, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1071, "mar": 0.3972, "perclos": 0.8418, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1525, "mar": 0.3415, "perclos": 0.823, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1245, "mar": 0.3761, "perclos": 0.9865, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1436, "mar": 0.3189, "perclos": 0.847, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1166, "mar": 0.3518, "perclos": 0.7556, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1386, "mar": 0.2967, "perclos": 0.8116, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1284, "mar": 0.4521, "perclos": 0.6627, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1005, "mar": 0.3257, "perclos": 0.8229, "fatigue_score": 97.5011, "state": "Sleep"}, {"ear": 0.1394, "mar": 0.3424, "perclos": 0.8293, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1314, "mar": 0.2241, "perclos": 0.8922, "fatigue_score": 99.3622, "state": "Sleep"}, {"ear": 0.1013, "mar": 0.4358, "perclos": 0.8563, "fatigue_score": 99.9517, "state": "Sleep"}, {"ear": 0.2459, "mar": 0.3775, "perclos": 0.9445, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1164, "mar": 0.3791, "perclos": 0.7972, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.12, "mar": 0.4209, "perclos": 0.8766, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.2459, "mar": 0.3221, "perclos": 0.8571, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1143, "mar": 0.291, "perclos": 0.8161, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1265, "mar": 0.3436, "perclos": 0.8488, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1319, "mar": 0.2839, "perclos": 0.8588, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1317, "mar": 0.485, "perclos": 0.7729, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.1105, "mar": 0.4197, "perclos": 0.9782, "fatigue_score": 98.2305, "state": "Sleep"}, {"ear": 0.1263, "mar": 0.3493, "perclos": 0.7534, "fatigue_score": 100.0, "state": "Sleep"}, {"ear": 0.2383, "mar": 0.3008, "perclos": 0.1991, "fatigue_score": 48.1317, "state": "Slightly Drowsy"}, {"ear": 0.2222, "mar": 0.4212, "perclos": 0.2002, "fatigue_score": 44.3517, "state": "Slightly Drowsy"}, {"ear": 0.2648, "mar": 0.3771, "perclos": 0.2141, "fatigue_score": 41.9364, "state": "Slightly Drowsy"}, {"ear": 0.2688, "mar": 0.3556, "perclos": 0.1711, "fatigue_score": 48.0521, "state": "Slightly Drowsy"}, {"ear": 0.2483, "mar": 0.3573, "perclos": 0.169, "fatigue_score": 44.8623, "state": "Slightly Drowsy"}, {"ear": 0.2136, "mar": 0.351, "perclos": 0.1437, "fatigue_score": 45.6158, "state": "Slightly Drowsy"}, {"ear": 0.2435, "mar": 0.4601, "perclos": 0.1134, "fatigue_score": 35.1556, "state": "Slightly Drowsy"}, {"ear": 0.2566, "mar": 0.3726, "perclos": 0.2071, "fatigue_score": 44.6416, "state": "Slightly Drowsy"}, {"ear": 0.2487, "mar": 0.3172, "perclos": 0.1928, "fatigue_score": 33.4734, "state": "Slightly Drowsy"}, {"ear": 0.2387, "mar": 0.3272, "perclos": 0.1477, "fatigue_score": 40.1444, "state": "Slightly Drowsy"}, {"ear": 0.2605, "mar": 0.2623, "perclos": 0.1698, "fatigue_score": 39.3872, "state": "Slightly Drowsy"}, {"ear": 0.2286, "mar": 0.3465, "perclos": 0.1161, "fatigue_score": 42.0034, "state": "Slightly Drowsy"}, {"ear": 0.2433, "mar": 0.3696, "perclos": 0.1413, "fatigue_score": 40.184, "state": "Slightly Drowsy"}, {"ear": 0.2435, "mar": 0.3801, "perclos": 0.1459, "fatigue_score": 48.9362, "state": "Slightly Drowsy"}, {"ear": 0.2697, "mar": 0.3606, "perclos": 0.2389, "fatigue_score": 45.7566, "state": "Slightly Drowsy"}, {"ear": 0.2749, "mar": 0.3476, "perclos": 0.2549, "fatigue_score": 41.2781, "state": "Slightly Drowsy"}, {"ear": 0.2298, "mar": 0.3266, "perclos": 0.1724, "fatigue_score": 46.5446, "state": "Slightly Drowsy"}, {"ear": 0.23, "mar": 0.353, "perclos": 0.1656, "fatigue_score": 46.4832, "state": "Slightly Drowsy"}, {"ear": 0.235, "mar": 0.4739, "perclos": 0.1334, "fatigue_score": 46.2776, "state": "Slightly Drowsy"}, {"ear": 0.25, "mar": 0.3186, "perclos": 0.1635, "fatigue_score": 35.4912, "state": "Slightly Drowsy"}, {"ear": 0.2528, "mar": 0.2976, "perclos": 0.1061, "fatigue_score": 42.4053, "state": "Slightly Drowsy"}, {"ear": 0.2656, "mar": 0.4328, "perclos": 0.1974, "fatigue_score": 40.4004, "state": "Slightly Drowsy"}, {"ear": 0.2424, "mar": 0.3697, "perclos": 0.1251, "fatigue_score": 35.2051, "state": "Slightly Drowsy"}, {"ear": 0.2366, "mar": 0.2979, "perclos": 0.2164, "fatigue_score": 45.006, "state": "Slightly Drowsy"}, {"ear": 0.2761, "mar": 0.3436, "perclos": 0.1387, "fatigue_score": 45.249, "state": "Slightly Drowsy"}, {"ear": 0.2378, "mar": 0.3212, "perclos": 0.1489, "fatigue_score": 36.4222, "state": "Slightly Drowsy"}, {"ear": 0.2249, "mar": 0.313, "perclos": 0.173, "fatigue_score": 38.2687, "state": "Slightly Drowsy"}, {"ear": 0.2528, "mar": 0.3169, "perclos": 0.2551, "fatigue_score": 35.6956, "state": "Slightly Drowsy"}, {"ear": 0.2599, "mar": 0.3462, "perclos": 0.1741, "fatigue_score": 48.501, "state": "Slightly Drowsy"}, {"ear": 0.2409, "mar": 0.4153, "perclos": 0.1602, "fatigue_score": 37.512, "state": "Slightly Drowsy"}, {"ear": 0.2611, "mar": 0.3278, "perclos": 0.0847, "fatigue_score": 40.3144, "state": "Slightly Drowsy"}, {"ear": 0.2435, "mar": 0.4257, "perclos": 0.1698, "fatigue_score": 36.4418, "state": "Slightly Drowsy"}, {"ear": 0.2971, "mar": 0.3307, "perclos": 0.2136, "fatigue_score": 33.8436, "state": "Slightly Drowsy"}, {"ear": 0.2464, "mar": 0.3614, "perclos": 0.2455, "fatigue_score": 46.3253, "state": "Slightly Drowsy"}, {"ear": 0.2686, "mar": 0.3686, "perclos": 0.1493, "fatigue_score": 41.397, "state": "Slightly Drowsy"}, {"ear": 0.2485, "mar": 0.4095, "perclos": 0.1453, "fatigue_score": 40.3588, "state": "Slightly Drowsy"}, {"ear": 0.2481, "mar": 0.355, "perclos": 0.1838, "fatigue_score": 43.71, "state": "Slightly Drowsy"}, {"ear": 0.222, "mar": 0.312, "perclos": 0.1536, "fatigue_score": 37.0841, "state": "Slightly Drowsy"}, {"ear": 0.2603, "mar": 0.4004, "perclos": 0.1861, "fatigue_score": 55.9331, "state": "Slightly Drowsy"}, {"ear": 0.2498, "mar": 0.351, "perclos": 0.1562, "fatigue_score": 42.3542, "state": "Slightly Drowsy"}, {"ear": 0.2377, "mar": 0.386, "perclos": 0.112, "fatigue_score": 44.2229, "state": "Slightly Drowsy"}, {"ear": 0.2371, "mar": 0.3899, "perclos": 0.1245, "fatigue_score": 47.8565, "state": "Slightly Drowsy"}, {"ear": 0.2501, "mar": 0.2687, "perclos": 0.1861, "fatigue_score": 38.7053, "state": "Slightly Drowsy"}, {"ear": 0.2601, "mar": 0.3496, "perclos": 0.2583, "fatigue_score": 43.007, "state": "Slightly Drowsy"}, {"ear": 0.2386, "mar": 0.3355, "perclos": 0.1454, "fatigue_score": 37.3465, "state": "Slightly Drowsy"}, {"ear": 0.2328, "mar": 0.4086, "perclos": 0.1274, "fatigue_score": 47.1366, "state": "Slightly Drowsy"}, {"ear": 0.2491, "mar": 0.3751, "perclos": 0.17, "fatigue_score": 41.2761, "state": "Slightly Drowsy"}, {"ear": 0.2547, "mar": 0.3601, "perclos": 0.1746, "fatigue_score": 49.6866, "state": "Slightly Drowsy"}, {"ear": 0.2151, "mar": 0.3955, "perclos": 0.2006, "fatigue_score": 31.3502, "state": "Slightly Drowsy"}, {"ear": 0.2469, "mar": 0.3619, "perclos": 0.1768, "fatigue_score": 32.7119, "state": "Slightly Drowsy"}, {"ear": 0.2669, "mar": 0.3614, "perclos": 0.2344, "fatigue_score": 33.7542, "state": "Slightly Drowsy"}, {"ear": 0.2434, "mar": 0.3109, "perclos": 0.1591, "fatigue_score": 34.0904, "state": "Slightly Drowsy"}, {"ear": 0.245, "mar": 0.3856, "perclos": 0.201, "fatigue_score": 55.1971, "state": "Slightly Drowsy"}, {"ear": 0.2889, "mar": 0.3575, "perclos": 0.1819, "fatigue_score": 42.3886, "state": "Slightly Drowsy"}, {"ear": 0.2272, "mar": 0.3423, "perclos": 0.2484, "fatigue_score": 48.6105, "state": "Slightly Drowsy"}, {"ear": 0.2323, "mar": 0.3629, "perclos": 0.1352, "fatigue_score": 53.9762, "state": "Slightly Drowsy"}, {"ear": 0.2459, "mar": 0.3, "perclos": 0.2192, "fatigue_score": 36.9859, "state": "Slightly Drowsy"}, {"ear": 0.2637, "mar": 0.3899, "perclos": 0.2218, "fatigue_score": 31.3956, "state": "Slightly Drowsy"}, {"ear": 0.2248, "mar": 0.3779, "perclos": 0.1674, "fatigue_score": 44.4534, "state": "Slightly Drowsy"}, {"ear": 0.2359, "mar": 0.3191, "perclos": 0.215, "fatigue_score": 47.7491, "state": "Slightly Drowsy"}, {"ear": 0.2571, "mar": 0.3203, "perclos": 0.2141, "fatigue_score": 32.053, "state": "Slightly Drowsy"}, {"ear": 0.2838, "mar": 0.3929, "perclos": 0.208, "fatigue_score": 37.5704, "state": "Slightly Drowsy"}, {"ear": 0.2398, "mar": 0.2651, "perclos": 0.1847, "fatigue_score": 36.8661, "state": "Slightly Drowsy"}, {"ear": 0.2386, "mar": 0.2959, "perclos": 0.1861, "fatigue_score": 43.8036, "state": "Slightly Drowsy"}, {"ear": 0.2444, "mar": 0.3014, "perclos": 0.1653, "fatigue_score": 54.744, "state": "Slightly Drowsy"}, {"ear": 0.2598, "mar": 0.3124, "perclos": 0.2223, "fatigue_score": 37.0903, "state": "Slightly Drowsy"}, {"ear": 0.2481, "mar": 0.3127, "perclos": 0.2012, "fatigue_score": 34.5461, "state": "Slightly Drowsy"}, {"ear": 0.2417, "mar": 0.3759, "perclos": 0.1739, "fatigue_score": 46.4025, "state": "Slightly Drowsy"}, {"ear": 0.2786, "mar": 0.3532, "perclos": 0.2384, "fatigue_score": 32.5881, "state": "Slightly Drowsy"}, {"ear": 0.2779, "mar": 0.4075, "perclos": 0.1754, "fatigue_score": 54.2321, "state": "Slightly Drowsy"}, {"ear": 0.2672, "mar": 0.3471, "perclos": 0.2166, "fatigue_score": 37.7324, "state": "Slightly Drowsy"}, {"ear": 0.2469, "mar": 0.4144, "perclos": 0.202, "fatigue_score": 34.5185, "state": "Slightly Drowsy"}, {"ear": 0.2523, "mar": 0.3677, "perclos": 0.0637, "fatigue_score": 47.7446, "state": "Slightly Drowsy"}, {"ear": 0.254, "mar": 0.3592, "perclos": 0.159, "fatigue_score": 42.656, "state": "Slightly Drowsy"}, {"ear": 0.2554, "mar": 0.3604, "perclos": 0.1028, "fatigue_score": 37.3276, "state": "Slightly Drowsy"}, {"ear": 0.2611, "mar": 0.3511, "perclos": 0.2132, "fatigue_score": 40.8272, "state": "Slightly Drowsy"}, {"ear": 0.2441, "mar": 0.3538, "perclos": 0.2188, "fatigue_score": 39.5385, "state": "Slightly Drowsy"}, {"ear": 0.2176, "mar": 0.3194, "perclos": 0.164, "fatigue_score": 34.6877, "state": "Slightly Drowsy"}, {"ear": 0.2459, "mar": 0.3148, "perclos": 0.2076, "fatigue_score": 29.8929, "state": "Slightly Drowsy"}, {"ear": 0.2494, "mar": 0.2876, "perclos": 0.1413, "fatigue_score": 41.0979, "state": "Slightly Drowsy"}, {"ear": 0.2785, "mar": 0.263, "perclos": 0.1732, "fatigue_score": 41.8499, "state": "Slightly Drowsy"}, {"ear": 0.2201, "mar": 0.2916, "perclos": 0.1101, "fatigue_score": 36.6243, "state": "Slightly Drowsy"}, {"ear": 0.2833, "mar": 0.3965, "perclos": 0.1621, "fatigue_score": 40.1007, "state": "Slightly Drowsy"}, {"ear": 0.2588, "mar": 0.3673, "perclos": 0.1435, "fatigue_score": 42.6428, "state": "Slightly Drowsy"}, {"ear": 0.258, "mar": 0.354, "perclos": 0.2129, "fatigue_score": 44.4382, "state": "Slightly Drowsy"}, {"ear": 0.2287, "mar": 0.3815, "perclos": 0.2029, "fatigue_score": 33.7283, "state": "Slightly Drowsy"}, {"ear": 0.276, "mar": 0.3234, "perclos": 0.1684, "fatigue_score": 41.2047, "state": "Slightly Drowsy"}, {"ear": 0.2283, "mar": 0.3456, "perclos": 0.1903, "fatigue_score": 39.5418, "state": "Slightly Drowsy"}, {"ear": 0.2539, "mar": 0.4394, "perclos": 0.2058, "fatigue_score": 41.3921, "state": "Slightly Drowsy"}, {"ear": 0.2526, "mar": 0.369, "perclos": 0.2913, "fatigue_score": 40.6354, "state": "Slightly Drowsy"}], "telemetry": [{"frame_idx": 0, "state_label": "Alert", "alert_level": 0, "fatigue_score": 0.1, "ear": 0.35, "mar": 0.15, "perclos": 0.05}, {"frame_idx": 10, "state_label": "Slightly Drowsy", "alert_level": 1, "fatigue_score": 0.55, "ear": 0.22, "mar": 0.55, "perclos": 0.35}, {"frame_idx": 11, "state_label": "Slightly Drowsy", "alert_level": 1, "fatigue_score": 0.55, "ear": 0.22, "mar": 0.55, "perclos": 0.35}, {"frame_idx": 12, "state_label": "Slightly Drowsy", "alert_level": 1, "fatigue_score": 0.55, "ear": 0.22, "mar": 0.55, "perclos": 0.35}, {"frame_idx": 13, "state_label": "Slightly Drowsy", "alert_level": 1, "fatigue_score": 0.55, "ear": 0.22, "mar": 0.55, "perclos": 0.35}, {"frame_idx": 14, "state_label": "Slightly Drowsy", "alert_level": 1, "fatigue_score": 0.55, "ear": 0.22, "mar": 0.55, "perclos": 0.35}, {"frame_idx": 15, "state_label": "Slightly Drowsy", "alert_level": 1, "fatigue_score": 0.55, "ear": 0.22, "mar": 0.55, "perclos": 0.35}, {"frame_idx": 16, "state_label": "Slightly Drowsy", "alert_level": 1, "fatigue_score": 0.55, "ear": 0.22, "mar": 0.55, "perclos": 0.35}, {"frame_idx": 17, "state_label": "Slightly Drowsy", "alert_level": 1, "fatigue_score": 0.55, "ear": 0.22, "mar": 0.55, "perclos": 0.35}, {"frame_idx": 18, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 19, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 20, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 21, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 22, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 23, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 24, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 25, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 26, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 27, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 28, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 29, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 30, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}, {"frame_idx": 31, "state_label": "Drowsy", "alert_level": 2, "fatigue_score": 0.85, "ear": 0.12, "mar": 0.2, "perclos": 0.75}]};

const STATE_COLORS = {
  "Alert": "#35D07F",
  "Slightly Drowsy": "#F5C242",
  "Drowsy": "#F0923A",
  "Sleep": "#F0453A",
};

const ALERT_LEVEL_COLOR = { 0: "#35D07F", 1: "#F5C242", 2: "#F0453A" };
const ALERT_LEVEL_LABEL = { 0: "SAFE", 1: "WARNING", 2: "CRITICAL" };

const CYAN = "#2DD4E8";
const MUTED = "#5C6773";
const PANEL_BG = "#0F1418";
const PANEL_BORDER = "#1D262D";
const BG = "#080B0D";
const TEXT = "#DCE4EA";

function Panel({ title, sub, children, className = "" }) {
  return (
    <div
      className={`rounded-md ${className}`}
      style={{ background: PANEL_BG, border: `1px solid ${PANEL_BORDER}` }}
    >
      <div className="flex items-baseline justify-between px-4 pt-3 pb-2" style={{ borderBottom: `1px solid ${PANEL_BORDER}` }}>
        <h3 className="text-[11px] font-mono tracking-[0.2em] uppercase" style={{ color: MUTED }}>
          {title}
        </h3>
        {sub && <span className="text-[10px] font-mono" style={{ color: MUTED }}>{sub}</span>}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function HudTooltip({ active, payload, label, unit = "" }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div
      className="font-mono text-[11px] px-3 py-2 rounded"
      style={{ background: "#0A0E11", border: `1px solid ${PANEL_BORDER}`, color: TEXT }}
    >
      {label !== undefined && <div style={{ color: MUTED }}>{String(label)}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || p.fill || CYAN }}>
          {p.name}: {typeof p.value === "number" ? p.value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : p.value}{unit}
        </div>
      ))}
    </div>
  );
}

function StatChip({ label, value, color }) {
  return (
    <div className="flex flex-col items-start">
      <span className="text-[9px] font-mono uppercase tracking-widest" style={{ color: MUTED }}>{label}</span>
      <span className="text-lg font-mono font-semibold" style={{ color: color || TEXT }}>{value}</span>
    </div>
  );
}

function TabButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-2 text-[11px] font-mono tracking-[0.15em] uppercase transition-colors"
      style={{
        color: active ? "#0A0D10" : MUTED,
        background: active ? CYAN : "transparent",
        borderRadius: 4,
      }}
    >
      {children}
    </button>
  );
}

export default function DrowsinessMLDashboard() {
  const [tab, setTab] = useState("leaderboard");
  const [metric, setMetric] = useState("Accuracy (%)");

  const { leaderboard, classDist, stateStats, scatter, telemetry } = DASHBOARD_DATA;

  const sortedLeaderboard = useMemo(
    () => [...leaderboard].sort((a, b) => b[metric] - a[metric]),
    [leaderboard, metric]
  );

  const bestModel = leaderboard.reduce((a, b) => (b["Accuracy (%)"] > a["Accuracy (%)"] ? b : a), leaderboard[0]);
  const fastestModel = leaderboard.reduce((a, b) => (b["Throughput (FPS)"] > a["Throughput (FPS)"] ? b : a), leaderboard[0]);
  const smallestModel = leaderboard.reduce((a, b) => (b["Size (KB)"] < a["Size (KB)"] ? b : a), leaderboard[0]);

  const radarModels = useMemo(() => {
    const top3 = [...leaderboard].sort((a, b) => b["Accuracy (%)"] - a["Accuracy (%)"]).slice(0, 3);
    const metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "Macro F1 (%)"];
    return metrics.map((m) => {
      const row = { metric: m.replace(" (%)", "") };
      top3.forEach((model) => { row[model.Model] = model[m]; });
      return row;
    });
  }, [leaderboard]);
  const top3Models = useMemo(
    () => [...leaderboard].sort((a, b) => b["Accuracy (%)"] - a["Accuracy (%)"]).slice(0, 3),
    [leaderboard]
  );
  const radarColors = [CYAN, "#35D07F", "#F5C242"];

  const donutData = Object.entries(classDist.counts).map(([name, value]) => ({ name, value }));

  const pieces = ["ear", "perclos", "mar", "fatigue_score"];
  const [scatterX, setScatterX] = useState("ear");
  const [scatterY, setScatterY] = useState("perclos");

  const scatterByState = useMemo(() => {
    const groups = {};
    scatter.forEach((row) => {
      if (!groups[row.state]) groups[row.state] = [];
      groups[row.state].push(row);
    });
    return groups;
  }, [scatter]);

  return (
    <div className="w-full min-h-screen font-sans" style={{ background: BG, color: TEXT }}>
      <div className="max-w-5xl mx-auto px-5 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-1">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: "#35D07F", boxShadow: "0 0 8px #35D07F" }} />
              <h1 className="text-sm font-mono tracking-[0.25em] uppercase" style={{ color: TEXT }}>
                Model Telemetry — Drowsiness Detection Stack
              </h1>
            </div>
            <p className="text-[11px] font-mono mt-1" style={{ color: MUTED }}>
              8 architectures · 4,001 labeled frames · 4-state taxonomy (Alert / Slightly Drowsy / Drowsy / Sleep)
            </p>
          </div>
        </div>

        {/* Top stat strip */}
        <div className="grid grid-cols-4 gap-3 mt-5 mb-5">
          <Panel title="Best Accuracy">
            <StatChip label={bestModel.Model} value={`${bestModel["Accuracy (%)"]}%`} color="#35D07F" />
          </Panel>
          <Panel title="Fastest Inference">
            <StatChip label={fastestModel.Model} value={`${Math.round(fastestModel["Throughput (FPS)"]).toLocaleString()} fps`} color={CYAN} />
          </Panel>
          <Panel title="Smallest Footprint">
            <StatChip label={smallestModel.Model} value={`${smallestModel["Size (KB)"]} KB`} color="#F5C242" />
          </Panel>
          <Panel title="Dataset Size">
            <StatChip label="labeled frames" value={classDist.total_samples.toLocaleString()} color={TEXT} />
          </Panel>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-4 p-1 rounded-md w-fit" style={{ background: PANEL_BG, border: `1px solid ${PANEL_BORDER}` }}>
          <TabButton active={tab === "leaderboard"} onClick={() => setTab("leaderboard")}>Leaderboard</TabButton>
          <TabButton active={tab === "features"} onClick={() => setTab("features")}>Feature Space</TabButton>
          <TabButton active={tab === "live"} onClick={() => setTab("live")}>Live Run</TabButton>
          <TabButton active={tab === "balance"} onClick={() => setTab("balance")}>Class Balance</TabButton>
        </div>

        {/* LEADERBOARD TAB */}
        {tab === "leaderboard" && (
          <div className="grid grid-cols-2 gap-4">
            <Panel
              title="Ranked by metric"
              sub={metric}
              className="col-span-2"
            >
              <div className="flex gap-1 mb-3">
                {["Accuracy (%)", "Macro F1 (%)", "ROC AUC", "Throughput (FPS)"].map((m) => (
                  <button
                    key={m}
                    onClick={() => setMetric(m)}
                    className="px-2 py-1 text-[10px] font-mono uppercase tracking-wide rounded"
                    style={{
                      background: metric === m ? "#1A2126" : "transparent",
                      color: metric === m ? CYAN : MUTED,
                      border: `1px solid ${metric === m ? CYAN : PANEL_BORDER}`,
                    }}
                  >
                    {m}
                  </button>
                ))}
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={sortedLeaderboard} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={PANEL_BORDER} horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }} stroke={PANEL_BORDER} />
                  <YAxis
                    type="category"
                    dataKey="Model"
                    width={140}
                    tick={{ fontSize: 10, fill: TEXT, fontFamily: "monospace" }}
                    stroke={PANEL_BORDER}
                  />
                  <Tooltip content={<HudTooltip />} cursor={{ fill: "rgba(45,212,232,0.06)" }} />
                  <Bar dataKey={metric} radius={[0, 3, 3, 0]}>
                    {sortedLeaderboard.map((entry, i) => (
                      <Cell key={i} fill={i === 0 ? "#35D07F" : CYAN} fillOpacity={i === 0 ? 1 : 0.55} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Top 3 · Precision / Recall / F1 Profile">
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarModels} outerRadius={85}>
                  <PolarGrid stroke={PANEL_BORDER} />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }} />
                  <PolarRadiusAxis domain={[90, 100]} tick={{ fontSize: 9, fill: MUTED }} tickCount={3} />
                  {top3Models.map((model, i) => (
                    <Radar
                      key={model.Model}
                      name={model.Model}
                      dataKey={model.Model}
                      stroke={radarColors[i]}
                      fill={radarColors[i]}
                      fillOpacity={0.12}
                      strokeWidth={2}
                    />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 10, fontFamily: "monospace", color: MUTED }} />
                  <Tooltip content={<HudTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Latency vs. Accuracy tradeoff" sub="log-scale latency (ms)">
              <ResponsiveContainer width="100%" height={260}>
                <ScatterChart margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                  <CartesianGrid stroke={PANEL_BORDER} />
                  <XAxis
                    type="number"
                    dataKey="Latency (ms)"
                    scale="log"
                    domain={["auto", "auto"]}
                    tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }}
                    stroke={PANEL_BORDER}
                    label={{ value: "Latency (ms, log)", position: "insideBottom", offset: -5, fontSize: 10, fill: MUTED }}
                  />
                  <YAxis
                    type="number"
                    dataKey="Accuracy (%)"
                    domain={[95, 100.5]}
                    tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }}
                    stroke={PANEL_BORDER}
                  />
                  <ZAxis type="number" dataKey="Size (KB)" range={[60, 400]} name="Size (KB)" />
                  <Tooltip content={<HudTooltip />} cursor={{ strokeDasharray: "3 3", stroke: MUTED }} />
                  <Scatter data={leaderboard} fill={CYAN} fillOpacity={0.75} />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-[10px] font-mono mt-1" style={{ color: MUTED }}>bubble size = model footprint (KB)</p>
            </Panel>
          </div>
        )}

        {/* FEATURE SPACE TAB */}
        {tab === "features" && (
          <div className="grid grid-cols-2 gap-4">
            <Panel title="Biometric feature space" sub={`${scatterX} vs ${scatterY}`} className="col-span-2">
              <div className="flex gap-4 mb-3">
                <div className="flex items-center gap-1">
                  <span className="text-[10px] font-mono uppercase" style={{ color: MUTED }}>X:</span>
                  {pieces.map((p) => (
                    <button key={p} onClick={() => setScatterX(p)}
                      className="px-2 py-0.5 text-[10px] font-mono rounded"
                      style={{ background: scatterX === p ? "#1A2126" : "transparent", color: scatterX === p ? CYAN : MUTED, border: `1px solid ${scatterX === p ? CYAN : PANEL_BORDER}` }}>
                      {p}
                    </button>
                  ))}
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-[10px] font-mono uppercase" style={{ color: MUTED }}>Y:</span>
                  {pieces.map((p) => (
                    <button key={p} onClick={() => setScatterY(p)}
                      className="px-2 py-0.5 text-[10px] font-mono rounded"
                      style={{ background: scatterY === p ? "#1A2126" : "transparent", color: scatterY === p ? CYAN : MUTED, border: `1px solid ${scatterY === p ? CYAN : PANEL_BORDER}` }}>
                      {p}
                    </button>
                  ))}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                  <CartesianGrid stroke={PANEL_BORDER} />
                  <XAxis type="number" dataKey={scatterX} tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }} stroke={PANEL_BORDER} name={scatterX} />
                  <YAxis type="number" dataKey={scatterY} tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }} stroke={PANEL_BORDER} name={scatterY} />
                  <Tooltip content={<HudTooltip />} cursor={{ strokeDasharray: "3 3", stroke: MUTED }} />
                  <Legend wrapperStyle={{ fontSize: 10, fontFamily: "monospace", color: MUTED }} />
                  {Object.entries(scatterByState).map(([state, rows]) => (
                    <Scatter key={state} name={state} data={rows} fill={STATE_COLORS[state]} fillOpacity={0.65} />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-[10px] font-mono mt-1" style={{ color: MUTED }}>360 stratified samples across 4 physiological states</p>
            </Panel>

            <Panel title="Mean fatigue score by state">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={stateStats} margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={PANEL_BORDER} vertical={false} />
                  <XAxis dataKey="state" tick={{ fontSize: 9, fill: MUTED, fontFamily: "monospace" }} stroke={PANEL_BORDER} interval={0} angle={-15} textAnchor="end" height={50} />
                  <YAxis tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }} stroke={PANEL_BORDER} />
                  <Tooltip content={<HudTooltip />} cursor={{ fill: "rgba(45,212,232,0.06)" }} />
                  <Bar dataKey="fatigue_score" radius={[3, 3, 0, 0]}>
                    {stateStats.map((entry, i) => (
                      <Cell key={i} fill={STATE_COLORS[entry.state]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Mean PERCLOS by state" sub="% eye closure">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={stateStats} margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={PANEL_BORDER} vertical={false} />
                  <XAxis dataKey="state" tick={{ fontSize: 9, fill: MUTED, fontFamily: "monospace" }} stroke={PANEL_BORDER} interval={0} angle={-15} textAnchor="end" height={50} />
                  <YAxis tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }} stroke={PANEL_BORDER} />
                  <Tooltip content={<HudTooltip />} cursor={{ fill: "rgba(45,212,232,0.06)" }} />
                  <Bar dataKey="perclos" radius={[3, 3, 0, 0]}>
                    {stateStats.map((entry, i) => (
                      <Cell key={i} fill={STATE_COLORS[entry.state]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Panel>
          </div>
        )}

        {/* LIVE RUN TAB */}
        {tab === "live" && (
          <div className="grid grid-cols-1 gap-4">
            <Panel title="Simulated cockpit run" sub={`${telemetry.length} logged frames`}>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={telemetry} margin={{ left: 0, right: 20, top: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={PANEL_BORDER} />
                  <XAxis dataKey="frame_idx" tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }} stroke={PANEL_BORDER} label={{ value: "frame index", position: "insideBottom", offset: -3, fontSize: 10, fill: MUTED }} />
                  <YAxis tick={{ fontSize: 10, fill: MUTED, fontFamily: "monospace" }} stroke={PANEL_BORDER} />
                  <Tooltip content={<HudTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 10, fontFamily: "monospace", color: MUTED }} />
                  <Line type="stepAfter" dataKey="fatigue_score" name="fatigue score" stroke={CYAN} strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="stepAfter" dataKey="ear" name="EAR" stroke="#35D07F" strokeWidth={1.5} dot={{ r: 2 }} />
                  <Line type="stepAfter" dataKey="perclos" name="PERCLOS" stroke="#F5C242" strokeWidth={1.5} dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Alert level per frame" sub="0=safe · 1=warning · 2=critical">
              <div className="flex flex-wrap gap-1">
                {telemetry.map((t, i) => (
                  <div
                    key={i}
                    title={`frame ${t.frame_idx}: ${t.state_label}`}
                    className="w-6 h-6 rounded-sm flex items-center justify-center text-[8px] font-mono"
                    style={{ background: ALERT_LEVEL_COLOR[t.alert_level] + "22", border: `1px solid ${ALERT_LEVEL_COLOR[t.alert_level]}`, color: ALERT_LEVEL_COLOR[t.alert_level] }}
                  >
                    {t.alert_level}
                  </div>
                ))}
              </div>
              <div className="flex gap-4 mt-3">
                {[0, 1, 2].map((lvl) => (
                  <div key={lvl} className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full" style={{ background: ALERT_LEVEL_COLOR[lvl] }} />
                    <span className="text-[10px] font-mono" style={{ color: MUTED }}>{ALERT_LEVEL_LABEL[lvl]}</span>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
        )}

        {/* CLASS BALANCE TAB */}
        {tab === "balance" && (
          <div className="grid grid-cols-2 gap-4">
            <Panel title="Class distribution" sub={`${classDist.total_samples.toLocaleString()} total frames`}>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={donutData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={65}
                    outerRadius={100}
                    paddingAngle={2}
                  >
                    {donutData.map((entry, i) => (
                      <Cell key={i} fill={STATE_COLORS[entry.name]} stroke={BG} strokeWidth={2} />
                    ))}
                  </Pie>
                  <Tooltip content={<HudTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 10, fontFamily: "monospace", color: MUTED }} />
                </PieChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Class share">
              <div className="flex flex-col gap-3 justify-center h-full">
                {Object.entries(classDist.percentages).map(([name, pct]) => (
                  <div key={name}>
                    <div className="flex justify-between text-[11px] font-mono mb-1">
                      <span style={{ color: STATE_COLORS[name] }}>{name}</span>
                      <span style={{ color: MUTED }}>{pct}% · {classDist.counts[name].toLocaleString()}</span>
                    </div>
                    <div className="w-full h-2 rounded-full" style={{ background: PANEL_BORDER }}>
                      <div className="h-2 rounded-full" style={{ width: `${pct}%`, background: STATE_COLORS[name] }} />
                    </div>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
        )}

        <p className="text-[10px] font-mono mt-6 text-center" style={{ color: MUTED }}>
          Data sourced from outputs/evaluation, outputs/eda, dataset/cleaned_features.csv, outputs/alert_log.csv
        </p>
      </div>
    </div>
  );
}
