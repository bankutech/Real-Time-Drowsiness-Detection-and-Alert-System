"""
Real-Time Drowsiness Detection and Alert System
Interactive Application & Web Control Center.

Features:
- Live MJPEG Video & HUD Telemetry Streaming (/video_feed)
- Real-Time JSON Telemetry API (/api/telemetry)
- Dynamic Driver Baseline Calibration API (/api/calibrate)
- Client-Side Web Audio Multi-Tier Warning Synthesizer (Warning Chime & Critical Buzzer)
- Speech vs Yawn Disambiguation & Low-Light CLAHE Boost
- Model Selection & Benchmark Results Gallery
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src import config
from src.utils import setup_logger
from src.realtime_detection import DrowsinessDetectorPipeline
from tests.test_phase8 import generate_synthetic_driver_frame

logger = setup_logger("AppServer")

# Global pipeline instance and stream state
pipeline_lock = threading.Lock()
global_pipeline: Optional[DrowsinessDetectorPipeline] = None
latest_frame_jpeg: bytes = b""
latest_telemetry: Dict[str, Any] = {}
stream_active = True
sim_state_idx = 0


def background_stream_worker(camera_idx: int = 0, use_simulation: bool = True):
    """Generates continuous frames and runs ML/HMM/Alert pipeline in background."""
    global latest_frame_jpeg, latest_telemetry, global_pipeline, stream_active, sim_state_idx

    logger.info(f"Starting stream worker (Simulation={use_simulation})...")

    cap = None
    if not use_simulation:
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            logger.warning(f"Webcam {camera_idx} unavailable. Falling back to dynamic simulation stream.")
            use_simulation = True

    sim_cycle = (
        ["alert"] * 70
        + ["drowsy"] * 80
        + ["sleeping"] * 60
        + ["alert"] * 50
    )

    while stream_active:
        frame = None
        if not use_simulation and cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            state = sim_cycle[sim_state_idx % len(sim_cycle)]
            frame = generate_synthetic_driver_frame(state)
            sim_state_idx += 1

        if frame is not None:
            with pipeline_lock:
                if global_pipeline is not None:
                    hud_frame, telem = global_pipeline.process_frame(frame)
                    latest_telemetry = telem

                    # Encode to JPEG
                    ret, jpeg = cv2.imencode(".jpg", hud_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        latest_frame_jpeg = jpeg.tobytes()

        time.sleep(0.035)  # ~28 FPS stream rate

    if cap:
        cap.release()


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Safety AI — Real-Time Drowsiness Detection System</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0d14;
            --bg-card: rgba(18, 24, 38, 0.90);
            --border-color: rgba(64, 85, 128, 0.35);
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-green: #00e676;
            --accent-yellow: #ffd600;
            --accent-orange: #ff9100;
            --accent-red: #ff1744;
            --text-primary: #e6edf8;
            --text-secondary: #8da2c0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.06), transparent 40%),
                              radial-gradient(circle at 85% 85%, rgba(79, 172, 254, 0.06), transparent 40%);
        }

        header {
            background: rgba(10, 13, 20, 0.95);
            backdrop-filter: blur(14px);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .brand-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: 1.5px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-sub {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .btn-action {
            background: rgba(25, 34, 52, 0.9);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }

        .btn-action:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            transform: translateY(-1px);
        }

        .btn-calibrate {
            background: linear-gradient(135deg, rgba(0, 242, 254, 0.15), rgba(79, 172, 254, 0.15));
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        .btn-calibrate:hover {
            background: linear-gradient(135deg, rgba(0, 242, 254, 0.3), rgba(79, 172, 254, 0.3));
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.25);
        }

        .badge-live {
            background: rgba(0, 230, 118, 0.15);
            border: 1px solid var(--accent-green);
            color: var(--accent-green);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .badge-live::before {
            content: '';
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 10px var(--accent-green);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.7; }
            50% { transform: scale(1.3); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.7; }
        }

        .nav-tabs {
            display: flex;
            gap: 8px;
            padding: 12px 32px 0 32px;
            background: rgba(14, 18, 28, 0.85);
            border-bottom: 1px solid var(--border-color);
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-size: 0.92rem;
            font-weight: 600;
            padding: 12px 24px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            color: var(--text-primary);
        }

        .tab-btn.active {
            color: var(--accent-cyan);
            border-bottom-color: var(--accent-cyan);
        }

        .main-container {
            padding: 24px 32px;
            flex: 1;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        .tab-pane {
            display: none;
        }

        .tab-pane.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .grid-live {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
        }

        @media (max-width: 1100px) {
            .grid-live {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
        }

        .card-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--accent-blue);
        }

        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
            aspect-ratio: 4 / 3;
            max-height: 520px;
        }

        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        /* Status Pills Row */
        .status-pills {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 16px;
        }

        .pill {
            background: rgba(14, 20, 32, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .pill-label {
            font-size: 0.72rem;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .pill-value {
            font-size: 0.9rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }

        .pill-value.good { color: var(--accent-green); }
        .pill-value.warn { color: var(--accent-yellow); }
        .pill-value.alert { color: var(--accent-red); }

        /* Alert Status Banner */
        .alert-banner {
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: 'Orbitron', sans-serif;
            font-weight: 800;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }

        .alert-level-0 {
            background: rgba(0, 230, 118, 0.12);
            border: 1px solid var(--accent-green);
            color: var(--accent-green);
        }

        .alert-level-1 {
            background: rgba(255, 214, 0, 0.15);
            border: 1px solid var(--accent-yellow);
            color: var(--accent-yellow);
            box-shadow: 0 0 20px rgba(255, 214, 0, 0.2);
            animation: pulse-warn 1s infinite;
        }

        .alert-level-2 {
            background: rgba(255, 23, 68, 0.2);
            border: 1px solid var(--accent-red);
            color: var(--accent-red);
            box-shadow: 0 0 30px rgba(255, 23, 68, 0.35);
            animation: flash-crit 0.6s infinite;
        }

        @keyframes pulse-warn {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.01); }
        }

        @keyframes flash-crit {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.85; transform: scale(1.02); }
        }

        /* Metric Gauges */
        .metric-row {
            margin-bottom: 14px;
        }

        .metric-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            margin-bottom: 6px;
        }

        .metric-name {
            font-weight: 600;
            color: var(--text-secondary);
        }

        .metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--text-primary);
        }

        .progress-bar-bg {
            background: rgba(40, 50, 70, 0.6);
            border-radius: 6px;
            height: 10px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.15s ease, background-color 0.2s ease;
            background-color: var(--accent-green);
        }

        .threshold-marker {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #ffffff;
            opacity: 0.7;
            z-index: 2;
        }

        /* Model Selector */
        .model-select {
            background: rgba(14, 20, 32, 0.9);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            outline: none;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }

        .model-select:focus {
            border-color: var(--accent-cyan);
        }

        /* Results Gallery */
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 20px;
        }

        .gallery-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .gallery-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(0, 242, 254, 0.15);
            border-color: var(--accent-cyan);
        }

        .gallery-card img {
            width: 100%;
            height: 220px;
            object-fit: cover;
            display: block;
            background: #0d1117;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
        }

        .gallery-info {
            padding: 16px;
        }

        .gallery-title {
            font-weight: 700;
            font-size: 0.95rem;
            color: var(--text-primary);
            margin-bottom: 6px;
        }

        .gallery-desc {
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        /* Modal for full size view */
        .img-modal {
            display: none;
            position: fixed;
            z-index: 999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            justify-content: center;
            align-items: center;
        }

        .img-modal img {
            max-width: 90%;
            max-height: 90%;
            border-radius: 8px;
            box-shadow: 0 0 40px rgba(0, 242, 254, 0.3);
        }

        .close-modal {
            position: absolute;
            top: 20px;
            right: 35px;
            color: #fff;
            font-size: 36px;
            font-weight: bold;
            cursor: pointer;
        }
    </style>
</head>
<body>

    <header>
        <div class="brand-container">
            <span class="brand-title">DRIVER SAFETY AI</span>
            <span class="brand-sub">REAL-TIME DROWSINESS DETECTION & ALERT SYSTEM</span>
        </div>
        <div class="header-actions">
            <button id="btn-sound" class="btn-action" onclick="toggleAudio()">
                <span id="sound-icon">🔊</span>
                <span id="sound-label">Audio: ON</span>
            </button>
            <button class="btn-action btn-calibrate" onclick="triggerCalibration()">
                <span>🎯</span>
                <span>Calibrate Baseline</span>
            </button>
            <div class="badge-live">LIVE ACTIVE</div>
        </div>
    </header>

    <div class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('live-tab')">🎥 Live Detection</button>
        <button class="tab-btn" onclick="switchTab('results-tab')">🖼️ Results & Empirical Evaluation</button>
    </div>

    <div class="main-container">
        
        <!-- TAB 1: Live Detection (Working Model) -->
        <div id="live-tab" class="tab-pane active">
            
            <div id="alert-banner" class="alert-banner alert-level-0">
                <span id="alert-text">STATUS: DRIVER ALERT & ATTENTIVE</span>
                <span id="alert-badge" style="font-size: 0.85rem;">LEVEL 0</span>
            </div>

            <div class="grid-live">
                <!-- Video Stream -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">LIVE COCKPIT HUD STREAM</span>
                        <span id="fps-badge" style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--accent-green);">FPS: -- | Latency: -- ms</span>
                    </div>
                    <div class="video-container">
                        <img src="/video_feed" alt="Real-Time Driver Stream" id="video-stream">
                    </div>

                    <!-- Status Pills -->
                    <div class="status-pills">
                        <div class="pill">
                            <span class="pill-label">Baseline EAR</span>
                            <span id="pill-baseline" class="pill-value good">0.320</span>
                        </div>
                        <div class="pill">
                            <span class="pill-label">Speech Filter</span>
                            <span id="pill-speech" class="pill-value good">Normal (Silent)</span>
                        </div>
                        <div class="pill">
                            <span class="pill-label">Cabin Lighting</span>
                            <span id="pill-light" class="pill-value good">Optimal</span>
                        </div>
                        <div class="pill">
                            <span class="pill-label">Eyewear Check</span>
                            <span id="pill-eyewear" class="pill-value good">Normal</span>
                        </div>
                    </div>
                </div>

                <!-- Live Telemetry Meters -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">BIOMETRIC TELEMETRY</span>
                        <span style="font-size: 0.75rem; color: var(--text-secondary);">Real-Time Sensor Fusion</span>
                    </div>

                    <!-- EAR Meter -->
                    <div class="metric-row">
                        <div class="metric-header">
                            <span class="metric-name">Eye Aspect Ratio (EAR)</span>
                            <span id="val-ear" class="metric-val">0.00</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div id="bar-ear" class="progress-bar-fill" style="width: 0%;"></div>
                            <div class="threshold-marker" style="left: 50%;"></div>
                        </div>
                    </div>

                    <!-- MAR Meter -->
                    <div class="metric-row">
                        <div class="metric-header">
                            <span class="metric-name">Mouth Aspect Ratio (MAR)</span>
                            <span id="val-mar" class="metric-val">0.00</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div id="bar-mar" class="progress-bar-fill" style="width: 0%;"></div>
                            <div class="threshold-marker" style="left: 65%;"></div>
                        </div>
                    </div>

                    <!-- PERCLOS Meter -->
                    <div class="metric-row">
                        <div class="metric-header">
                            <span class="metric-name">PERCLOS (% Eye Closure)</span>
                            <span id="val-perclos" class="metric-val">0.00</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div id="bar-perclos" class="progress-bar-fill" style="width: 0%;"></div>
                            <div class="threshold-marker" style="left: 20%;"></div>
                        </div>
                    </div>

                    <!-- Continuous Fatigue Regression -->
                    <div class="metric-row">
                        <div class="metric-header">
                            <span class="metric-name">Continuous Fatigue Index</span>
                            <span id="val-fatigue" class="metric-val">0.00</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div id="bar-fatigue" class="progress-bar-fill" style="width: 0%;"></div>
                            <div class="threshold-marker" style="left: 45%;"></div>
                        </div>
                    </div>

                    <!-- Head Pose Display -->
                    <div style="margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border-color);">
                        <div class="metric-header">
                            <span class="metric-name">Head Pose Orientation</span>
                            <span id="val-pose" class="metric-val">P: 0° | Y: 0° | R: 0°</span>
                        </div>
                    </div>

                    <!-- Primary Classification Engine Selection -->
                    <div style="margin-top: 20px; padding-top: 14px; border-top: 1px solid var(--border-color);">
                        <label for="model-selector" class="metric-name" style="display: block;">Active Inference Model:</label>
                        <select id="model-selector" class="model-select" onchange="changeModel(this.value)">
                            <option value="ensemble">Stacking Ensemble (RF + SVM + Bayes)</option>
                            <option value="rf">Random Forest (Bagging Ensemble)</option>
                            <option value="bayes">Bayesian Logistic Regression</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: Results & Empirical Evaluation -->
        <div id="results-tab" class="tab-pane">
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header">
                    <span class="card-title">PROJECT EVALUATION ARTIFACTS & MODEL BENCHMARKS</span>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
                    Comprehensive output charts generated across Machine Learning Units 1–5: Exploratory Data Analysis, Linear Models & Bayesian Posteriors, Dimensionality Reduction & Clustering, Temporal Markov Chains, and Ensemble Classifiers.
                </p>
            </div>

            <div class="gallery-grid">
                <div class="gallery-card">
                    <img src="/outputs/evaluation/benchmark_comparison.png" alt="Benchmark Comparison" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Unified Multi-Model Benchmark</div>
                        <div class="gallery-desc">Comparison of Accuracy, F1-Score, ROC-AUC, and Real-Time Latency across all 8 algorithms.</div>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="/outputs/evaluation/multi_model_roc_curves.png" alt="ROC Curves" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Multi-Model ROC Curves</div>
                        <div class="gallery-desc">Receiver Operating Characteristic curves demonstrating discriminative power and TPR vs FPR.</div>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="/outputs/evaluation/all_models_confusion_matrices.png" alt="Confusion Matrices" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Confusion Matrices (All Models)</div>
                        <div class="gallery-desc">True vs Predicted class distribution for Alert, Drowsy, and Sleeping states.</div>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="/outputs/evaluation/hmm_state_sequence_decoding.png" alt="HMM Decoding" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">HMM Temporal State Decoding</div>
                        <div class="gallery-desc">Viterbi state trajectory smoothing out single-frame camera sensor noise.</div>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="/outputs/evaluation/hmm_transition_matrix.png" alt="HMM Transition Matrix" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">HMM Transition Matrix Heatmap</div>
                        <div class="gallery-desc">Empirical transition probabilities between physiological driver alertness states.</div>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="/outputs/evaluation/random_forest_feature_importance.png" alt="Feature Importance" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Random Forest Feature Importance</div>
                        <div class="gallery-desc">Relative Gini importance of EAR, PERCLOS, MAR, and Head Nodding dynamics.</div>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="/outputs/evaluation/svm_decision_boundary_rbf.png" alt="SVM Decision Boundary" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">RBF SVM Non-Linear Boundary</div>
                        <div class="gallery-desc">Kernel support vector machine separating alert vs fatigued states in PCA space.</div>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="/outputs/evaluation/bayesian_posterior_sample.png" alt="Bayesian Posteriors" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Bayesian Parameter Posteriors</div>
                        <div class="gallery-desc">Laplace-approximated parameter distributions providing epistemic uncertainty.</div>
                    </div>
                </div>
                <div class="gallery-card">
                    <img src="/outputs/evaluation/regression_residuals_ridge.png" alt="Regression Residuals" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Continuous Fatigue Residuals</div>
                        <div class="gallery-desc">Residual distribution for Ridge regression fatigue index estimation.</div>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <!-- Image Modal -->
    <div id="imgModal" class="img-modal" onclick="closeModal()">
        <span class="close-modal">&times;</span>
        <img id="modalImg" src="" alt="Full view">
    </div>

    <script>
        // Web Audio Synthesizer for Browser Alerts
        let audioCtx = null;
        let soundEnabled = true;
        let lastSoundTime = 0;

        function initAudio() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
        }

        function toggleAudio() {
            initAudio();
            soundEnabled = !soundEnabled;
            document.getElementById('sound-icon').innerText = soundEnabled ? '🔊' : '🔇';
            document.getElementById('sound-label').innerText = soundEnabled ? 'Audio: ON' : 'Audio: MUTE';
            const btn = document.getElementById('btn-sound');
            if (soundEnabled) {
                btn.style.borderColor = 'var(--accent-green)';
                btn.style.color = 'var(--accent-green)';
            } else {
                btn.style.borderColor = 'var(--border-color)';
                btn.style.color = 'var(--text-secondary)';
            }
        }

        function playAlertTone(level) {
            if (!soundEnabled) return;
            initAudio();
            const now = Date.now();
            if (now - lastSoundTime < 1100) return; // 1.1s cooldown
            lastSoundTime = now;

            try {
                if (level === 1) {
                    // Warning Chime: Dual frequency 880 -> 1100 Hz
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
                    osc.frequency.exponentialRampToValueAtTime(1100, audioCtx.currentTime + 0.25);
                    gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.35);
                } else if (level === 2) {
                    // Critical Alarm: Pulsed Urgent Buzzer 1300 Hz
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(1300, audioCtx.currentTime);
                    gain.gain.setValueAtTime(0.4, audioCtx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.45);
                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.45);
                }
            } catch (err) {
                console.debug('Audio trigger notice:', err);
            }
        }

        function triggerCalibration() {
            initAudio();
            fetch('/api/calibrate')
                .then(res => res.json())
                .then(data => {
                    const banner = document.getElementById('alert-banner');
                    banner.className = 'alert-banner alert-level-1';
                    document.getElementById('alert-text').innerText = 'CALIBRATING BASELINE (LOOK FORWARD NORMALLY)...';
                })
                .catch(err => console.error('Calibration error:', err));
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
            
            event.target.classList.add('active');
            const targetPane = document.getElementById(tabId);
            if (targetPane) targetPane.classList.add('active');
        }

        function changeModel(modelName) {
            fetch('/api/set_model?model=' + encodeURIComponent(modelName))
                .then(res => res.json())
                .then(data => console.log('Model switched:', data))
                .catch(err => console.error('Error changing model:', err));
        }

        // Live Telemetry Poller
        function pollTelemetry() {
            fetch('/api/telemetry')
                .then(res => res.json())
                .then(data => {
                    if (!data || Object.keys(data).length === 0) return;

                    // FPS & Latency
                    document.getElementById('fps-badge').innerText = `FPS: ${data.fps || '--'} | Latency: ${(data.latency_ms || 0).toFixed(1)} ms`;

                    // Values
                    document.getElementById('val-ear').innerText = (data.ear || 0).toFixed(2);
                    document.getElementById('val-mar').innerText = (data.mar || 0).toFixed(2);
                    document.getElementById('val-perclos').innerText = ((data.perclos || 0) * 100).toFixed(1) + '%';
                    document.getElementById('val-fatigue').innerText = ((data.fatigue_score || 0) * 100).toFixed(1) + '%';
                    document.getElementById('val-pose').innerText = `P: ${(data.pitch || 0).toFixed(0)}° | Y: ${(data.yaw || 0).toFixed(0)}° | R: ${(data.roll || 0).toFixed(0)}°`;

                    // Progress Bars
                    const earPct = Math.min(100, Math.max(0, ((data.ear || 0) / 0.45) * 100));
                    const marPct = Math.min(100, Math.max(0, ((data.mar || 0) / 0.85) * 100));
                    const perclosPct = Math.min(100, Math.max(0, (data.perclos || 0) * 100));
                    const fatiguePct = Math.min(100, Math.max(0, (data.fatigue_score || 0) * 100));

                    const barEar = document.getElementById('bar-ear');
                    barEar.style.width = earPct + '%';
                    barEar.style.backgroundColor = (data.ear < 0.23) ? 'var(--accent-red)' : 'var(--accent-green)';

                    const barMar = document.getElementById('bar-mar');
                    barMar.style.width = marPct + '%';
                    barMar.style.backgroundColor = (data.mar > 0.60) ? 'var(--accent-red)' : 'var(--accent-green)';

                    const barPerclos = document.getElementById('bar-perclos');
                    barPerclos.style.width = perclosPct + '%';
                    barPerclos.style.backgroundColor = (data.perclos > 0.20) ? 'var(--accent-red)' : 'var(--accent-green)';

                    const barFatigue = document.getElementById('bar-fatigue');
                    barFatigue.style.width = fatiguePct + '%';
                    barFatigue.style.backgroundColor = (data.fatigue_score > 0.65) ? 'var(--accent-red)' : ((data.fatigue_score > 0.40) ? 'var(--accent-yellow)' : 'var(--accent-green)');

                    // Status Banner & Level
                    const level = data.alert_level || 0;
                    const banner = document.getElementById('alert-banner');
                    banner.className = `alert-banner alert-level-${level}`;
                    document.getElementById('alert-text').innerText = data.status_text || 'STATUS: DRIVER ALERT';
                    document.getElementById('alert-badge').innerText = `LEVEL ${level}`;

                    // Play Audio Tone if Alert Level > 0
                    if (level > 0) {
                        playAlertTone(level);
                    }

                    // Customer Critical Status Pills
                    const pBase = document.getElementById('pill-baseline');
                    if (data.calibrating) {
                        pBase.innerText = `Calibrating ${data.calibration_progress || 0}%`;
                        pBase.className = 'pill-value warn';
                    } else {
                        pBase.innerText = `${(data.baseline_ear || 0.32).toFixed(3)} (Ready)`;
                        pBase.className = 'pill-value good';
                    }

                    const pSpeech = document.getElementById('pill-speech');
                    if (data.is_speaking) {
                        pSpeech.innerText = '🗣️ Speaking';
                        pSpeech.className = 'pill-value good';
                    } else {
                        pSpeech.innerText = 'Silent';
                        pSpeech.className = 'pill-value';
                    }

                    const pLight = document.getElementById('pill-light');
                    pLight.innerText = data.lighting_quality || 'Optimal';
                    pLight.className = data.low_light ? 'pill-value warn' : 'pill-value good';

                    const pEye = document.getElementById('pill-eyewear');
                    if (data.eyewear_detected) {
                        pEye.innerText = '🕶️ Eyewear Active';
                        pEye.className = 'pill-value warn';
                    } else {
                        pEye.innerText = 'Normal';
                        pEye.className = 'pill-value good';
                    }
                })
                .catch(err => console.debug('Telemetry poll notice:', err));
        }

        setInterval(pollTelemetry, 150);

        // Modal functions
        function openModal(src) {
            document.getElementById('modalImg').src = src;
            document.getElementById('imgModal').style.display = 'flex';
        }

        function closeModal() {
            document.getElementById('imgModal').style.display = 'none';
        }
    </script>
</body>
</html>
"""


class StreamingHTTPHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP handler for live MJPEG video and telemetry REST API."""

    def do_GET(self):
        global global_pipeline, latest_frame_jpeg, latest_telemetry

        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))

        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while stream_active:
                if latest_frame_jpeg:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(latest_frame_jpeg)))
                        self.end_headers()
                        self.wfile.write(latest_frame_jpeg)
                        self.wfile.write(b"\r\n")
                    except Exception:
                        break
                time.sleep(0.035)

        elif self.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(latest_telemetry).encode("utf-8"))

        elif self.path == "/api/calibrate":
            with pipeline_lock:
                if global_pipeline:
                    global_pipeline.start_calibration()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "calibrating"}).encode("utf-8"))

        elif self.path.startswith("/api/set_model"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            new_model = query.get("model", ["ensemble"])[0]
            with pipeline_lock:
                if global_pipeline:
                    global_pipeline.primary_model_type = new_model
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "active_model": new_model}).encode("utf-8"))

        elif self.path == "/api/leaderboard":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            report_csv = config.EVALUATION_REPORT_PATH
            rows = []
            if report_csv.exists():
                import pandas as pd
                try:
                    df = pd.read_csv(report_csv)
                    rows = df.to_dict(orient="records")
                except Exception:
                    pass
            self.wfile.write(json.dumps(rows).encode("utf-8"))

        elif self.path.startswith("/outputs/"):
            rel_path = self.path.lstrip("/")
            file_path = PROJECT_ROOT / rel_path
            if file_path.exists() and file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles HTTP requests asynchronously in separate threads."""
    daemon_threads = True


def run_web_server(port: int = 8080, camera_idx: int = 0, use_simulation: bool = True):
    """Starts the real-time web server and background stream worker."""
    global global_pipeline, stream_active

    global_pipeline = DrowsinessDetectorPipeline(primary_model_type="ensemble", enable_audio=False)

    worker_thread = threading.Thread(
        target=background_stream_worker,
        args=(camera_idx, use_simulation),
        daemon=True,
    )
    worker_thread.start()

    server_address = ("", port)
    httpd = ThreadedHTTPServer(server_address, StreamingHTTPHandler)
    logger.info("=" * 70)
    logger.info(f"REAL-TIME DASHBOARD SERVER ACTIVE: http://localhost:{port}")
    logger.info("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping web dashboard server...")
    finally:
        stream_active = False
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Drowsiness Detection Web App")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--live", action="store_true", help="Use live webcam instead of simulation")
    args = parser.parse_args()

    run_web_server(port=args.port, camera_idx=args.camera, use_simulation=not args.live)
