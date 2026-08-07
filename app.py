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
from typing import Dict, Any, Optional, Tuple

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
last_client_post_time: float = 0.0
pipeline_lock = threading.Lock()
frame_condition = threading.Condition()
frame_version: int = 0
global_pipeline: Optional[DrowsinessDetectorPipeline] = None
latest_frame_jpeg: bytes = b""
latest_telemetry: Dict[str, Any] = {}
stream_active = True
sim_state_idx = 0


class AsyncVideoCapture:
    """Non-blocking asynchronous camera capture thread that eliminates hardware buffer latency."""

    def __init__(self, camera_idx: int = 0):
        self.camera_idx = camera_idx
        self.cap = None
        self.running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        self.has_new_frame = False

        if sys.platform.startswith("win"):
            try:
                self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception as e:
                logger.debug(f"DirectShow init notice: {e}")

        if self.cap is None or not self.cap.isOpened():
            try:
                self.cap = cv2.VideoCapture(camera_idx)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception as e:
                logger.debug(f"Default VideoCapture notice: {e}")

        if self.cap and self.cap.isOpened():
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def _capture_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest_frame = frame
                self.has_new_frame = True

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


server_async_cam = None

def background_stream_worker(camera_idx: int = 0, use_simulation: bool = True):
    """Generates continuous low-latency live frames and runs ML/HMM/Alert pipeline."""
    global latest_frame_jpeg, latest_telemetry, global_pipeline, stream_active, sim_state_idx, last_client_post_time, frame_version, server_async_cam

    if not use_simulation:
        server_async_cam = AsyncVideoCapture(camera_idx)
        if not server_async_cam.is_opened():
            logger.warning(f"Physical webcam {camera_idx} unavailable. Falling back to dynamic simulation stream.")
            use_simulation = True
            server_async_cam = None
        else:
            logger.info(f"Connected to physical webcam device index {camera_idx} via low-latency async grabber.")
    else:
        logger.info("Starting stream worker in synthetic simulation mode (webcam hardware free for browser).")

    sim_cycle = (
        ["alert"] * 70
        + ["drowsy"] * 80
        + ["sleeping"] * 60
        + ["alert"] * 50
    )

    failed_reads = 0
    while stream_active:
        # If client browser is actively POSTing frames via /api/process_frame, pause worker loop to avoid clashing
        if time.time() - last_client_post_time < 1.5:
            time.sleep(0.04)
            continue

        frame = None
        if not use_simulation and server_async_cam and server_async_cam.is_opened():
            ret, frame = server_async_cam.read_latest()
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads > 250:
                    logger.warning("Physical webcam did not return frames after 2.5s warmup. Releasing camera handle and switching to dynamic simulation fallback.")
                    try:
                        server_async_cam.release()
                    except Exception:
                        pass
                    server_async_cam = None
                    use_simulation = True
                time.sleep(0.01)
                continue
            failed_reads = 0
        else:
            state = sim_cycle[sim_state_idx % len(sim_cycle)]
            sim_state_idx += 1
            frame = generate_synthetic_driver_frame(state)
            cv2.putText(
                frame,
                f"SIMULATED DRIVER SCENARIO: {state.upper()}",
                (15, 460),
                cv2.FONT_HERSHEY_DUPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

        if frame is not None and global_pipeline is not None:
            with pipeline_lock:
                hud_frame, telem = global_pipeline.process_frame(frame)
                latest_telemetry = telem
                ret, jpeg = cv2.imencode(".jpg", hud_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                if ret:
                    with frame_condition:
                        latest_frame_jpeg = jpeg.tobytes()
                        frame_version += 1
                        frame_condition.notify_all()

        time.sleep(0.005)

    if server_async_cam:
        try:
            server_async_cam.release()
        except Exception:
            pass
        server_async_cam = None


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Safety AI &mdash; Real-Time Drowsiness Detection System</title>
    <meta name="description" content="Production-grade real-time drowsiness detection & alert system with live video telemetry, multi-model ML inference, and interactive dashboard.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800;900&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* ========== DESIGN TOKENS ========== */
        :root {
            --bg-deep: #050a18;
            --bg-surface: rgba(12, 19, 36, 0.82);
            --bg-card: rgba(15, 23, 46, 0.72);
            --bg-card-hover: rgba(20, 30, 56, 0.85);
            --bg-elevated: rgba(22, 34, 62, 0.65);
            --border: rgba(56, 80, 135, 0.28);
            --border-hover: rgba(80, 120, 200, 0.45);
            --border-glow: rgba(0, 200, 255, 0.35);
            --accent: #00d4ff;
            --accent-2: #6366f1;
            --accent-gradient: linear-gradient(135deg, #00d4ff 0%, #6366f1 100%);
            --accent-gradient-h: linear-gradient(90deg, #00d4ff 0%, #6366f1 100%);
            --success: #10b981;
            --success-bg: rgba(16, 185, 129, 0.12);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.12);
            --danger: #f43f5e;
            --danger-bg: rgba(244, 63, 94, 0.15);
            --text-1: #e8edf5;
            --text-2: #8896b3;
            --text-3: #5a6a8a;
            --radius: 14px;
            --radius-sm: 8px;
            --radius-lg: 20px;
            --shadow: 0 8px 32px rgba(0,0,0,0.45);
            --shadow-glow: 0 0 30px rgba(0, 212, 255, 0.12);
            --transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
            --font-display: 'Orbitron', var(--font-sans);
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
        }

        /* ========== RESET ========== */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { scroll-behavior: smooth; }
        body {
            background: var(--bg-deep);
            color: var(--text-1);
            font-family: var(--font-sans);
            min-height: 100vh;
            line-height: 1.6;
            overflow-x: hidden;
        }

        /* ========== CUSTOM SCROLLBAR ========== */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-deep); }
        ::-webkit-scrollbar-thumb { background: rgba(100, 120, 180, 0.35); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

        /* ========== ANIMATED BACKGROUND ========== */
        .bg-grid {
            position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background-image:
                radial-gradient(circle at 1px 1px, rgba(100, 130, 200, 0.08) 1px, transparent 0);
            background-size: 40px 40px;
        }
        .bg-grid::before {
            content: ''; position: absolute; inset: 0;
            background: radial-gradient(ellipse at 20% 10%, rgba(0, 212, 255, 0.07) 0%, transparent 50%),
                        radial-gradient(ellipse at 80% 90%, rgba(99, 102, 241, 0.06) 0%, transparent 50%),
                        radial-gradient(ellipse at 50% 50%, rgba(0, 212, 255, 0.03) 0%, transparent 70%);
        }

        /* ========== LAYOUT WRAPPER ========== */
        .app-shell { position: relative; z-index: 1; display: flex; flex-direction: column; min-height: 100vh; }

        /* ========== HEADER ========== */
        header {
            background: rgba(8, 12, 24, 0.88);
            backdrop-filter: blur(20px) saturate(1.4);
            -webkit-backdrop-filter: blur(20px) saturate(1.4);
            border-bottom: 1px solid var(--border);
            padding: 14px 28px;
            display: flex; justify-content: space-between; align-items: center;
            position: sticky; top: 0; z-index: 100;
        }
        .brand { display: flex; align-items: center; gap: 16px; }
        .brand-logo {
            width: 42px; height: 42px; border-radius: 12px;
            background: var(--accent-gradient);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.3rem; box-shadow: 0 0 20px rgba(0, 212, 255, 0.25);
            animation: logoPulse 3s ease-in-out infinite;
        }
        @keyframes logoPulse {
            0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.25); }
            50% { box-shadow: 0 0 35px rgba(0, 212, 255, 0.45); }
        }
        .brand-text { display: flex; flex-direction: column; gap: 2px; }
        .brand-name {
            font-family: var(--font-display); font-weight: 800; font-size: 1.15rem;
            letter-spacing: 2px;
            background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .brand-sub { font-size: 0.7rem; color: var(--text-3); font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }
        .header-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        .btn {
            background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-2);
            padding: 7px 14px; border-radius: var(--radius-sm); font-size: 0.78rem; font-weight: 600;
            cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
            font-family: var(--font-sans); transition: all var(--transition); white-space: nowrap;
        }
        .btn:hover { border-color: var(--border-hover); color: var(--text-1); transform: translateY(-1px); background: var(--bg-card-hover); }
        .btn-accent {
            background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(99,102,241,0.12));
            border-color: rgba(0,212,255,0.4); color: var(--accent);
        }
        .btn-accent:hover { border-color: var(--accent); box-shadow: 0 0 18px rgba(0,212,255,0.2); }
        .badge-live {
            background: var(--success-bg); border: 1px solid var(--success); color: var(--success);
            padding: 5px 14px; border-radius: 20px; font-family: var(--font-display);
            font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px;
            display: flex; align-items: center; gap: 8px;
        }
        .badge-live::before {
            content: ''; width: 7px; height: 7px; background: var(--success); border-radius: 50%;
            box-shadow: 0 0 10px var(--success); animation: blink 1.5s infinite;
        }
        @keyframes blink {
            0%, 100% { opacity: .6; transform: scale(.85); }
            50% { opacity: 1; transform: scale(1.2); }
        }

        /* ========== STATS RIBBON ========== */
        .stats-ribbon {
            display: flex; gap: 0; border-bottom: 1px solid var(--border);
            background: rgba(8, 14, 28, 0.6); backdrop-filter: blur(10px);
            overflow-x: auto;
        }
        .stat-item {
            flex: 1; min-width: 140px; padding: 10px 20px;
            border-right: 1px solid var(--border);
            display: flex; align-items: center; gap: 10px;
        }
        .stat-item:last-child { border-right: none; }
        .stat-icon { font-size: 1.1rem; opacity: 0.7; }
        .stat-info { display: flex; flex-direction: column; }
        .stat-label { font-size: 0.65rem; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
        .stat-value { font-family: var(--font-mono); font-size: 0.82rem; font-weight: 600; color: var(--text-1); }

        /* ========== TAB NAVIGATION ========== */
        .tab-nav {
            display: flex; gap: 4px; padding: 10px 28px;
            background: rgba(8, 14, 28, 0.5);
            border-bottom: 1px solid var(--border);
        }
        .tab-btn {
            background: none; border: 1px solid transparent; color: var(--text-3);
            font-family: var(--font-sans); font-size: 0.85rem; font-weight: 600;
            padding: 9px 22px; border-radius: 10px; cursor: pointer;
            transition: all var(--transition); display: flex; align-items: center; gap: 8px;
        }
        .tab-btn:hover { color: var(--text-2); background: rgba(255,255,255,0.03); }
        .tab-btn.active {
            color: var(--accent); background: rgba(0,212,255,0.08);
            border-color: rgba(0,212,255,0.25); box-shadow: 0 0 15px rgba(0,212,255,0.08);
        }

        /* ========== MAIN CONTAINER ========== */
        .main { padding: 24px 28px; flex: 1; max-width: 1640px; margin: 0 auto; width: 100%; }
        .tab-pane { display: none; animation: fadeSlideIn 0.4s ease; }
        .tab-pane.active { display: block; }
        @keyframes fadeSlideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

        /* ========== ALERT BANNER ========== */
        .alert-banner {
            border-radius: var(--radius); padding: 14px 22px; margin-bottom: 22px;
            display: flex; align-items: center; justify-content: space-between;
            font-family: var(--font-display); font-weight: 700; font-size: 0.88rem;
            letter-spacing: 1.5px; transition: all 0.4s ease;
            backdrop-filter: blur(10px);
        }
        .alert-level-0 { background: var(--success-bg); border: 1px solid rgba(16,185,129,0.35); color: var(--success); }
        .alert-level-1 {
            background: var(--warning-bg); border: 1px solid rgba(245,158,11,0.5); color: var(--warning);
            box-shadow: 0 0 20px rgba(245,158,11,0.15); animation: pulseWarn 1.8s infinite;
        }
        .alert-level-2 {
            background: var(--danger-bg); border: 1px solid rgba(244,63,94,0.7); color: var(--danger);
            box-shadow: 0 0 35px rgba(244,63,94,0.35); animation: flashCrit 0.8s infinite;
        }
        @keyframes pulseWarn { 0%,100% { box-shadow: 0 0 15px rgba(245,158,11,0.12); opacity: 0.95; } 50% { box-shadow: 0 0 28px rgba(245,158,11,0.25); opacity: 1; } }
        @keyframes flashCrit { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.88; transform: scale(1.005); } }

        /* ========== CARDS ========== */
        .card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 22px;
            backdrop-filter: blur(12px) saturate(1.2);
            -webkit-backdrop-filter: blur(12px) saturate(1.2);
            box-shadow: var(--shadow); transition: all var(--transition);
        }
        .card:hover { border-color: var(--border-hover); }
        .card-head {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid var(--border);
        }
        .card-title {
            font-family: var(--font-display); font-size: 0.82rem; font-weight: 700;
            letter-spacing: 1.5px; color: var(--accent); text-transform: uppercase;
        }
        .card-badge { font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-3); }

        /* ========== LIVE GRID ========== */
        .grid-live { display: grid; grid-template-columns: 1.8fr 1fr; gap: 22px; }
        @media (max-width: 1100px) { .grid-live { grid-template-columns: 1fr; } }

        /* ========== VIDEO CONTAINER ========== */
        .video-wrap {
            position: relative; width: 100%; background: #000; border-radius: 10px;
            overflow: hidden; border: 1px solid rgba(255,255,255,0.06);
            aspect-ratio: 16 / 10; max-height: 540px;
        }
        .video-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
        /* Scanline overlay */
        .video-wrap::before {
            content: ''; position: absolute; inset: 0; z-index: 2; pointer-events: none;
            background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.04) 2px, rgba(0,0,0,0.04) 4px);
        }
        /* Corner brackets */
        .video-wrap::after {
            content: ''; position: absolute; inset: 8px; z-index: 3; pointer-events: none;
            border: 2px solid transparent;
            border-image: linear-gradient(135deg, var(--accent) 0%, transparent 25%, transparent 75%, var(--accent-2) 100%) 1;
            opacity: 0.5;
        }
        .corner { position: absolute; z-index: 4; width: 18px; height: 18px; pointer-events: none; }
        .corner--tl { top: 6px; left: 6px; border-top: 2px solid var(--accent); border-left: 2px solid var(--accent); }
        .corner--tr { top: 6px; right: 6px; border-top: 2px solid var(--accent); border-right: 2px solid var(--accent); }
        .corner--bl { bottom: 6px; left: 6px; border-bottom: 2px solid var(--accent-2); border-left: 2px solid var(--accent-2); }
        .corner--br { bottom: 6px; right: 6px; border-bottom: 2px solid var(--accent-2); border-right: 2px solid var(--accent-2); }

        /* ========== STATUS PILLS ========== */
        .pills { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 14px; }
        @media (max-width: 700px) { .pills { grid-template-columns: repeat(2, 1fr); } }
        .pill {
            background: rgba(10, 16, 30, 0.7); border: 1px solid var(--border);
            border-radius: var(--radius-sm); padding: 10px 12px;
            display: flex; flex-direction: column; gap: 3px;
        }
        .pill-label { font-size: 0.65rem; color: var(--text-3); font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
        .pill-val { font-family: var(--font-mono); font-size: 0.82rem; font-weight: 600; }
        .pill-val.ok { color: var(--success); }
        .pill-val.warn { color: var(--warning); }
        .pill-val.crit { color: var(--danger); }

        /* ========== TELEMETRY GAUGES ========== */
        .metric { margin-bottom: 16px; }
        .metric-head { display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 5px; }
        .metric-name { font-weight: 600; color: var(--text-2); }
        .metric-val { font-family: var(--font-mono); font-weight: 700; color: var(--text-1); }
        .gauge-track {
            background: rgba(30, 42, 70, 0.5); border-radius: 6px; height: 10px;
            overflow: hidden; position: relative;
        }
        .gauge-scale {
            display: flex; justify-content: space-between; font-family: var(--font-mono);
            font-size: 0.62rem; color: var(--text-3); margin-top: 4px;
        }
        .gauge-fill {
            height: 100%; border-radius: 6px;
            transition: width 0.2s ease, background 0.3s ease;
            position: relative;
            background: linear-gradient(90deg, var(--success) 0%, var(--success) 100%);
        }
        .gauge-fill::after {
            content: ''; position: absolute; top: 0; right: 0; bottom: 0; width: 30px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15));
            border-radius: 0 6px 6px 0;
        }
        .gauge-fill.warn-fill { background: linear-gradient(90deg, var(--warning), #f97316); }
        .gauge-fill.danger-fill { background: linear-gradient(90deg, var(--danger), #e11d48); }
        .gauge-marker {
            position: absolute; top: -1px; bottom: -1px; width: 2px;
            background: rgba(255,255,255,0.7); z-index: 2; border-radius: 1px;
            box-shadow: 0 0 4px rgba(255,255,255,0.8);
        }
        .divider { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
        .model-select {
            background: rgba(10, 16, 30, 0.8); border: 1px solid var(--border);
            color: var(--text-1); padding: 9px 14px; border-radius: var(--radius-sm);
            font-family: var(--font-sans); font-size: 0.82rem; font-weight: 600;
            outline: none; cursor: pointer; width: 100%; margin-top: 8px;
            transition: border-color var(--transition);
        }
        .model-select:focus { border-color: var(--accent); }
        .model-select option { background: #0c1324; }

        /* ========== RESULTS TAB ========== */
        .results-intro {
            margin-bottom: 28px; padding: 22px 26px;
            background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
            backdrop-filter: blur(12px);
        }
        .results-intro h2 {
            font-family: var(--font-display); font-size: 1.05rem; font-weight: 800;
            letter-spacing: 1.5px; margin-bottom: 8px;
            background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .results-intro p { color: var(--text-2); font-size: 0.88rem; line-height: 1.7; }

        /* ========== LEADERBOARD TABLE ========== */
        .table-wrap {
            background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
            backdrop-filter: blur(12px); overflow: hidden; margin-bottom: 32px;
        }
        .table-head { padding: 18px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; }
        .table-head-title { font-family: var(--font-display); font-size: 0.82rem; font-weight: 700; letter-spacing: 1.5px; color: var(--accent); }
        .leaderboard { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
        .leaderboard thead th {
            background: rgba(10, 16, 32, 0.6); padding: 11px 14px;
            text-align: left; font-weight: 700; color: var(--text-2); font-size: 0.72rem;
            text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid var(--border);
            white-space: nowrap; cursor: pointer; user-select: none; position: relative;
        }
        .leaderboard thead th:hover { color: var(--accent); }
        .leaderboard thead th.sort-asc::after { content: ' \\25B2'; font-size: 0.6rem; color: var(--accent); }
        .leaderboard thead th.sort-desc::after { content: ' \\25BC'; font-size: 0.6rem; color: var(--accent); }
        .leaderboard tbody td {
            padding: 10px 14px; border-bottom: 1px solid rgba(56,80,135,0.12);
            font-family: var(--font-mono); font-weight: 500; color: var(--text-1);
            white-space: nowrap;
        }
        .leaderboard tbody tr { transition: background var(--transition); }
        .leaderboard tbody tr:hover { background: rgba(0, 212, 255, 0.04); }
        .leaderboard tbody tr:first-child td { color: var(--accent); font-weight: 700; }
        .leaderboard .model-name { font-family: var(--font-sans); font-weight: 700; }
        .rank-badge {
            display: inline-flex; align-items: center; justify-content: center;
            width: 24px; height: 24px; border-radius: 6px; font-size: 0.7rem; font-weight: 800;
            font-family: var(--font-mono);
        }
        .rank-1 { background: linear-gradient(135deg, #ffd700, #f59e0b); color: #1a1a2e; }
        .rank-2 { background: linear-gradient(135deg, #c0c0c0, #94a3b8); color: #1a1a2e; }
        .rank-3 { background: linear-gradient(135deg, #cd7f32, #b45309); color: #fff; }
        .rank-n { background: rgba(100,120,180,0.15); color: var(--text-3); }
        .acc-perfect { color: var(--success); font-weight: 700; }

        /* ========== GALLERY ========== */
        .section-header {
            display: flex; align-items: center; gap: 12px;
            margin: 32px 0 16px; padding-bottom: 10px; border-bottom: 1px solid var(--border);
            cursor: pointer; user-select: none;
        }
        .section-header:first-of-type { margin-top: 0; }
        .section-icon { font-size: 1.2rem; }
        .section-title {
            font-family: var(--font-display); font-size: 0.8rem; font-weight: 700;
            letter-spacing: 1.5px; color: var(--text-2); text-transform: uppercase; flex: 1;
        }
        .section-toggle { font-size: 0.7rem; color: var(--text-3); transition: transform var(--transition); }
        .section-toggle.collapsed { transform: rotate(-90deg); }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 18px; margin-bottom: 8px; }
        .gallery.collapsed { display: none; }
        .g-card {
            background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
            overflow: hidden; transition: all var(--transition); cursor: pointer;
        }
        .g-card:hover {
            transform: translateY(-4px); border-color: var(--border-glow);
            box-shadow: 0 12px 35px rgba(0, 212, 255, 0.1);
        }
        .g-card img {
            width: 100%; height: 200px; object-fit: cover; display: block;
            border-bottom: 1px solid var(--border); background: #080e1c;
            transition: transform 0.4s ease;
        }
        .g-card:hover img { transform: scale(1.03); }
        .g-card-body { padding: 14px 16px; }
        .g-card-title { font-weight: 700; font-size: 0.88rem; color: var(--text-1); margin-bottom: 4px; }
        .g-card-desc { font-size: 0.75rem; color: var(--text-3); line-height: 1.5; }

        /* ========== MODAL ========== */
        .modal-overlay {
            display: none; position: fixed; inset: 0; z-index: 999;
            background: rgba(3, 6, 15, 0.92); backdrop-filter: blur(8px);
            justify-content: center; align-items: center;
            animation: modalIn 0.3s ease;
        }
        @keyframes modalIn { from { opacity: 0; } to { opacity: 1; } }
        .modal-overlay img {
            max-width: 90vw; max-height: 88vh; border-radius: 10px;
            box-shadow: 0 0 60px rgba(0, 212, 255, 0.15);
            animation: modalZoom 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes modalZoom { from { transform: scale(0.85); opacity: 0; } to { transform: scale(1); opacity: 1; } }
        .modal-close {
            position: absolute; top: 20px; right: 30px; color: var(--text-2);
            font-size: 32px; cursor: pointer; transition: color var(--transition);
            background: none; border: none; font-weight: 300; line-height: 1;
        }
        .modal-close:hover { color: #fff; }
        .modal-caption {
            position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%);
            font-family: var(--font-sans); font-size: 0.85rem; color: var(--text-2);
            background: rgba(10, 16, 30, 0.8); padding: 8px 20px; border-radius: 8px;
            backdrop-filter: blur(10px); white-space: nowrap;
        }

        /* ========== FOOTER ========== */
        footer {
            background: rgba(8, 12, 24, 0.7); border-top: 1px solid var(--border);
            padding: 16px 28px; display: flex; justify-content: space-between; align-items: center;
            font-size: 0.72rem; color: var(--text-3); backdrop-filter: blur(10px);
        }
        footer a { color: var(--accent); text-decoration: none; }
        footer a:hover { text-decoration: underline; }

        /* ========== RESPONSIVE ========== */
        @media (max-width: 900px) {
            header { padding: 12px 16px; flex-wrap: wrap; gap: 10px; }
            .header-actions { flex-wrap: wrap; }
            .main { padding: 16px; }
            .tab-nav { padding: 8px 16px; overflow-x: auto; }
            .stats-ribbon { flex-wrap: wrap; }
            .stat-item { min-width: 120px; }
            footer { flex-direction: column; gap: 6px; text-align: center; }
        }
        @media (max-width: 600px) {
            .gallery { grid-template-columns: 1fr; }
            .brand-name { font-size: 0.95rem; }
            .brand-sub { display: none; }
        }

        /* ========== UTILITY ========== */
        .stagger > * { animation: fadeSlideIn 0.4s ease both; }
        .stagger > *:nth-child(1) { animation-delay: 0.05s; }
        .stagger > *:nth-child(2) { animation-delay: 0.1s; }
        .stagger > *:nth-child(3) { animation-delay: 0.15s; }
        .stagger > *:nth-child(4) { animation-delay: 0.2s; }
        .stagger > *:nth-child(5) { animation-delay: 0.25s; }
        .stagger > *:nth-child(6) { animation-delay: 0.3s; }
    </style>
</head>
<body>

<div class="bg-grid"></div>

<div class="app-shell">

    <!-- ===== HEADER ===== -->
    <header>
        <div class="brand">
            <div class="brand-logo">&#128663;</div>
            <div class="brand-text">
                <span class="brand-name">DRIVER SAFETY AI</span>
                <span class="brand-sub">Real-Time Drowsiness Detection &amp; Edge Safety System</span>
            </div>
        </div>
        <div class="header-actions">
            <button id="btn-webcam" class="btn" onclick="toggleBrowserWebcam()">
                <span id="cam-icon">&#128247;</span>
                <span id="cam-label">My Webcam</span>
            </button>
            <button id="btn-mesh" class="btn" onclick="toggleMeshOverlay()">
                <span id="mesh-icon">&#127915;</span>
                <span id="mesh-label">HUD Mesh: OFF</span>
            </button>
            <button id="btn-sound" class="btn" onclick="toggleAudio()">
                <span id="sound-icon">&#128266;</span>
                <span id="sound-label">Audio: ON</span>
            </button>
            <button class="btn btn-accent" onclick="triggerCalibration()">
                <span>&#127919;</span>
                <span>Calibrate</span>
            </button>
            <div class="badge-live">LIVE</div>
        </div>
    </header>

    <!-- ===== STATS RIBBON ===== -->
    <div class="stats-ribbon">
        <div class="stat-item">
            <span class="stat-icon">&#9889;</span>
            <div class="stat-info">
                <span class="stat-label">Active Model</span>
                <span id="stat-model" class="stat-value">Stacking Ensemble</span>
            </div>
        </div>
        <div class="stat-item">
            <span class="stat-icon">&#128200;</span>
            <div class="stat-info">
                <span class="stat-label">Throughput</span>
                <span id="stat-fps" class="stat-value">-- FPS</span>
            </div>
        </div>
        <div class="stat-item">
            <span class="stat-icon">&#9201;</span>
            <div class="stat-info">
                <span class="stat-label">System Latency</span>
                <span id="stat-latency" class="stat-value">-- ms</span>
            </div>
        </div>
        <div class="stat-item">
            <span class="stat-icon">&#128338;</span>
            <div class="stat-info">
                <span class="stat-label">Uptime</span>
                <span id="stat-uptime" class="stat-value">00:00:00</span>
            </div>
        </div>
        <div class="stat-item">
            <span class="stat-icon">&#129504;</span>
            <div class="stat-info">
                <span class="stat-label">Edge AI Engine</span>
                <span class="stat-value" style="color:var(--success);">Active (INT8 Quantized)</span>
            </div>
        </div>
    </div>

    <!-- ===== TAB NAV ===== -->
    <nav class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('live-tab', this)">&#127909; Live Detection</button>
        <button class="tab-btn" onclick="switchTab('results-tab', this)">&#128202; Results &amp; Evaluation</button>
    </nav>

    <!-- ===== MAIN ===== -->
    <main class="main">

        <!-- TAB 1: LIVE DETECTION -->
        <div id="live-tab" class="tab-pane active">

            <div id="alert-banner" class="alert-banner alert-level-0">
                <span id="alert-text">STATUS: DRIVER ALERT &amp; ATTENTIVE</span>
                <span id="alert-badge" style="font-size:0.78rem;">LEVEL 0</span>
            </div>

            <div class="grid-live">
                <!-- Video Card -->
                <div class="card">
                    <div class="card-head">
                        <span class="card-title">Live Cockpit View</span>
                        <span id="cam-badge" class="card-badge">HD Vision Stream</span>
                    </div>
                    <div class="video-wrap" id="video-wrap">
                        <div class="corner corner--tl"></div>
                        <div class="corner corner--tr"></div>
                        <div class="corner corner--bl"></div>
                        <div class="corner corner--br"></div>
                        <img src="/video_feed" alt="Real-Time Driver Stream" id="video-stream">
                        <video id="browser-video" autoplay playsinline muted style="display:none; width:100%; height:100%; object-fit:cover; border-radius:10px;"></video>
                        <canvas id="hud-overlay-canvas" style="display:none; position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; border-radius:10px; z-index:5;"></canvas>
                    </div>
                    <div class="pills stagger">
                        <div class="pill">
                            <span class="pill-label">Baseline EAR</span>
                            <span id="pill-baseline" class="pill-val ok">0.320 (Ready)</span>
                        </div>
                        <div class="pill">
                            <span class="pill-label">Speech Filter</span>
                            <span id="pill-speech" class="pill-val ok">Silent</span>
                        </div>
                        <div class="pill">
                            <span class="pill-label">Cabin Lighting</span>
                            <span id="pill-light" class="pill-val ok">Optimal</span>
                        </div>
                        <div class="pill">
                            <span class="pill-label">Eyewear Analysis</span>
                            <span id="pill-eyewear" class="pill-val ok">Normal</span>
                        </div>
                    </div>
                </div>

                <!-- Telemetry Card -->
                <div class="card">
                    <div class="card-head">
                        <span class="card-title">Biometric Telemetry</span>
                        <span class="card-badge">Sensor Fusion</span>
                    </div>

                    <!-- EAR -->
                    <div class="metric">
                        <div class="metric-head">
                            <span class="metric-name">Eye Aspect Ratio (EAR)</span>
                            <span id="val-ear" class="metric-val">0.32</span>
                        </div>
                        <div class="gauge-track">
                            <div id="bar-ear" class="gauge-fill" style="width:70%;"></div>
                            <div class="gauge-marker" style="left:51%;" title="Threshold: 0.23"></div>
                        </div>
                        <div class="gauge-scale">
                            <span>0.00 (Closed)</span>
                            <span style="color:var(--accent-2);">0.23 Thresh</span>
                            <span>0.45 (Open)</span>
                        </div>
                    </div>

                    <!-- MAR -->
                    <div class="metric">
                        <div class="metric-head">
                            <span class="metric-name">Mouth Aspect Ratio (MAR)</span>
                            <span id="val-mar" class="metric-val">0.22</span>
                        </div>
                        <div class="gauge-track">
                            <div id="bar-mar" class="gauge-fill" style="width:25%;"></div>
                            <div class="gauge-marker" style="left:65%;" title="Threshold: 0.55"></div>
                        </div>
                        <div class="gauge-scale">
                            <span>0.00 (Closed)</span>
                            <span style="color:var(--accent-2);">0.55 Yawn Thresh</span>
                            <span>0.85 (Open)</span>
                        </div>
                    </div>

                    <!-- PERCLOS -->
                    <div class="metric">
                        <div class="metric-head">
                            <span class="metric-name">PERCLOS (% Eye Closure)</span>
                            <span id="val-perclos" class="metric-val">0.0%</span>
                        </div>
                        <div class="gauge-track">
                            <div id="bar-perclos" class="gauge-fill" style="width:0%;"></div>
                            <div class="gauge-marker" style="left:20%;" title="Threshold: 20%"></div>
                        </div>
                        <div class="gauge-scale">
                            <span>0% (Alert)</span>
                            <span style="color:var(--accent-2);">20% Thresh</span>
                            <span>100% (Sleep)</span>
                        </div>
                    </div>

                    <!-- Fatigue -->
                    <div class="metric">
                        <div class="metric-head">
                            <span class="metric-name">Continuous Fatigue Index</span>
                            <span id="val-fatigue" class="metric-val">0.0%</span>
                        </div>
                        <div class="gauge-track">
                            <div id="bar-fatigue" class="gauge-fill" style="width:0%;"></div>
                            <div class="gauge-marker" style="left:70%;" title="Critical: 70%"></div>
                        </div>
                        <div class="gauge-scale">
                            <span>0% (Vigilant)</span>
                            <span style="color:var(--warning);">45% Caution</span>
                            <span style="color:var(--danger);">70% Critical</span>
                        </div>
                    </div>

                    <hr class="divider">

                    <!-- Head Pose & Gaze -->
                    <div class="metric">
                        <div class="metric-head">
                            <span class="metric-name">Head Pose &amp; Gaze</span>
                            <span id="val-pose-direction" style="font-weight:700; color:var(--success);">Facing Ahead (Attentive)</span>
                        </div>
                        <div id="val-pose" style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-3); margin-top:3px;">
                            Pitch: 0&deg; | Yaw: 0&deg; | Roll: 0&deg;
                        </div>
                    </div>

                    <hr class="divider">

                    <!-- Model Selector -->
                    <label for="model-selector" class="metric-name" style="display:block;">Active Inference Model</label>
                    <select id="model-selector" class="model-select" onchange="changeModel(this.value)">
                        <option value="ensemble">Stacking Ensemble (RF + SVM + Bayes)</option>
                        <option value="rf">Random Forest (Bagging Ensemble)</option>
                        <option value="bayes">Bayesian Logistic Regression</option>
                    </select>
                </div>
            </div>
        </div>

        <!-- TAB 2: RESULTS & EVALUATION -->
        <div id="results-tab" class="tab-pane">

            <div class="results-intro">
                <h2>Model Evaluation &amp; Benchmark Suite</h2>
                <p>Comprehensive evaluation artifacts generated across Machine Learning Units 1&ndash;5: Exploratory Data Analysis, Linear Models &amp; Bayesian Posteriors, Dimensionality Reduction &amp; Clustering, Temporal Markov Chains, and Ensemble Classifiers. All models were evaluated on 801 stratified test samples.</p>
            </div>

            <!-- LEADERBOARD -->
            <div class="table-wrap">
                <div class="table-head">
                    <span class="table-head-title">&#127942; Unified Multi-Model Benchmark Leaderboard</span>
                </div>
                <div style="overflow-x:auto;">
                    <table class="leaderboard" id="leaderboard-table">
                        <thead>
                            <tr>
                                <th data-col="rank">#</th>
                                <th data-col="name">Model</th>
                                <th data-col="accuracy">Accuracy</th>
                                <th data-col="f1">Macro F1</th>
                                <th data-col="precision">Precision</th>
                                <th data-col="recall">Recall</th>
                                <th data-col="auc">ROC AUC</th>
                                <th data-col="latency">Latency</th>
                                <th data-col="fps">Throughput</th>
                                <th data-col="size">Size</th>
                            </tr>
                        </thead>
                        <tbody id="leaderboard-body">
                            <tr><td><span class="rank-badge rank-1">1</span></td><td class="model-name">Bayesian Logistic</td><td class="acc-perfect">100.00%</td><td class="acc-perfect">100.00%</td><td>100.00%</td><td>100.00%</td><td>1.0000</td><td>0.001 ms</td><td>742,043</td><td>1.9 KB</td></tr>
                            <tr><td><span class="rank-badge rank-1">1</span></td><td class="model-name">SVM (Linear)</td><td class="acc-perfect">100.00%</td><td class="acc-perfect">100.00%</td><td>100.00%</td><td>100.00%</td><td>1.0000</td><td>0.004 ms</td><td>225,002</td><td>10.5 KB</td></tr>
                            <tr><td><span class="rank-badge rank-1">1</span></td><td class="model-name">SVM (RBF)</td><td class="acc-perfect">100.00%</td><td class="acc-perfect">100.00%</td><td>100.00%</td><td>100.00%</td><td>1.0000</td><td>0.049 ms</td><td>20,630</td><td>20.7 KB</td></tr>
                            <tr><td><span class="rank-badge rank-1">1</span></td><td class="model-name">Random Forest</td><td class="acc-perfect">100.00%</td><td class="acc-perfect">100.00%</td><td>100.00%</td><td>100.00%</td><td>1.0000</td><td>0.101 ms</td><td>9,923</td><td>414.9 KB</td></tr>
                            <tr><td><span class="rank-badge rank-1">1</span></td><td class="model-name">Stacking Ensemble</td><td class="acc-perfect">100.00%</td><td class="acc-perfect">100.00%</td><td>100.00%</td><td>100.00%</td><td>1.0000</td><td>0.181 ms</td><td>5,514</td><td>277.7 KB</td></tr>
                            <tr><td><span class="rank-badge rank-2">6</span></td><td class="model-name">Decision Tree</td><td>99.25%</td><td>99.30%</td><td>99.37%</td><td>99.25%</td><td>0.9958</td><td>0.001 ms</td><td>1,062,289</td><td>4.6 KB</td></tr>
                            <tr><td><span class="rank-badge rank-3">7</span></td><td class="model-name">AdaBoost</td><td>99.13%</td><td>99.19%</td><td>99.15%</td><td>99.26%</td><td>0.9998</td><td>0.040 ms</td><td>25,044</td><td>29.1 KB</td></tr>
                            <tr><td><span class="rank-badge rank-n">8</span></td><td class="model-name">HMM (Viterbi)</td><td>96.00%</td><td>96.16%</td><td>96.22%</td><td>96.15%</td><td>1.0000</td><td>0.050 ms</td><td>19,930</td><td>1.4 KB</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- GALLERY SECTIONS -->

            <!-- Benchmark Overview -->
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-icon">&#127942;</span>
                <span class="section-title">Model Performance Benchmarks</span>
                <span class="section-toggle">&#9660;</span>
            </div>
            <div class="gallery stagger">
                <div class="g-card" onclick="openModal('/outputs/evaluation/benchmark_comparison.png', 'Unified Multi-Model Benchmark Comparison')">
                    <img src="/outputs/evaluation/benchmark_comparison.png" alt="Benchmark Comparison" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Multi-Model Benchmark</div>
                        <div class="g-card-desc">Accuracy, F1-Score, ROC-AUC and latency across all 8 algorithms.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/evaluation/multi_model_roc_curves.png', 'Multi-Model ROC Curves')">
                    <img src="/outputs/evaluation/multi_model_roc_curves.png" alt="ROC Curves" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">ROC Curves (One-vs-Rest)</div>
                        <div class="g-card-desc">Receiver Operating Characteristic curves demonstrating discriminative power.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/evaluation/all_models_confusion_matrices.png', 'All Models Confusion Matrices')">
                    <img src="/outputs/evaluation/all_models_confusion_matrices.png" alt="Confusion Matrices" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Confusion Matrices</div>
                        <div class="g-card-desc">True vs Predicted class distribution for Alert, Drowsy, and Sleeping states.</div>
                    </div>
                </div>
            </div>

            <!-- Unit 1: EDA -->
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-icon">&#128202;</span>
                <span class="section-title">Unit 1 &mdash; Exploratory Data Analysis</span>
                <span class="section-toggle">&#9660;</span>
            </div>
            <div class="gallery stagger">
                <div class="g-card" onclick="openModal('/outputs/eda/correlation_heatmap.png', 'Feature Correlation Heatmap')">
                    <img src="/outputs/eda/correlation_heatmap.png" alt="Correlation Heatmap" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Correlation Heatmap</div>
                        <div class="g-card-desc">Pearson correlation matrix for all biometric feature channels.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/eda/feature_histograms.png', 'Feature Distribution Histograms')">
                    <img src="/outputs/eda/feature_histograms.png" alt="Feature Histograms" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Feature Histograms</div>
                        <div class="g-card-desc">Distribution shape and spread of each extracted biometric feature.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/eda/feature_boxplots.png', 'Feature Boxplots')">
                    <img src="/outputs/eda/feature_boxplots.png" alt="Feature Boxplots" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Feature Boxplots</div>
                        <div class="g-card-desc">IQR analysis and outlier visualization across features.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/eda/class_distribution.png', 'Class Distribution')">
                    <img src="/outputs/eda/class_distribution.png" alt="Class Distribution" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Class Distribution</div>
                        <div class="g-card-desc">Sample count balance across Alert, Drowsy, and Sleeping classes.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/eda/scatter_plots.png', 'Feature Scatter Plots')">
                    <img src="/outputs/eda/scatter_plots.png" alt="Scatter Plots" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Scatter Plots</div>
                        <div class="g-card-desc">Pairwise feature relationships colored by driver alertness state.</div>
                    </div>
                </div>
            </div>

            <!-- Unit 2: Linear Models & SVM -->
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-icon">&#128640;</span>
                <span class="section-title">Unit 2 &mdash; Linear Models &amp; Support Vector Machines</span>
                <span class="section-toggle">&#9660;</span>
            </div>
            <div class="gallery stagger">
                <div class="g-card" onclick="openModal('/outputs/evaluation/svm_decision_boundary_rbf.png', 'RBF SVM Decision Boundary')">
                    <img src="/outputs/evaluation/svm_decision_boundary_rbf.png" alt="SVM Decision Boundary" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">RBF SVM Decision Boundary</div>
                        <div class="g-card-desc">Non-linear kernel SVM separating alert vs fatigued states in PCA space.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/evaluation/bayesian_posterior_sample.png', 'Bayesian Parameter Posteriors')">
                    <img src="/outputs/evaluation/bayesian_posterior_sample.png" alt="Bayesian Posteriors" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Bayesian Posteriors</div>
                        <div class="g-card-desc">Laplace-approximated parameter distributions with epistemic uncertainty.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/evaluation/regression_residuals_ridge.png', 'Ridge Regression Residuals')">
                    <img src="/outputs/evaluation/regression_residuals_ridge.png" alt="Regression Residuals" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Fatigue Score Residuals</div>
                        <div class="g-card-desc">Residual distribution for Ridge regression continuous fatigue index estimation.</div>
                    </div>
                </div>
            </div>

            <!-- Unit 3: Clustering -->
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-icon">&#127760;</span>
                <span class="section-title">Unit 3 &mdash; Dimensionality Reduction &amp; Clustering</span>
                <span class="section-toggle">&#9660;</span>
            </div>
            <div class="gallery stagger">
                <div class="g-card" onclick="openModal('/outputs/clustering/pca_scree_plot.png', 'PCA Scree Plot')">
                    <img src="/outputs/clustering/pca_scree_plot.png" alt="PCA Scree" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">PCA Scree Plot</div>
                        <div class="g-card-desc">Explained variance ratio per principal component with cumulative curve.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/clustering/pca_2d_projection.png', 'PCA 2D Projection')">
                    <img src="/outputs/clustering/pca_2d_projection.png" alt="PCA 2D" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">PCA 2D Projection</div>
                        <div class="g-card-desc">Training samples projected onto the first two principal components.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/clustering/kmeans_elbow_silhouette.png', 'K-Means Elbow & Silhouette')">
                    <img src="/outputs/clustering/kmeans_elbow_silhouette.png" alt="K-Means Elbow" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">K-Means Elbow &amp; Silhouette</div>
                        <div class="g-card-desc">Optimal cluster count via inertia elbow method and silhouette coefficient.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/clustering/kmeans_clusters_2d.png', 'K-Means Clusters 2D')">
                    <img src="/outputs/clustering/kmeans_clusters_2d.png" alt="K-Means Clusters" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">K-Means Clusters</div>
                        <div class="g-card-desc">Cluster assignments visualized in 2D PCA space with centroids.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/clustering/gmm_aic_bic.png', 'GMM AIC/BIC Curves')">
                    <img src="/outputs/clustering/gmm_aic_bic.png" alt="GMM AIC/BIC" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">GMM AIC / BIC</div>
                        <div class="g-card-desc">Model complexity selection via Akaike and Bayesian information criteria.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/clustering/gmm_clusters_ellipses.png', 'GMM Cluster Ellipses')">
                    <img src="/outputs/clustering/gmm_clusters_ellipses.png" alt="GMM Ellipses" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">GMM Cluster Ellipses</div>
                        <div class="g-card-desc">Gaussian Mixture component covariances shown as confidence ellipses.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/clustering/hierarchical_dendrogram.png', 'Hierarchical Dendrogram')">
                    <img src="/outputs/clustering/hierarchical_dendrogram.png" alt="Dendrogram" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Hierarchical Dendrogram</div>
                        <div class="g-card-desc">Ward linkage agglomerative clustering tree structure.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/clustering/hierarchical_clusters_2d.png', 'Hierarchical Clusters 2D')">
                    <img src="/outputs/clustering/hierarchical_clusters_2d.png" alt="Hierarchical Clusters" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Hierarchical Clusters</div>
                        <div class="g-card-desc">Agglomerative cluster assignments in reduced dimensional space.</div>
                    </div>
                </div>
            </div>

            <!-- Unit 4: HMM -->
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-icon">&#9201;</span>
                <span class="section-title">Unit 4 &mdash; Hidden Markov Model (Temporal Dynamics)</span>
                <span class="section-toggle">&#9660;</span>
            </div>
            <div class="gallery stagger">
                <div class="g-card" onclick="openModal('/outputs/evaluation/hmm_transition_matrix.png', 'HMM Transition Matrix')">
                    <img src="/outputs/evaluation/hmm_transition_matrix.png" alt="HMM Transition" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Transition Matrix Heatmap</div>
                        <div class="g-card-desc">Empirical state transition probabilities between driver alertness states.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/evaluation/hmm_state_sequence_decoding.png', 'HMM Viterbi State Decoding')">
                    <img src="/outputs/evaluation/hmm_state_sequence_decoding.png" alt="HMM Decoding" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Viterbi State Decoding</div>
                        <div class="g-card-desc">Temporal trajectory smoothing via dynamic programming eliminates jitter.</div>
                    </div>
                </div>
            </div>

            <!-- Unit 5: Trees & Ensembles -->
            <div class="section-header" onclick="toggleSection(this)">
                <span class="section-icon">&#127795;</span>
                <span class="section-title">Unit 5 &mdash; Tree-Based &amp; Ensemble Architectures</span>
                <span class="section-toggle">&#9660;</span>
            </div>
            <div class="gallery stagger">
                <div class="g-card" onclick="openModal('/outputs/evaluation/random_forest_feature_importance.png', 'Random Forest Feature Importance')">
                    <img src="/outputs/evaluation/random_forest_feature_importance.png" alt="RF Feature Importance" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">RF Feature Importance</div>
                        <div class="g-card-desc">Gini importance ranking of EAR, PERCLOS, MAR, and head dynamics.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/evaluation/random_forest_oob_trees.png', 'Random Forest OOB Error')">
                    <img src="/outputs/evaluation/random_forest_oob_trees.png" alt="RF OOB" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">OOB Error Convergence</div>
                        <div class="g-card-desc">Out-of-bag error reduction as ensemble size increases.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/evaluation/decision_tree_structure.png', 'Decision Tree Structure')">
                    <img src="/outputs/evaluation/decision_tree_structure.png" alt="Decision Tree" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">Decision Tree Structure</div>
                        <div class="g-card-desc">Cost-complexity pruned tree with Gini split criteria visualization.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/evaluation/decision_tree_feature_importances.png', 'Decision Tree Feature Importances')">
                    <img src="/outputs/evaluation/decision_tree_feature_importances.png" alt="DT Features" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">DT Feature Importances</div>
                        <div class="g-card-desc">Feature contribution ranking from the pruned decision tree classifier.</div>
                    </div>
                </div>
                <div class="g-card" onclick="openModal('/outputs/evaluation/adaboost_stagewise_error.png', 'AdaBoost Stagewise Error')">
                    <img src="/outputs/evaluation/adaboost_stagewise_error.png" alt="AdaBoost Error" loading="lazy">
                    <div class="g-card-body">
                        <div class="g-card-title">AdaBoost Stagewise Error</div>
                        <div class="g-card-desc">Training error reduction across sequential boosting iterations.</div>
                    </div>
                </div>
            </div>

        </div><!-- /results-tab -->

    </main>

    <!-- ===== MODAL ===== -->
    <div id="imgModal" class="modal-overlay" onclick="closeModal()">
        <button class="modal-close" onclick="closeModal()">&times;</button>
        <img id="modalImg" src="" alt="Full view">
        <div id="modalCaption" class="modal-caption"></div>
    </div>

    <!-- ===== FOOTER ===== -->
    <footer>
        <span>Driver Safety AI &mdash; Real-Time Drowsiness Detection &amp; Alert System</span>
        <span>Built with OpenCV &bull; MediaPipe &bull; scikit-learn &bull; NumPy | <a href="https://github.com/bankutech/Real-Time-Drowsiness-Detection-and-Alert-System" target="_blank">GitHub</a></span>
    </footer>

</div><!-- /app-shell -->

<script>
    // ===== UPTIME COUNTER =====
    const startTime = Date.now();
    function updateUptime() {
        const s = Math.floor((Date.now() - startTime) / 1000);
        const h = String(Math.floor(s / 3600)).padStart(2, '0');
        const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
        const sec = String(s % 60).padStart(2, '0');
        document.getElementById('stat-uptime').innerText = h + ':' + m + ':' + sec;
    }
    setInterval(updateUptime, 1000);

    // ===== MESH OVERLAY TOGGLE =====
    let meshOverlayActive = false;
    function toggleMeshOverlay() {
        meshOverlayActive = !meshOverlayActive;
        const btn = document.getElementById('btn-mesh');
        const lbl = document.getElementById('mesh-label');
        if (meshOverlayActive) {
            lbl.innerText = 'HUD Mesh: ON';
            btn.style.borderColor = 'var(--accent)';
            btn.style.color = 'var(--accent)';
        } else {
            lbl.innerText = 'HUD Mesh: OFF';
            btn.style.borderColor = 'var(--border)';
            btn.style.color = 'var(--text-2)';
        }
    }

    // ===== BROWSER WEBCAM STREAMING =====
    let browserCamStream = null;
    let browserCamActive = false;
    let isPostingFrame = false;
    let camCanvas = null;

    async function toggleBrowserWebcam() {
        initAudio();
        const vidEl = document.getElementById('browser-video');
        const hudCanvas = document.getElementById('hud-overlay-canvas');
        const imgEl = document.getElementById('video-stream');

        if (!browserCamActive) {
            try {
                await fetch('/api/release_camera').catch(() => {});
                
                let stream = null;
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
                        audio: false
                    });
                } catch (e1) {
                    stream = await navigator.mediaDevices.getUserMedia({ video: true });
                }

                browserCamStream = stream;

                vidEl.srcObject = browserCamStream;
                await vidEl.play();

                imgEl.style.display = 'none';
                vidEl.style.display = 'block';
                hudCanvas.style.display = 'block';
                hudCanvas.width = vidEl.videoWidth || 640;
                hudCanvas.height = vidEl.videoHeight || 480;

                if (!camCanvas) {
                    camCanvas = document.createElement('canvas');
                    camCanvas.width = 320;
                    camCanvas.height = 240;
                }

                browserCamActive = true;
                document.getElementById('cam-icon').innerText = String.fromCodePoint(0x23F9);
                document.getElementById('cam-label').innerText = 'Stop Webcam';
                document.getElementById('btn-webcam').style.borderColor = 'var(--accent)';
                document.getElementById('btn-webcam').style.color = 'var(--accent)';

                async function streamLoop() {
                    const ctx = camCanvas.getContext('2d', { willReadFrequently: true });
                    let lastFrameTime = performance.now();
                    while (browserCamActive) {
                        if (vidEl.readyState >= 2) {
                            try {
                                ctx.drawImage(vidEl, 0, 0, 320, 240);
                                const blob = await new Promise(r => camCanvas.toBlob(r, 'image/jpeg', 0.60));
                                if (blob && browserCamActive) {
                                    const tStart = performance.now();
                                    const resp = await fetch('/api/process_frame', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'image/jpeg' },
                                        body: blob
                                    });
                                    if (resp.ok && browserCamActive) {
                                        const telem = await resp.json();
                                        const now = performance.now();
                                        const dt = (now - lastFrameTime) / 1000.0;
                                        lastFrameTime = now;
                                        if (dt > 0) {
                                            telem.fps = Math.min(60.0, 1.0 / dt);
                                        }
                                        telem.latency_ms = now - tStart;
                                        renderTelemetryData(telem);
                                        drawClientHud(hudCanvas, telem);
                                    }
                                }
                            } catch (err) {
                                // Ignore transient drops
                            }
                        }
                        // Strict sequential backpressure: schedule next frame only after response is handled
                        await new Promise(r => requestAnimationFrame(r));
                    }
                }
                streamLoop();
            } catch (err) {
                alert('Webcam access notice: ' + err.message);
            }
        } else {
            browserCamActive = false;
            if (browserCamStream) {
                browserCamStream.getTracks().forEach(t => t.stop());
                browserCamStream = null;
            }
            vidEl.style.display = 'none';
            hudCanvas.style.display = 'none';
            imgEl.style.display = 'block';

            document.getElementById('cam-icon').innerText = String.fromCodePoint(0x1F4F7);
            document.getElementById('cam-label').innerText = 'My Webcam';
            document.getElementById('btn-webcam').style.borderColor = 'var(--border)';
            document.getElementById('btn-webcam').style.color = 'var(--text-2)';
        }
    }

    function drawClientHud(canvas, d) {
        if (!canvas || !d) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const w = canvas.width;
        const h = canvas.height;

        const lvl = d.alert_level || 0;

        // 1. Tiered Perimeter Border Indicator
        if (lvl === 2) {
            ctx.lineWidth = 6;
            ctx.strokeStyle = '#f43f5e';
            ctx.strokeRect(0, 0, w, h);
        } else if (lvl === 1) {
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#f59e0b';
            ctx.strokeRect(0, 0, w, h);
        }

        // 2. Minimalist Top Status Pill Banner
        ctx.fillStyle = 'rgba(10, 16, 30, 0.72)';
        ctx.fillRect(0, 0, w, 36);

        // Status indicator dot
        const dotColor = lvl === 2 ? '#f43f5e' : (lvl === 1 ? '#f59e0b' : '#10b981');
        ctx.fillStyle = dotColor;
        ctx.beginPath();
        ctx.arc(18, 18, 5, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px "Inter", sans-serif';
        const stText = d.status_text || 'DRIVER STATUS: ALERT';
        ctx.fillText(stText, 32, 22);

        // Right side badges: Head Pose Direction & Eyewear
        const poseDir = d.head_pose_direction || 'Facing Ahead';
        const eyeBadge = d.eyewear_detected ? ' [Glasses]' : '';
        const badgeStr = poseDir + eyeBadge;
        ctx.fillStyle = 'rgba(180, 225, 255, 0.9)';
        ctx.font = '11px "JetBrains Mono", monospace';
        const tw = ctx.measureText(badgeStr).width;
        ctx.fillText(badgeStr, w - tw - 16, 22);

        // 3. Optional Mesh Wireframes (Only drawn if HUD Mesh is toggled ON)
        if (meshOverlayActive && d.landmarks) {
            const lm = d.landmarks;
            function drawPoly(pts, strokeColor) {
                if (!pts || pts.length === 0) return;
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(pts[0][0] * w, pts[0][1] * h);
                for (let i = 1; i < pts.length; i++) {
                    ctx.lineTo(pts[i][0] * w, pts[i][1] * h);
                }
                ctx.closePath();
                ctx.stroke();
            }

            drawPoly(lm.left_eye, 'rgba(0, 212, 255, 0.85)');
            drawPoly(lm.right_eye, 'rgba(0, 212, 255, 0.85)');
            drawPoly(lm.mouth, 'rgba(245, 158, 11, 0.85)');

            // 3D Nose Direction Indicator
            if (lm.nose_tip) {
                const nx = lm.nose_tip[0] * w;
                const ny = lm.nose_tip[1] * h;
                const pitch = (d.pitch || 0) * (Math.PI / 180);
                const yaw = (d.yaw || 0) * (Math.PI / 180);
                const len = 30;

                ctx.strokeStyle = '#f43f5e';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(nx, ny);
                ctx.lineTo(nx + len * Math.cos(yaw), ny);
                ctx.stroke();

                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(nx, ny);
                ctx.lineTo(nx, ny - len * Math.sin(pitch));
                ctx.stroke();

                ctx.fillStyle = '#00d4ff';
                ctx.beginPath();
                ctx.arc(nx, ny, 3, 0, 2 * Math.PI);
                ctx.fill();
            }

            // Face Reticle
            if (lm.face_box) {
                const bx1 = lm.face_box[0] * w, by1 = lm.face_box[1] * h;
                const bx2 = lm.face_box[2] * w, by2 = lm.face_box[3] * h;
                const cLen = 12;
                ctx.strokeStyle = 'rgba(0, 212, 255, 0.6)';
                ctx.lineWidth = 1.5;

                ctx.beginPath(); ctx.moveTo(bx1, by1 + cLen); ctx.lineTo(bx1, by1); ctx.lineTo(bx1 + cLen, by1); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(bx2 - cLen, by1); ctx.lineTo(bx2, by1); ctx.lineTo(bx2, by1 + cLen); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(bx1, by2 - cLen); ctx.lineTo(bx1, by2); ctx.lineTo(bx1 + cLen, by2); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(bx2 - cLen, by2); ctx.lineTo(bx2, by2); ctx.lineTo(bx2, by2 - cLen); ctx.stroke();
            }
        }
    }

    // ===== WEB AUDIO =====
    let audioCtx = null;
    let soundEnabled = true;
    let lastSoundTime = 0;

    function initAudio() {
        if (!audioCtx) { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
        if (audioCtx.state === 'suspended') { audioCtx.resume(); }
    }

    function toggleAudio() {
        initAudio();
        soundEnabled = !soundEnabled;
        document.getElementById('sound-icon').innerText = soundEnabled ? String.fromCodePoint(0x1F50A) : String.fromCodePoint(0x1F507);
        document.getElementById('sound-label').innerText = soundEnabled ? 'Audio: ON' : 'Audio: MUTE';
        const btn = document.getElementById('btn-sound');
        btn.style.borderColor = soundEnabled ? 'var(--success)' : 'var(--border)';
        btn.style.color = soundEnabled ? 'var(--success)' : 'var(--text-3)';
    }

    function playAlertTone(level) {
        if (!soundEnabled) return;
        initAudio();
        const now = Date.now();
        if (now - lastSoundTime < 1400) return;
        lastSoundTime = now;
        try {
            if (level === 1) {
                const osc = audioCtx.createOscillator(); const gain = audioCtx.createGain();
                osc.connect(gain); gain.connect(audioCtx.destination);
                osc.type = 'sine'; osc.frequency.setValueAtTime(880, audioCtx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(1100, audioCtx.currentTime + 0.25);
                gain.gain.setValueAtTime(0.20, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
                osc.start(); osc.stop(audioCtx.currentTime + 0.35);
            } else if (level === 2) {
                const osc = audioCtx.createOscillator(); const gain = audioCtx.createGain();
                osc.connect(gain); gain.connect(audioCtx.destination);
                osc.type = 'sawtooth'; osc.frequency.setValueAtTime(1300, audioCtx.currentTime);
                gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.45);
                osc.start(); osc.stop(audioCtx.currentTime + 0.45);
            }
        } catch (err) { console.debug('Audio:', err); }
    }

    // ===== CALIBRATION =====
    function triggerCalibration() {
        initAudio();
        fetch('/api/calibrate').then(r => r.json()).then(() => {
            document.getElementById('alert-banner').className = 'alert-banner alert-level-1';
            document.getElementById('alert-text').innerText = 'CALIBRATING BASELINE (LOOK FORWARD NORMALLY)...';
        }).catch(e => console.error('Calibration error:', e));
    }

    // ===== TABS =====
    function switchTab(tabId, btn) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        if (btn) btn.classList.add('active');
        const pane = document.getElementById(tabId);
        if (pane) pane.classList.add('active');
    }

    // ===== MODEL SWITCH =====
    const modelNames = { ensemble: 'Stacking Ensemble', rf: 'Random Forest', bayes: 'Bayesian Logistic' };
    function changeModel(v) {
        fetch('/api/set_model?model=' + encodeURIComponent(v)).then(r => r.json()).then(d => {
            document.getElementById('stat-model').innerText = modelNames[v] || v;
        }).catch(e => console.error('Model switch error:', e));
    }

    // ===== TELEMETRY RENDERER =====
    function renderTelemetryData(d) {
        if (!d || Object.keys(d).length === 0) return;

        const fps = d.fps !== undefined ? Number(d.fps).toFixed(1) : '--';
        const lat = d.latency_ms !== undefined ? Number(d.latency_ms).toFixed(1) : '--';
        document.getElementById('stat-fps').innerText = fps + ' FPS';
        document.getElementById('stat-latency').innerText = lat + ' ms';

        document.getElementById('val-ear').innerText = (d.ear || 0).toFixed(2);
        document.getElementById('val-mar').innerText = (d.mar || 0).toFixed(2);
        document.getElementById('val-perclos').innerText = ((d.perclos || 0) * 100).toFixed(1) + '%';
        document.getElementById('val-fatigue').innerText = ((d.fatigue_score || 0) * 100).toFixed(1) + '%';
        
        // Head Pose
        const poseDir = d.head_pose_direction || 'Facing Ahead';
        const poseEl = document.getElementById('val-pose-direction');
        poseEl.innerText = poseDir;
        if (poseDir.includes('Ahead') || poseDir.includes('Attentive')) {
            poseEl.style.color = 'var(--success)';
        } else if (poseDir.includes('Down') || poseDir.includes('Distracted')) {
            poseEl.style.color = 'var(--warning)';
        } else {
            poseEl.style.color = 'var(--accent-2)';
        }
        document.getElementById('val-pose').innerText = 'Pitch: ' + (d.pitch||0).toFixed(0) + '\u00B0 | Yaw: ' + (d.yaw||0).toFixed(0) + '\u00B0 | Roll: ' + (d.roll||0).toFixed(0) + '\u00B0';

        // Gauge fills
        const earP = Math.min(100, Math.max(0, ((d.ear||0) / 0.45) * 100));
        const marP = Math.min(100, Math.max(0, ((d.mar||0) / 0.85) * 100));
        const pclP = Math.min(100, Math.max(0, (d.perclos||0) * 100));
        const fatP = Math.min(100, Math.max(0, (d.fatigue_score||0) * 100));

        setGauge('bar-ear', earP, d.ear < 0.23 ? 'danger-fill' : '');
        setGauge('bar-mar', marP, d.mar > 0.55 ? 'danger-fill' : '');
        setGauge('bar-perclos', pclP, d.perclos > 0.20 ? 'danger-fill' : '');
        setGauge('bar-fatigue', fatP, d.fatigue_score >= 0.70 ? 'danger-fill' : (d.fatigue_score >= 0.45 ? 'warn-fill' : ''));

        // Tiered Alert Banner
        const lvl = d.alert_level || 0;
        document.getElementById('alert-banner').className = 'alert-banner alert-level-' + lvl;
        document.getElementById('alert-text').innerText = d.status_text || 'STATUS: DRIVER ALERT & ATTENTIVE';
        document.getElementById('alert-badge').innerText = 'LEVEL ' + lvl;
        if (lvl > 0) playAlertTone(lvl);

        // Status Pills
        const pBase = document.getElementById('pill-baseline');
        if (d.calibrating) { pBase.innerText = 'Calibrating ' + (d.calibration_progress||0) + '%'; pBase.className = 'pill-val warn'; }
        else { pBase.innerText = (d.baseline_ear||0.32).toFixed(3) + ' (Ready)'; pBase.className = 'pill-val ok'; }

        const pSp = document.getElementById('pill-speech');
        pSp.innerText = d.is_speaking ? (String.fromCodePoint(0x1F5E3) + ' Speaking') : 'Silent';
        pSp.className = d.is_speaking ? 'pill-val ok' : 'pill-val';

        const pLi = document.getElementById('pill-light');
        pLi.innerText = d.lighting_quality || 'Optimal';
        pLi.className = d.low_light ? 'pill-val warn' : 'pill-val ok';

        const pEy = document.getElementById('pill-eyewear');
        if (d.eyewear_detected) { pEy.innerText = String.fromCodePoint(0x1F576) + ' ' + (d.eyewear_label || 'Glasses'); pEy.className = 'pill-val warn'; }
        else { pEy.innerText = 'Normal'; pEy.className = 'pill-val ok'; }
    }

    function setGauge(id, pct, cls) {
        const el = document.getElementById(id);
        if (!el) return;
        el.style.width = pct + '%';
        el.className = 'gauge-fill' + (cls ? ' ' + cls : '');
    }

    function pollTelemetry() {
        if (!browserCamActive) {
            fetch('/api/telemetry').then(r => r.json()).then(renderTelemetryData).catch(() => {});
        }
    }

    setInterval(pollTelemetry, 100);

    // ===== GALLERY SECTIONS =====
    function toggleSection(header) {
        const toggle = header.querySelector('.section-toggle');
        const gallery = header.nextElementSibling;
        if (gallery) gallery.classList.toggle('collapsed');
        if (toggle) toggle.classList.toggle('collapsed');
    }

    // ===== MODAL =====
    function openModal(src, caption) {
        document.getElementById('modalImg').src = src;
        document.getElementById('modalCaption').innerText = caption || '';
        document.getElementById('imgModal').style.display = 'flex';
    }
    function closeModal() {
        document.getElementById('imgModal').style.display = 'none';
    }
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
</script>
</body>
</html>
"""


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


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
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()

            client_frame_ver = -1
            while stream_active:
                with frame_condition:
                    if frame_version == client_frame_ver:
                        frame_condition.wait(timeout=0.06)
                    if frame_version == client_frame_ver:
                        continue
                    current_jpeg = latest_frame_jpeg
                    client_frame_ver = frame_version

                if current_jpeg:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(current_jpeg)))
                        self.end_headers()
                        self.wfile.write(current_jpeg)
                        self.wfile.write(b"\r\n")
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                        break
                    except Exception:
                        break

        elif self.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(latest_telemetry, cls=NpEncoder).encode("utf-8"))

        elif self.path == "/api/release_camera":
            global server_async_cam
            with pipeline_lock:
                if server_async_cam:
                    try:
                        server_async_cam.release()
                    except Exception:
                        pass
                    server_async_cam = None
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "released"}).encode("utf-8"))

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
            self.wfile.write(json.dumps(rows, cls=NpEncoder).encode("utf-8"))

        elif self.path == "/api/onnx_benchmark":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                from src.onnx_exporter import export_all_models
                benchmarks = export_all_models()
            except Exception as e:
                benchmarks = [{"error": str(e)}]
            self.wfile.write(json.dumps(benchmarks, cls=NpEncoder).encode("utf-8"))

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

    def do_POST(self):
        global global_pipeline, latest_frame_jpeg, latest_telemetry, last_client_post_time, frame_version
        if self.path == "/api/process_frame":
            last_client_post_time = time.time()
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                nparr = np.frombuffer(post_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None and global_pipeline is not None:
                    with pipeline_lock:
                        _, telem = global_pipeline.process_frame(frame, render_hud=False)
                        latest_telemetry = telem
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(latest_telemetry, cls=NpEncoder).encode("utf-8"))
                    return
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                return
            except Exception as e:
                logger.error(f"Error processing client frame: {e}")
            try:
                self.send_response(400)
                self.end_headers()
            except Exception:
                pass
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
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--physical-cam", action="store_true", help="Lock physical webcam on backend server instead of browser")
    args = parser.parse_args()

    run_web_server(port=args.port, camera_idx=args.camera, use_simulation=not args.physical_cam)
