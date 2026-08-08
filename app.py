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
_JPEG_ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 75, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]


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
                ret, jpeg = cv2.imencode(".jpg", hud_frame, _JPEG_ENCODE_PARAMS)
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


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="fav-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#020617"/>
      <stop offset="100%" stop-color="#0b132b"/>
    </linearGradient>
    <linearGradient id="fav-neon" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00f0ff"/>
      <stop offset="50%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#6366f1"/>
    </linearGradient>
    <filter id="fav-glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="1.5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <rect width="64" height="64" rx="14" fill="url(#fav-bg)"/>
  <rect width="60" height="60" x="2" y="2" rx="12" fill="none" stroke="url(#fav-neon)" stroke-width="2" opacity="0.6"/>
  <circle cx="32" cy="32" r="23" fill="none" stroke="url(#fav-neon)" stroke-width="1.8" stroke-dasharray="4 3" opacity="0.5"/>
  <path d="M10 32C16 20 23 15 32 15C41 15 48 20 54 32C48 44 41 49 32 49C23 49 16 44 10 32Z" fill="#060e22" stroke="url(#fav-neon)" stroke-width="2.5" stroke-linejoin="round"/>
  <path d="M21 41L26 32H38L43 41" fill="none" stroke="#00f0ff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="25" cy="35" r="1.8" fill="#00f0ff"/>
  <circle cx="39" cy="35" r="1.8" fill="#00f0ff"/>
  <circle cx="32" cy="27" r="6" fill="#020617" stroke="#00f0ff" stroke-width="2"/>
  <circle cx="32" cy="27" r="3.2" fill="#00f0ff" filter="url(#fav-glow)"/>
</svg>"""

FAVICON_DATA_URI = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='fav-bg' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23020617'/%3E%3Cstop offset='100%25' stop-color='%230b132b'/%3E%3C/linearGradient%3E%3ClinearGradient id='fav-neon' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%2300f0ff'/%3E%3Cstop offset='50%25' stop-color='%2338bdf8'/%3E%3Cstop offset='100%25' stop-color='%236366f1'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='64' height='64' rx='14' fill='url(%23fav-bg)'/%3E%3Crect width='60' height='60' x='2' y='2' rx='12' fill='none' stroke='url(%23fav-neon)' stroke-width='2' opacity='0.6'/%3E%3Ccircle cx='32' cy='32' r='23' fill='none' stroke='url(%23fav-neon)' stroke-width='1.8' stroke-dasharray='4 3' opacity='0.5'/%3E%3Cpath d='M10 32C16 20 23 15 32 15C41 15 48 20 54 32C48 44 41 49 32 49C23 49 16 44 10 32Z' fill='%23060e22' stroke='url(%23fav-neon)' stroke-width='2.5' stroke-linejoin='round'/%3E%3Cpath d='M21 41L26 32H38L43 41' fill='none' stroke='%2300f0ff' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='25' cy='35' r='1.8' fill='%2300f0ff'/%3E%3Ccircle cx='39' cy='35' r='1.8' fill='%2300f0ff'/%3E%3Ccircle cx='32' cy='27' r='6' fill='%23020617' stroke='%2300f0ff' stroke-width='2'/%3E%3Ccircle cx='32' cy='27' r='3.2' fill='%2300f0ff'/%3E%3C/svg%3E"


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, viewport-fit=cover">
    <meta name="theme-color" content="#040714">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Driver Safety AI</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='fav-bg' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23020617'/%3E%3Cstop offset='100%25' stop-color='%230b132b'/%3E%3C/linearGradient%3E%3ClinearGradient id='fav-neon' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%2300f0ff'/%3E%3Cstop offset='50%25' stop-color='%2338bdf8'/%3E%3Cstop offset='100%25' stop-color='%236366f1'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='64' height='64' rx='14' fill='url(%23fav-bg)'/%3E%3Crect width='60' height='60' x='2' y='2' rx='12' fill='none' stroke='url(%23fav-neon)' stroke-width='2' opacity='0.6'/%3E%3Ccircle cx='32' cy='32' r='23' fill='none' stroke='url(%23fav-neon)' stroke-width='1.8' stroke-dasharray='4 3' opacity='0.5'/%3E%3Cpath d='M10 32C16 20 23 15 32 15C41 15 48 20 54 32C48 44 41 49 32 49C23 49 16 44 10 32Z' fill='%23060e22' stroke='url(%23fav-neon)' stroke-width='2.5' stroke-linejoin='round'/%3E%3Cpath d='M21 41L26 32H38L43 41' fill='none' stroke='%2300f0ff' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='25' cy='35' r='1.8' fill='%2300f0ff'/%3E%3Ccircle cx='39' cy='35' r='1.8' fill='%2300f0ff'/%3E%3Ccircle cx='32' cy='27' r='6' fill='%23020617' stroke='%2300f0ff' stroke-width='2'/%3E%3Ccircle cx='32' cy='27' r='3.2' fill='%2300f0ff'/%3E%3C/svg%3E">
    <link rel="shortcut icon" href="/favicon.ico" type="image/svg+xml">
    <link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='fav-bg' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23020617'/%3E%3Cstop offset='100%25' stop-color='%230b132b'/%3E%3C/linearGradient%3E%3ClinearGradient id='fav-neon' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%2300f0ff'/%3E%3Cstop offset='50%25' stop-color='%2338bdf8'/%3E%3Cstop offset='100%25' stop-color='%236366f1'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='64' height='64' rx='14' fill='url(%23fav-bg)'/%3E%3Crect width='60' height='60' x='2' y='2' rx='12' fill='none' stroke='url(%23fav-neon)' stroke-width='2' opacity='0.6'/%3E%3Ccircle cx='32' cy='32' r='23' fill='none' stroke='url(%23fav-neon)' stroke-width='1.8' stroke-dasharray='4 3' opacity='0.5'/%3E%3Cpath d='M10 32C16 20 23 15 32 15C41 15 48 20 54 32C48 44 41 49 32 49C23 49 16 44 10 32Z' fill='%23060e22' stroke='url(%23fav-neon)' stroke-width='2.5' stroke-linejoin='round'/%3E%3Cpath d='M21 41L26 32H38L43 41' fill='none' stroke='%2300f0ff' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='25' cy='35' r='1.8' fill='%2300f0ff'/%3E%3Ccircle cx='39' cy='35' r='1.8' fill='%2300f0ff'/%3E%3Ccircle cx='32' cy='27' r='6' fill='%23020617' stroke='%2300f0ff' stroke-width='2'/%3E%3Ccircle cx='32' cy='27' r='3.2' fill='%2300f0ff'/%3E%3C/svg%3E">
    <meta name="description" content="Production-grade real-time driver drowsiness detection & alert system with live video telemetry, biometric oscilloscope, multi-model ML inference, and interactive diagnostics.">
    
    <!-- Modern High-Tech Typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        /* ==========================================================================
           1. MODERN CYBER-COCKPIT DESIGN SYSTEM & TOKENS
           ========================================================================== */
        :root {
            --bg-void: #030611;
            --bg-deep: #060a18;
            --bg-surface: rgba(10, 16, 34, 0.78);
            --bg-card: rgba(13, 22, 46, 0.72);
            --bg-card-hover: rgba(18, 30, 62, 0.88);
            --bg-elevated: rgba(22, 36, 74, 0.65);
            --bg-glass: rgba(15, 25, 52, 0.45);
            
            --border-subtle: rgba(70, 100, 165, 0.22);
            --border-medium: rgba(85, 125, 205, 0.38);
            --border-glow: rgba(0, 240, 255, 0.45);
            --border-bright: rgba(255, 255, 255, 0.14);
            
            --cyan: #00f0ff;
            --cyan-glow: rgba(0, 240, 255, 0.35);
            --cyan-dim: rgba(0, 240, 255, 0.12);
            
            --indigo: #6366f1;
            --indigo-light: #818cf8;
            --indigo-glow: rgba(99, 102, 241, 0.35);
            
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.35);
            --success-bg: rgba(16, 185, 129, 0.12);
            
            --warning: #f59e0b;
            --warning-glow: rgba(245, 158, 11, 0.4);
            --warning-bg: rgba(245, 158, 11, 0.14);
            
            --danger: #f43f5e;
            --danger-glow: rgba(244, 63, 94, 0.5);
            --danger-bg: rgba(244, 63, 94, 0.16);
            
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --text-sub: #64748b;
            --text-accent: #38bdf8;
            
            --grad-brand: linear-gradient(135deg, #00f0ff 0%, #6366f1 100%);
            --grad-brand-h: linear-gradient(90deg, #00f0ff 0%, #818cf8 100%);
            --grad-accent: linear-gradient(135deg, rgba(0, 240, 255, 0.18), rgba(99, 102, 241, 0.18));
            --grad-danger: linear-gradient(90deg, #f43f5e, #e11d48);
            --grad-warn: linear-gradient(90deg, #f59e0b, #ea580c);
            --grad-success: linear-gradient(90deg, #10b981, #059669);
            
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 18px;
            --radius-xl: 24px;
            
            --shadow-subtle: 0 4px 20px rgba(0, 0, 0, 0.4);
            --shadow-card: 0 12px 36px -4px rgba(0, 0, 0, 0.55), 0 0 0 1px var(--border-subtle);
            --shadow-glow: 0 0 35px var(--cyan-dim);
            
            --font-display: 'Orbitron', -apple-system, sans-serif;
            --font-body: 'Plus Jakarta Sans', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            
            --transition-smooth: 0.28s cubic-bezier(0.16, 1, 0.3, 1);
            --transition-bounce: 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        /* ==========================================================================
           2. BASE RESETS & AMBIENT SCI-FI BACKGROUND
           ========================================================================== */
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            -webkit-tap-highlight-color: transparent;
        }

        html {
            scroll-behavior: smooth;
            font-size: 16px;
        }

        body {
            background-color: var(--bg-void);
            color: var(--text-main);
            font-family: var(--font-body);
            min-height: 100vh;
            min-height: -webkit-fill-available;
            line-height: 1.55;
            overflow-x: hidden;
            touch-action: manipulation;
            position: relative;
        }

        /* Dynamic Grid & Ambient Lighting Flairs */
        .ambient-background {
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            overflow: hidden;
        }

        .ambient-grid {
            position: absolute;
            inset: 0;
            background-image: 
                linear-gradient(to right, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
            background-size: 48px 48px;
            mask-image: radial-gradient(ellipse 80% 60% at 50% 35%, black 40%, transparent 80%);
            -webkit-mask-image: radial-gradient(ellipse 80% 60% at 50% 35%, black 40%, transparent 80%);
        }

        .ambient-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(90px);
            opacity: 0.35;
            pointer-events: none;
            animation: orbFloat 18s ease-in-out infinite alternate;
        }

        .orb-1 {
            width: 500px; height: 500px;
            background: radial-gradient(circle, rgba(0, 240, 255, 0.3), transparent 70%);
            top: -100px; left: -100px;
        }

        .orb-2 {
            width: 600px; height: 600px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.25), transparent 70%);
            top: 20%; right: -150px;
            animation-duration: 24s;
            animation-delay: -5s;
        }

        .orb-3 {
            width: 450px; height: 450px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.18), transparent 70%);
            bottom: -50px; left: 30%;
            animation-duration: 20s;
        }

        @keyframes orbFloat {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(40px, 60px) scale(1.1); }
            100% { transform: translate(-30px, -40px) scale(0.95); }
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-void); }
        ::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.25); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

        /* ==========================================================================
           3. APP SHELL & LAYOUT ARCHITECTURE
           ========================================================================== */
        .app-shell {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            padding-left: env(safe-area-inset-left);
            padding-right: env(safe-area-inset-right);
        }

        /* ==========================================================================
           4. ULTRA-MODERN COCKPIT HEADER
           ========================================================================== */
        header {
            background: rgba(6, 10, 24, 0.82);
            backdrop-filter: blur(24px) saturate(160%);
            -webkit-backdrop-filter: blur(24px) saturate(160%);
            border-bottom: 1px solid var(--border-subtle);
            padding: 12px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
            padding-top: max(12px, env(safe-area-inset-top));
            box-shadow: 0 4px 25px rgba(0,0,0,0.5);
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-icon-wrap {
            position: relative;
            width: 44px;
            height: 44px;
            border-radius: var(--radius-md);
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.15) 0%, rgba(99, 102, 241, 0.22) 100%);
            border: 1px solid rgba(0, 240, 255, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.25), inset 0 0 12px rgba(0, 240, 255, 0.08);
            flex-shrink: 0;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .brand-icon-wrap:hover {
            transform: scale(1.06) rotate(-2deg);
            border-color: var(--cyan);
            box-shadow: 0 0 28px rgba(0, 240, 255, 0.5), inset 0 0 16px rgba(0, 240, 255, 0.2);
        }

        .brand-logo-svg {
            width: 32px;
            height: 32px;
            display: block;
            filter: drop-shadow(0 0 6px rgba(0, 240, 255, 0.6));
            animation: pulseGlow 4s ease-in-out infinite alternate;
        }

        @keyframes pulseGlow {
            0% { filter: drop-shadow(0 0 4px rgba(0, 240, 255, 0.4)); }
            100% { filter: drop-shadow(0 0 8px rgba(0, 240, 255, 0.8)); }
        }

        .brand-title-wrap {
            display: flex;
            flex-direction: column;
            gap: 1px;
        }

        .brand-title {
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: 2px;
            background: var(--grad-brand-h);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .brand-subtitle {
            font-size: 0.68rem;
            color: var(--text-muted);
            font-weight: 600;
            letter-spacing: 0.6px;
            text-transform: uppercase;
        }

        .brand-subtitle span {
            color: var(--cyan);
        }

        /* Header Control Actions */
        .header-actions {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .btn-cockpit {
            background: var(--bg-surface);
            border: 1px solid var(--border-medium);
            color: var(--text-main);
            padding: 8px 14px;
            border-radius: var(--radius-sm);
            font-family: var(--font-body);
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 7px;
            transition: all var(--transition-smooth);
            user-select: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }

        .btn-cockpit:hover {
            border-color: var(--cyan);
            color: #fff;
            transform: translateY(-1px);
            background: var(--bg-card-hover);
            box-shadow: 0 4px 16px var(--cyan-dim);
        }

        .btn-cockpit:active {
            transform: translateY(1px) scale(0.98);
        }

        .btn-cockpit.btn-primary {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.15), rgba(99, 102, 241, 0.2));
            border-color: var(--cyan);
            color: var(--cyan);
            box-shadow: 0 0 16px rgba(0, 240, 255, 0.18);
        }

        .btn-cockpit.btn-primary:hover {
            background: var(--grad-brand);
            color: #030611;
            font-weight: 700;
            box-shadow: 0 0 24px var(--cyan-glow);
        }

        .btn-cockpit.btn-primary:hover .btn-icon {
            filter: drop-shadow(0 0 2px #030611);
        }

        .btn-cockpit.active-toggle {
            border-color: var(--cyan);
            color: var(--cyan);
            background: rgba(0, 240, 255, 0.14);
            box-shadow: 0 0 14px var(--cyan-dim);
        }

        .badge-live-pulse {
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.6);
            color: var(--success);
            padding: 6px 14px;
            border-radius: 20px;
            font-family: var(--font-display);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
        }

        .badge-live-pulse::before {
            content: '';
            width: 7px;
            height: 7px;
            background: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--success);
            animation: pulseRadar 1.6s infinite ease-out;
        }

        @keyframes pulseRadar {
            0% { transform: scale(0.8); opacity: 0.6; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
            50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.8); opacity: 0.6; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* Sound Wave Animation */
        .sound-wave {
            display: flex;
            align-items: center;
            gap: 2px;
            height: 12px;
        }
        .sound-wave span {
            width: 2px;
            background: currentColor;
            border-radius: 1px;
            animation: soundBar 1s ease-in-out infinite;
        }
        .sound-wave span:nth-child(1) { height: 4px; animation-delay: 0.1s; }
        .sound-wave span:nth-child(2) { height: 10px; animation-delay: 0.3s; }
        .sound-wave span:nth-child(3) { height: 7px; animation-delay: 0.2s; }
        .sound-wave span:nth-child(4) { height: 12px; animation-delay: 0.4s; }

        @keyframes soundBar {
            0%, 100% { transform: scaleY(0.4); }
            50% { transform: scaleY(1.2); }
        }

        /* ==========================================================================
           5. COCKPIT TELEMETRY RIBBON (KPI STRIP)
           ========================================================================== */
        .stats-ribbon-wrap {
            background: rgba(8, 14, 30, 0.85);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-subtle);
            padding: 0 28px;
            position: relative;
            z-index: 10;
        }

        .stats-ribbon {
            display: grid;
            grid-template-columns: minmax(220px, 1.35fr) minmax(130px, 0.9fr) minmax(130px, 0.9fr) minmax(140px, 0.9fr) minmax(190px, 1.15fr);
            gap: 1px;
            background: var(--border-subtle);
            max-width: 1640px;
            margin: 0 auto;
            overflow-x: auto;
        }

        .stat-card {
            background: rgba(7, 12, 28, 0.94);
            padding: 10px 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: background var(--transition-smooth);
            position: relative;
            overflow: hidden;
            min-width: 0;
        }

        .stat-card:hover {
            background: rgba(14, 24, 52, 0.95);
        }

        .stat-card::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: transparent;
            transition: background var(--transition-smooth);
        }

        .stat-card:hover::after {
            background: var(--grad-brand-h);
        }

        .stat-icon-box {
            width: 34px;
            height: 34px;
            border-radius: var(--radius-sm);
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            color: var(--cyan);
            flex-shrink: 0;
        }

        .stat-content {
            display: flex;
            flex-direction: column;
            min-width: 0;
            flex: 1;
        }

        .stat-label {
            font-size: 0.62rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .stat-value {
            font-family: var(--font-mono);
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-main);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .stat-badge-sub {
            font-size: 0.58rem;
            padding: 1px 5px;
            border-radius: 4px;
            background: rgba(0, 240, 255, 0.12);
            color: var(--cyan);
            border: 1px solid rgba(0, 240, 255, 0.3);
            font-family: var(--font-body);
            font-weight: 700;
            flex-shrink: 0;
            display: inline-flex;
            align-items: center;
            line-height: 1.2;
        }

        /* ==========================================================================
           6. MODERN TAB NAVIGATION
           ========================================================================== */
        .tab-nav-bar {
            background: rgba(6, 10, 22, 0.7);
            border-bottom: 1px solid var(--border-subtle);
            padding: 8px 28px;
        }

        .tab-nav-inner {
            max-width: 1640px;
            margin: 0 auto;
            display: flex;
            gap: 8px;
        }

        .tab-pill {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            font-family: var(--font-body);
            font-size: 0.85rem;
            font-weight: 600;
            padding: 9px 20px;
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all var(--transition-smooth);
            display: flex;
            align-items: center;
            gap: 8px;
            user-select: none;
        }

        .tab-pill:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.04);
        }

        .tab-pill.active {
            color: var(--cyan);
            background: rgba(0, 240, 255, 0.08);
            border-color: rgba(0, 240, 255, 0.35);
            box-shadow: 0 0 18px rgba(0, 240, 255, 0.1);
        }

        .tab-pill .pill-badge {
            font-size: 0.65rem;
            padding: 2px 7px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
        }

        .tab-pill.active .pill-badge {
            background: var(--cyan);
            color: #030611;
            font-weight: 700;
        }

        /* ==========================================================================
           7. MAIN CONTENT AREA & PANELS
           ========================================================================== */
        main.main-content {
            padding: 24px 28px;
            flex: 1;
            max-width: 1640px;
            margin: 0 auto;
            width: 100%;
        }

        .tab-pane {
            display: none;
        }

        .tab-pane.active {
            display: block;
            animation: fadeInPane 0.35s ease-out forwards;
        }

        @keyframes fadeInPane {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ==========================================================================
           8. ALERT BANNER SYSTEM (TIERED COCKPIT WARN)
           ========================================================================== */
        .alert-cockpit-banner {
            border-radius: var(--radius-md);
            padding: 14px 22px;
            margin-bottom: 22px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 0.88rem;
            letter-spacing: 1.5px;
            transition: all 0.35s ease;
            backdrop-filter: blur(12px);
            position: relative;
            overflow: hidden;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        .alert-cockpit-banner::before {
            content: '';
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 5px;
            background: currentColor;
        }

        .alert-cockpit-banner.alert-level-0 {
            background: var(--success-bg);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: var(--success);
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.12);
        }

        .alert-cockpit-banner.alert-level-1 {
            background: var(--warning-bg);
            border: 1px solid rgba(245, 158, 11, 0.6);
            color: var(--warning);
            box-shadow: 0 0 35px rgba(245, 158, 11, 0.25);
            animation: pulseWarnBanner 1.6s infinite;
        }

        .alert-cockpit-banner.alert-level-2 {
            background: var(--danger-bg);
            border: 1px solid rgba(244, 63, 94, 0.8);
            color: var(--danger);
            box-shadow: 0 0 45px rgba(244, 63, 94, 0.45);
            animation: flashCritBanner 0.7s infinite;
        }

        @keyframes pulseWarnBanner {
            0%, 100% { opacity: 0.95; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.003); }
        }

        @keyframes flashCritBanner {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.85; transform: scale(1.008); }
        }

        .banner-left-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .banner-status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 10px currentColor;
        }

        .banner-badge-tag {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            padding: 3px 10px;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid currentColor;
            letter-spacing: 1px;
        }

        /* ==========================================================================
           9. LIVE GRID & GLASS CARDS
           ========================================================================== */
        .cockpit-grid {
            display: grid;
            grid-template-columns: 1.85fr 1.15fr;
            gap: 22px;
            align-items: start;
        }

        .cockpit-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 22px;
            backdrop-filter: blur(16px) saturate(140%);
            -webkit-backdrop-filter: blur(16px) saturate(140%);
            box-shadow: var(--shadow-card);
            transition: border-color var(--transition-smooth), box-shadow var(--transition-smooth);
            position: relative;
        }

        .cockpit-card:hover {
            border-color: var(--border-medium);
        }

        .card-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .card-header-title {
            font-family: var(--font-display);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            color: var(--cyan);
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-header-badge {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.04);
            padding: 3px 10px;
            border-radius: 6px;
            border: 1px solid var(--border-subtle);
        }

        /* ==========================================================================
           10. VIDEO STREAM HUD & CAMERA CONTAINER
           ========================================================================== */
        .video-viewport {
            position: relative;
            width: 100%;
            background: #02040a;
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
            aspect-ratio: 16 / 10;
            max-height: 520px;
            min-height: 220px;
            box-shadow: inset 0 0 40px rgba(0,0,0,0.8);
        }

        .video-viewport img, .video-viewport video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        /* Sci-Fi Targeting Corner Brackets */
        .hud-corner {
            position: absolute;
            width: 20px;
            height: 20px;
            z-index: 5;
            pointer-events: none;
            transition: all var(--transition-smooth);
        }

        .hud-corner-tl { top: 8px; left: 8px; border-top: 2px solid var(--cyan); border-left: 2px solid var(--cyan); }
        .hud-corner-tr { top: 8px; right: 8px; border-top: 2px solid var(--cyan); border-right: 2px solid var(--cyan); }
        .hud-corner-bl { bottom: 8px; left: 8px; border-bottom: 2px solid var(--indigo-light); border-left: 2px solid var(--indigo-light); }
        .hud-corner-br { bottom: 8px; right: 8px; border-bottom: 2px solid var(--indigo-light); border-right: 2px solid var(--indigo-light); }

        /* Dynamic HUD Center Crosshair / Scanline */
        .hud-scanline {
            position: absolute;
            inset: 0;
            z-index: 3;
            pointer-events: none;
            background: linear-gradient(
                to bottom,
                transparent 50%,
                rgba(0, 240, 255, 0.02) 51%,
                transparent 52%
            );
            background-size: 100% 6px;
        }

        /* Status Diagnostic Pills */
        .diagnostic-pills-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 14px;
        }

        .diagnostic-pill {
            background: rgba(8, 14, 28, 0.8);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 9px 12px;
            display: flex;
            flex-direction: column;
            gap: 3px;
            transition: all var(--transition-smooth);
        }

        .diagnostic-pill:hover {
            border-color: var(--border-medium);
            background: rgba(12, 20, 42, 0.9);
        }

        .diag-label {
            font-size: 0.62rem;
            color: var(--text-sub);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }

        .diag-value {
            font-family: var(--font-mono);
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .diag-value.ok { color: var(--success); }
        .diag-value.warn { color: var(--warning); }
        .diag-value.crit { color: var(--danger); }

        /* ==========================================================================
           11. REAL-TIME OSCILLOSCOPE CANVAS (WAVEFORM GRAPH)
           ========================================================================== */
        .oscilloscope-wrap {
            margin-top: 16px;
            background: rgba(4, 8, 20, 0.85);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            position: relative;
        }

        .oscilloscope-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .osc-title {
            font-family: var(--font-display);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .osc-legends {
            display: flex;
            gap: 12px;
            font-family: var(--font-mono);
            font-size: 0.65rem;
        }

        .osc-legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .osc-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
        }

        .osc-canvas {
            width: 100%;
            height: 90px;
            display: block;
            border-radius: var(--radius-sm);
            background: rgba(2, 5, 14, 0.6);
        }

        /* ==========================================================================
           12. TELEMETRY GAUGES & COCKPIT GAZE ORIENTATION COMPASS
           ========================================================================== */
        .biometric-meter {
            margin-bottom: 16px;
        }

        .meter-info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.78rem;
            margin-bottom: 5px;
        }

        .meter-name {
            font-weight: 600;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .meter-val {
            font-family: var(--font-mono);
            font-weight: 700;
            color: var(--text-main);
            font-size: 0.85rem;
        }

        .meter-track {
            background: rgba(20, 32, 60, 0.5);
            border-radius: 6px;
            height: 8px;
            overflow: hidden;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .meter-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.2s ease, background 0.3s ease;
            position: relative;
            background: var(--grad-success);
        }

        .meter-fill.warn-fill { background: var(--grad-warn); }
        .meter-fill.danger-fill { background: var(--grad-danger); }

        .meter-fill::after {
            content: '';
            position: absolute;
            top: 0; right: 0; bottom: 0;
            width: 25px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.25));
            border-radius: 0 5px 5px 0;
        }

        .meter-marker {
            position: absolute;
            top: -1px; bottom: -1px;
            width: 2px;
            background: #fff;
            z-index: 3;
            box-shadow: 0 0 6px rgba(255, 255, 255, 0.9);
        }

        .meter-scale-legend {
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 0.62rem;
            color: var(--text-sub);
            margin-top: 4px;
        }

        .meter-divider {
            border: none;
            border-top: 1px solid var(--border-subtle);
            margin: 16px 0;
        }

        /* 3D Gaze Orientation & Gyroscope Widget */
        .gaze-gyro-widget {
            background: rgba(8, 14, 28, 0.7);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 12px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }

        .gyro-left {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }

        .gyro-title {
            font-size: 0.65rem;
            color: var(--text-sub);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }

        .gyro-heading {
            font-weight: 700;
            font-size: 0.85rem;
            color: var(--success);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .gyro-angles {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            color: var(--text-muted);
        }

        .gyro-visual-dial {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: rgba(14, 24, 48, 0.9);
            border: 1.5px solid var(--border-medium);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            flex-shrink: 0;
        }

        .gyro-arrow {
            width: 3px;
            height: 22px;
            background: var(--cyan);
            border-radius: 2px;
            box-shadow: 0 0 8px var(--cyan);
            transform-origin: center;
            transition: transform 0.2s ease;
        }

        /* Quick Simulator Controls */
        .sim-controls-wrap {
            margin-top: 14px;
            background: rgba(8, 14, 28, 0.6);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 14px;
        }

        .sim-label {
            font-size: 0.65rem;
            color: var(--text-sub);
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .sim-buttons-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
        }

        .btn-sim {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            padding: 6px 8px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-smooth);
            text-align: center;
            white-space: nowrap;
        }

        .btn-sim:hover {
            border-color: var(--indigo-light);
            color: #fff;
            background: rgba(99, 102, 241, 0.15);
        }

        /* Model Selector Dropdown */
        .custom-select-wrap {
            margin-top: 14px;
        }

        .select-label {
            font-size: 0.68rem;
            color: var(--text-sub);
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 6px;
            display: block;
        }

        .model-select-box {
            background: rgba(8, 14, 30, 0.9);
            border: 1px solid var(--border-medium);
            color: var(--text-main);
            padding: 10px 14px;
            border-radius: var(--radius-sm);
            font-family: var(--font-body);
            font-size: 0.82rem;
            font-weight: 600;
            outline: none;
            cursor: pointer;
            width: 100%;
            transition: all var(--transition-smooth);
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
        }

        .model-select-box:focus {
            border-color: var(--cyan);
            box-shadow: 0 0 16px var(--cyan-dim);
        }

        .model-select-box option {
            background: #080f22;
            color: #fff;
            padding: 8px;
        }

        /* ==========================================================================
           13. RESULTS TAB: INTRO, LEADERBOARD, & CATEGORIZED GALLERY
           ========================================================================== */
        .results-hero-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 24px 28px;
            margin-bottom: 24px;
            backdrop-filter: blur(16px);
            position: relative;
            overflow: hidden;
        }

        .results-hero-card::before {
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 250px; height: 250px;
            background: radial-gradient(circle, var(--cyan-dim), transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }

        .results-hero-title {
            font-family: var(--font-display);
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
            background: var(--grad-brand-h);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .results-hero-desc {
            color: var(--text-muted);
            font-size: 0.88rem;
            line-height: 1.65;
            max-width: 1200px;
        }

        /* Leaderboard Controls (Search & Sort) */
        .leaderboard-container {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            backdrop-filter: blur(16px);
            overflow: hidden;
            margin-bottom: 32px;
            box-shadow: var(--shadow-card);
        }

        .leaderboard-toolbar {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .lb-title-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .lb-title {
            font-family: var(--font-display);
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            color: var(--cyan);
        }

        .lb-search-box {
            background: rgba(6, 10, 24, 0.8);
            border: 1px solid var(--border-subtle);
            color: var(--text-main);
            padding: 7px 14px;
            border-radius: var(--radius-sm);
            font-family: var(--font-body);
            font-size: 0.78rem;
            outline: none;
            min-width: 220px;
            transition: all var(--transition-smooth);
        }

        .lb-search-box:focus {
            border-color: var(--cyan);
            box-shadow: 0 0 12px var(--cyan-dim);
        }

        .table-scroll-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        table.leaderboard-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }

        table.leaderboard-table thead th {
            background: rgba(6, 11, 26, 0.75);
            padding: 12px 16px;
            text-align: left;
            font-weight: 700;
            color: var(--text-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            border-bottom: 1px solid var(--border-subtle);
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
            transition: color var(--transition-smooth);
        }

        table.leaderboard-table thead th:hover {
            color: var(--cyan);
        }

        table.leaderboard-table tbody td {
            padding: 12px 16px;
            border-bottom: 1px solid rgba(70, 100, 165, 0.12);
            font-family: var(--font-mono);
            font-weight: 500;
            color: var(--text-main);
            white-space: nowrap;
        }

        table.leaderboard-table tbody tr {
            transition: background var(--transition-smooth);
        }

        table.leaderboard-table tbody tr:hover {
            background: rgba(0, 240, 255, 0.05);
        }

        .rank-medal {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 800;
            font-family: var(--font-mono);
        }

        .rank-gold { background: linear-gradient(135deg, #ffd700, #f59e0b); color: #000; box-shadow: 0 0 10px rgba(255, 215, 0, 0.35); }
        .rank-silver { background: linear-gradient(135deg, #e2e8f0, #94a3b8); color: #000; }
        .rank-bronze { background: linear-gradient(135deg, #f97316, #b45309); color: #fff; }
        .rank-standard { background: rgba(255, 255, 255, 0.06); color: var(--text-sub); }

        .acc-highlight { color: var(--success); font-weight: 700; }

        /* Gallery Category Filter Tabs */
        .gallery-filter-nav {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            padding-bottom: 6px;
        }

        .filter-chip {
            background: rgba(14, 24, 48, 0.6);
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            padding: 7px 16px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-smooth);
            white-space: nowrap;
            user-select: none;
        }

        .filter-chip:hover {
            color: #fff;
            border-color: var(--border-medium);
        }

        .filter-chip.active {
            background: var(--grad-brand);
            color: #030611;
            font-weight: 700;
            border-color: transparent;
            box-shadow: 0 0 16px var(--cyan-glow);
        }

        /* Gallery Sections & Cards */
        .unit-section-wrap {
            margin-bottom: 28px;
        }

        .unit-section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 14px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-subtle);
            cursor: pointer;
            user-select: none;
        }

        .unit-section-title {
            font-family: var(--font-display);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 1.2px;
            color: var(--text-muted);
            text-transform: uppercase;
            flex: 1;
        }

        .unit-toggle-arrow {
            font-size: 0.75rem;
            color: var(--text-sub);
            transition: transform var(--transition-smooth);
        }

        .unit-toggle-arrow.collapsed { transform: rotate(-90deg); }

        .gallery-cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
            gap: 18px;
        }

        .gallery-cards-grid.collapsed { display: none; }

        .visual-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            overflow: hidden;
            transition: all var(--transition-smooth);
            cursor: pointer;
            box-shadow: var(--shadow-subtle);
            position: relative;
        }

        .visual-card:hover {
            transform: translateY(-4px);
            border-color: var(--cyan);
            box-shadow: 0 14px 36px rgba(0, 240, 255, 0.12);
        }

        .visual-card-img-wrap {
            position: relative;
            width: 100%;
            height: 190px;
            background: #040816;
            overflow: hidden;
            border-bottom: 1px solid var(--border-subtle);
        }

        .visual-card-img-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform 0.4s ease;
        }

        .visual-card:hover .visual-card-img-wrap img {
            transform: scale(1.04);
        }

        .card-zoom-badge {
            position: absolute;
            top: 10px; right: 10px;
            background: rgba(6, 10, 24, 0.85);
            border: 1px solid var(--border-medium);
            color: var(--cyan);
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.68rem;
            font-weight: 700;
            opacity: 0;
            transform: translateY(-4px);
            transition: all var(--transition-smooth);
        }

        .visual-card:hover .card-zoom-badge {
            opacity: 1;
            transform: translateY(0);
        }

        .visual-card-body {
            padding: 14px 16px;
        }

        .visual-card-title {
            font-weight: 700;
            font-size: 0.88rem;
            color: var(--text-main);
            margin-bottom: 4px;
        }

        .visual-card-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        /* ==========================================================================
           14. LIGHTBOX MODAL WITH ZOOM & NAVIGATION
           ========================================================================== */
        .modal-lightbox-overlay {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 9999;
            background: rgba(2, 4, 10, 0.94);
            backdrop-filter: blur(14px);
            justify-content: center;
            align-items: center;
            padding: 20px;
            animation: modalFadeIn 0.25s ease-out;
        }

        @keyframes modalFadeIn { from { opacity: 0; } to { opacity: 1; } }

        .modal-inner-container {
            position: relative;
            max-width: 92vw;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .modal-main-img {
            max-width: 90vw;
            max-height: 78vh;
            border-radius: var(--radius-md);
            box-shadow: 0 0 60px rgba(0, 240, 255, 0.2);
            object-fit: contain;
            border: 1px solid var(--border-medium);
            animation: modalZoomIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes modalZoomIn {
            from { transform: scale(0.9); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        .modal-nav-btn {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: rgba(10, 16, 36, 0.85);
            border: 1px solid var(--border-medium);
            color: #fff;
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all var(--transition-smooth);
            z-index: 10;
        }

        .modal-nav-btn:hover {
            border-color: var(--cyan);
            color: var(--cyan);
            box-shadow: 0 0 20px var(--cyan-dim);
            transform: translateY(-50%) scale(1.08);
        }

        .modal-btn-prev { left: -60px; }
        .modal-btn-next { right: -60px; }

        .modal-close-btn {
            position: absolute;
            top: -45px;
            right: 0;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: rgba(10, 16, 36, 0.85);
            border: 1px solid var(--border-medium);
            color: var(--text-muted);
            font-size: 1.4rem;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all var(--transition-smooth);
        }

        .modal-close-btn:hover {
            border-color: var(--danger);
            color: var(--danger);
        }

        .modal-bottom-bar {
            margin-top: 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            gap: 16px;
            background: rgba(10, 16, 36, 0.88);
            border: 1px solid var(--border-subtle);
            padding: 8px 18px;
            border-radius: var(--radius-sm);
        }

        .modal-caption-text {
            font-family: var(--font-body);
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .modal-action-links {
            display: flex;
            gap: 10px;
        }

        .modal-action-btn {
            font-size: 0.72rem;
            color: var(--cyan);
            text-decoration: none;
            background: rgba(0, 240, 255, 0.1);
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid rgba(0, 240, 255, 0.3);
            font-weight: 600;
            transition: all var(--transition-smooth);
        }

        .modal-action-btn:hover {
            background: var(--cyan);
            color: #030611;
        }

        /* ==========================================================================
           15. TOAST NOTIFICATION CONTAINER
           ========================================================================== */
        .toast-shelf {
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: none;
        }

        .toast-item {
            background: rgba(8, 14, 30, 0.95);
            border: 1px solid var(--border-medium);
            color: var(--text-main);
            padding: 12px 18px;
            border-radius: var(--radius-md);
            font-size: 0.8rem;
            font-weight: 600;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            display: flex;
            align-items: center;
            gap: 10px;
            backdrop-filter: blur(16px);
            animation: toastIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            pointer-events: auto;
        }

        @keyframes toastIn {
            from { transform: translateX(50px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        .toast-item.toast-success { border-color: var(--success); color: var(--success); }
        .toast-item.toast-warn { border-color: var(--warning); color: var(--warning); }
        .toast-item.toast-info { border-color: var(--cyan); color: var(--cyan); }

        /* ==========================================================================
           16. FOOTER
           ========================================================================== */
        footer {
            background: rgba(6, 10, 24, 0.85);
            border-top: 1px solid var(--border-subtle);
            padding: 16px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.72rem;
            color: var(--text-sub);
            backdrop-filter: blur(14px);
            padding-bottom: max(16px, env(safe-area-inset-bottom));
        }

        footer a { color: var(--cyan); text-decoration: none; font-weight: 600; }
        footer a:hover { text-decoration: underline; }

        /* ==========================================================================
           17. ADVANCED ULTRA-RESPONSIVE MEDIA QUERIES (MOBILE / TABLET)
           ========================================================================== */
        @media (max-width: 1280px) {
            .stats-ribbon { grid-template-columns: repeat(3, 1fr); }
            .stat-card:nth-child(4), .stat-card:nth-child(5) { grid-column: span 1; }
        }

        @media (max-width: 1120px) {
            .cockpit-grid { grid-template-columns: 1fr; }
        }

        @media (max-width: 768px) {
            header {
                padding: 10px 14px;
                flex-direction: column;
                gap: 10px;
                align-items: stretch;
            }
            .brand-container {
                width: 100%;
                justify-content: space-between;
            }
            .header-actions {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 6px;
                width: 100%;
            }
            .btn-cockpit {
                padding: 8px 6px;
                font-size: 0.72rem;
                justify-content: center;
                gap: 4px;
                border-radius: 7px;
                min-height: 38px;
            }
            .badge-live-desktop { display: none; }

            /* Stats Ribbon 2-Column Grid on Mobile */
            .stats-ribbon-wrap { padding: 0; }
            .stats-ribbon { grid-template-columns: repeat(2, 1fr); gap: 1px; }
            .stat-card:nth-child(5) { grid-column: span 2; justify-content: center; }
            .stat-card { padding: 9px 12px; gap: 8px; }
            .stat-icon-box { width: 30px; height: 30px; font-size: 0.95rem; }
            .stat-label { font-size: 0.58rem; }
            .stat-value { font-size: 0.76rem; }

            .tab-nav-bar { padding: 6px 12px; }
            .tab-pill { flex: 1; justify-content: center; padding: 8px 10px; font-size: 0.78rem; }

            main.main-content { padding: 12px 10px; }
            .cockpit-card { padding: 14px 12px; border-radius: var(--radius-md); }
            .card-header-row { margin-bottom: 12px; padding-bottom: 8px; }
            .card-header-title { font-size: 0.76rem; }

            .video-viewport { max-height: 380px; }
            .diagnostic-pills-grid { grid-template-columns: repeat(2, 1fr); gap: 6px; }

            .sim-buttons-grid { grid-template-columns: repeat(2, 1fr); }

            .gallery-cards-grid { grid-template-columns: 1fr; }
            .visual-card-img-wrap { height: 180px; }

            .modal-btn-prev { left: 8px; }
            .modal-btn-next { right: 8px; }
            .modal-nav-btn { width: 40px; height: 40px; font-size: 1rem; }

            footer {
                flex-direction: column;
                gap: 6px;
                text-align: center;
                padding: 12px 14px;
            }
        }

        @media (max-width: 440px) {
            .header-actions { grid-template-columns: repeat(2, 1fr); }
            .brand-subtitle { display: none; }
            .brand-title { font-size: 0.95rem; }
            .alert-cockpit-banner { font-size: 0.74rem; padding: 10px 14px; }
        }
    </style>
</head>
<body>

    <!-- Ambient Visual Glow Orbs -->
    <div class="ambient-background">
        <div class="ambient-grid"></div>
        <div class="ambient-orb orb-1"></div>
        <div class="ambient-orb orb-2"></div>
        <div class="ambient-orb orb-3"></div>
    </div>

    <div class="app-shell">

        <!-- ===== COCKPIT HEADER ===== -->
        <header>
            <div class="brand-container">
                <div class="brand-icon-wrap" title="Driver Safety AI Cockpit">
                    <svg class="brand-logo-svg" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="cyber-grad-primary" x1="4" y1="4" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#00f0ff"/>
                                <stop offset="0.5" stop-color="#38bdf8"/>
                                <stop offset="1" stop-color="#6366f1"/>
                            </linearGradient>
                            <linearGradient id="cyber-grad-accent" x1="12" y1="14" x2="32" y2="30" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#00f0ff"/>
                                <stop offset="1" stop-color="#a855f7"/>
                            </linearGradient>
                        </defs>
                        <!-- Outer Radar Reticle Ring -->
                        <circle cx="22" cy="22" r="18" stroke="url(#cyber-grad-primary)" stroke-width="1.2" stroke-dasharray="3 2.5" opacity="0.5"/>
                        
                        <!-- Biometric Vision Eye & Cockpit Shield -->
                        <path d="M7 22C11 14 16 10 22 10C28 10 33 14 37 22C33 30 28 34 22 34C16 34 11 30 7 22Z" 
                              stroke="url(#cyber-grad-primary)" stroke-width="2" stroke-linejoin="round" fill="rgba(6, 14, 34, 0.7)"/>
                        
                        <!-- High-Tech Vehicle Hood / Horizon Vectors -->
                        <path d="M14 28L18 22H26L30 28" stroke="url(#cyber-grad-accent)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                        
                        <!-- Twin Laser Headlights -->
                        <circle cx="17.5" cy="24" r="1.3" fill="#00f0ff"/>
                        <circle cx="26.5" cy="24" r="1.3" fill="#00f0ff"/>
                        
                        <!-- Core Neural Pupil & Optical Sensor -->
                        <circle cx="22" cy="18" r="4.2" stroke="#00f0ff" stroke-width="1.5" fill="#030712"/>
                        <circle cx="22" cy="18" r="2.2" fill="#00f0ff"/>
                        
                        <!-- Targeting Crosshairs -->
                        <line x1="22" y1="5" x2="22" y2="8" stroke="#00f0ff" stroke-width="1.8" stroke-linecap="round"/>
                        <line x1="22" y1="36" x2="22" y2="39" stroke="#6366f1" stroke-width="1.8" stroke-linecap="round"/>
                        <line x1="4" y1="22" x2="6.5" y2="22" stroke="#00f0ff" stroke-width="1.8" stroke-linecap="round"/>
                        <line x1="37.5" y1="22" x2="40" y2="22" stroke="#00f0ff" stroke-width="1.8" stroke-linecap="round"/>
                    </svg>
                </div>
                <div class="brand-title-wrap">
                    <span class="brand-title">DRIVER SAFETY AI</span>
                    <span class="brand-subtitle">Real-Time Edge Intelligence &bull; <span>INT8 ONNX Engine</span></span>
                </div>
                <div class="badge-live-pulse" style="display:none;" id="mobile-live-badge">LIVE</div>
            </div>

            <div class="header-actions">
                <button id="btn-webcam" class="btn-cockpit" onclick="toggleBrowserWebcam()">
                    <span id="cam-icon">&#128247;</span>
                    <span id="cam-label">My Webcam</span>
                </button>
                <button id="btn-mesh" class="btn-cockpit" onclick="toggleMeshOverlay()">
                    <span id="mesh-icon">&#127915;</span>
                    <span id="mesh-label">HUD Mesh: OFF</span>
                </button>
                <button id="btn-sound" class="btn-cockpit" onclick="toggleAudio()">
                    <span id="sound-icon">&#128266;</span>
                    <span id="sound-label">Audio: ON</span>
                    <div class="sound-wave" id="sound-visual-bars">
                        <span></span><span></span><span></span><span></span>
                    </div>
                </button>
                <button class="btn-cockpit btn-primary" onclick="triggerCalibration()">
                    <span>&#127919;</span>
                    <span>Calibrate</span>
                </button>
                <div class="badge-live-pulse badge-live-desktop">LIVE &bull; 60 FPS</div>
            </div>
        </header>

        <!-- ===== COCKPIT TELEMETRY STRIP ===== -->
        <div class="stats-ribbon-wrap">
            <div class="stats-ribbon">
                <div class="stat-card">
                    <div class="stat-icon-box">&#9889;</div>
                    <div class="stat-content">
                        <span class="stat-label">Active Inference Engine</span>
                        <span id="stat-model" class="stat-value">Stacking Ensemble <span class="stat-badge-sub">INT8</span></span>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon-box">&#128200;</div>
                    <div class="stat-content">
                        <span class="stat-label">Vision Throughput</span>
                        <span id="stat-fps" class="stat-value">-- FPS</span>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon-box">&#9201;</div>
                    <div class="stat-content">
                        <span class="stat-label">Inference Latency</span>
                        <span id="stat-latency" class="stat-value">-- ms</span>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon-box">&#128338;</div>
                    <div class="stat-content">
                        <span class="stat-label">Mission Uptime</span>
                        <span id="stat-uptime" class="stat-value">00:00:00</span>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon-box">&#129504;</div>
                    <div class="stat-content">
                        <span class="stat-label">Temporal Smoothing</span>
                        <span class="stat-value" style="color:var(--success);">HMM Viterbi Active</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- ===== TAB SELECTION BAR ===== -->
        <div class="tab-nav-bar">
            <div class="tab-nav-inner">
                <button class="tab-pill active" onclick="switchTab('live-tab', this)">
                    <span>&#127909; Live Cockpit Telemetry</span>
                </button>
                <button class="tab-pill" onclick="switchTab('results-tab', this)">
                    <span>&#128202; ML Benchmark &amp; Unit Artifacts</span>
                    <span class="pill-badge">18 Plots</span>
                </button>
            </div>
        </div>

        <!-- ===== MAIN CONTENT ===== -->
        <main class="main-content">

            <!-- TAB 1: LIVE DETECTION COCKPIT -->
            <div id="live-tab" class="tab-pane active">

                <!-- Tiered Alert Banner -->
                <div id="alert-banner" class="alert-cockpit-banner alert-level-0">
                    <div class="banner-left-info">
                        <div class="banner-status-dot"></div>
                        <span id="alert-text">STATUS: DRIVER ALERT &amp; ATTENTIVE</span>
                    </div>
                    <span id="alert-badge" class="banner-badge-tag">LEVEL 0 &bull; NORMAL</span>
                </div>

                <div class="cockpit-grid">
                    
                    <!-- Left: Video Stream & Live Waveform Oscilloscope -->
                    <div class="cockpit-card">
                        <div class="card-header-row">
                            <span class="card-header-title">&#128065; Real-Time Driver Vision HUD</span>
                            <span id="cam-badge" class="card-header-badge">HD Vision Stream &bull; 640x480</span>
                        </div>

                        <!-- Video Viewport -->
                        <div class="video-viewport" id="video-viewport">
                            <div class="hud-corner hud-corner-tl"></div>
                            <div class="hud-corner hud-corner-tr"></div>
                            <div class="hud-corner hud-corner-bl"></div>
                            <div class="hud-corner hud-corner-br"></div>
                            <div class="hud-scanline"></div>
                            <img src="/video_feed" alt="Real-Time Driver Stream" id="video-stream">
                            <video id="browser-video" autoplay playsinline muted style="display:none; width:100%; height:100%; object-fit:cover;"></video>
                            <canvas id="hud-overlay-canvas" style="display:none; position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:4;"></canvas>
                        </div>

                        <!-- 4 Diagnostic Status Pills -->
                        <div class="diagnostic-pills-grid">
                            <div class="diagnostic-pill">
                                <span class="diag-label">Dynamic Baseline</span>
                                <span id="pill-baseline" class="diag-value ok">0.320 (Ready)</span>
                            </div>
                            <div class="diagnostic-pill">
                                <span class="diag-label">Speech Disambiguation</span>
                                <span id="pill-speech" class="diag-value ok">Silent</span>
                            </div>
                            <div class="diagnostic-pill">
                                <span class="diag-label">Cabin Illumination</span>
                                <span id="pill-light" class="diag-value ok">Optimal</span>
                            </div>
                            <div class="diagnostic-pill">
                                <span class="diag-label">Eyewear Analysis</span>
                                <span id="pill-eyewear" class="diag-value ok">Normal</span>
                            </div>
                        </div>

                        <!-- Real-Time Rolling Biometric Oscilloscope -->
                        <div class="oscilloscope-wrap">
                            <div class="oscilloscope-header">
                                <span class="osc-title">&#128201; Real-Time Biometric Oscilloscope (Rolling 60s)</span>
                                <div class="osc-legends">
                                    <div class="osc-legend-item">
                                        <span class="osc-dot" style="background:var(--cyan);"></span>
                                        <span>EAR</span>
                                    </div>
                                    <div class="osc-legend-item">
                                        <span class="osc-dot" style="background:var(--warning);"></span>
                                        <span>MAR</span>
                                    </div>
                                    <div class="osc-legend-item">
                                        <span class="osc-dot" style="background:var(--danger);"></span>
                                        <span>Fatigue</span>
                                    </div>
                                </div>
                            </div>
                            <canvas id="oscCanvas" class="osc-canvas" width="600" height="90"></canvas>
                        </div>
                    </div>

                    <!-- Right: Biometric Instruments & Pose Gyroscope -->
                    <div class="cockpit-card">
                        <div class="card-header-row">
                            <span class="card-header-title">&#9881; Biometric Sensor Fusion</span>
                            <span class="card-header-badge">Multi-Modal</span>
                        </div>

                        <!-- Eye Aspect Ratio (EAR) Gauge -->
                        <div class="biometric-meter">
                            <div class="meter-info-row">
                                <span class="meter-name">&#128065; Eye Aspect Ratio (EAR)</span>
                                <span id="val-ear" class="meter-val">0.32</span>
                            </div>
                            <div class="meter-track">
                                <div id="bar-ear" class="meter-fill" style="width:70%;"></div>
                                <div class="meter-marker" style="left:51%;" title="Critical Thresh: 0.23"></div>
                            </div>
                            <div class="meter-scale-legend">
                                <span>0.00 (Closed)</span>
                                <span style="color:var(--text-accent);">0.23 Alert Thresh</span>
                                <span>0.45 (Open)</span>
                            </div>
                        </div>

                        <!-- Mouth Aspect Ratio (MAR) Gauge -->
                        <div class="biometric-meter">
                            <div class="meter-info-row">
                                <span class="meter-name">&#128564; Mouth Aspect Ratio (MAR)</span>
                                <span id="val-mar" class="meter-val">0.22</span>
                            </div>
                            <div class="meter-track">
                                <div id="bar-mar" class="meter-fill" style="width:25%;"></div>
                                <div class="meter-marker" style="left:65%;" title="Yawn Thresh: 0.55"></div>
                            </div>
                            <div class="meter-scale-legend">
                                <span>0.00 (Closed)</span>
                                <span style="color:var(--warning);">0.55 Yawn Thresh</span>
                                <span>0.85 (Open)</span>
                            </div>
                        </div>

                        <!-- PERCLOS Gauge -->
                        <div class="biometric-meter">
                            <div class="meter-info-row">
                                <span class="meter-name">&#9203; PERCLOS (% Eye Closure)</span>
                                <span id="val-perclos" class="meter-val">0.0%</span>
                            </div>
                            <div class="meter-track">
                                <div id="bar-perclos" class="meter-fill" style="width:0%;"></div>
                                <div class="meter-marker" style="left:20%;" title="Fatigue Thresh: 20%"></div>
                            </div>
                            <div class="meter-scale-legend">
                                <span>0% (Alert)</span>
                                <span style="color:var(--text-accent);">20% Thresh</span>
                                <span>100% (Sleep)</span>
                            </div>
                        </div>

                        <!-- Continuous Fatigue Index Gauge -->
                        <div class="biometric-meter">
                            <div class="meter-info-row">
                                <span class="meter-name">&#128293; Continuous Fatigue Score</span>
                                <span id="val-fatigue" class="meter-val">0.0%</span>
                            </div>
                            <div class="meter-track">
                                <div id="bar-fatigue" class="meter-fill" style="width:0%;"></div>
                                <div class="meter-marker" style="left:70%;" title="Critical Thresh: 70%"></div>
                            </div>
                            <div class="meter-scale-legend">
                                <span>0% (Vigilant)</span>
                                <span style="color:var(--warning);">45% Caution</span>
                                <span style="color:var(--danger);">70% Critical</span>
                            </div>
                        </div>

                        <hr class="meter-divider">

                        <!-- 3D Gaze Orientation & Gyroscope Widget -->
                        <div class="gaze-gyro-widget">
                            <div class="gyro-left">
                                <span class="gyro-title">Head Pose &amp; Spatial Orientation</span>
                                <span id="val-pose-direction" class="gyro-heading">&#10132; Facing Ahead (Attentive)</span>
                                <span id="val-pose" class="gyro-angles">Pitch: 0&deg; | Yaw: 0&deg; | Roll: 0&deg;</span>
                            </div>
                            <div class="gyro-visual-dial">
                                <div id="gyro-pointer" class="gyro-arrow"></div>
                            </div>
                        </div>

                        <hr class="meter-divider">

                        <!-- Model Selection Dropdown -->
                        <div class="custom-select-wrap">
                            <label for="model-selector" class="select-label">Active Inference Architecture</label>
                            <select id="model-selector" class="model-select-box" onchange="changeModel(this.value)">
                                <option value="ensemble">&#129504; Stacking Ensemble (RF + SVM + Bayes) [INT8]</option>
                                <option value="rf">&#127795; Random Forest Classifier (Bagging) [INT8]</option>
                                <option value="bayes">&#9879; Bayesian Logistic Regression (Posterior) [INT8]</option>
                            </select>
                        </div>

                        <!-- Quick Simulator Action Bar -->
                        <div class="sim-controls-wrap">
                            <span class="sim-label">&#128302; Simulation Quick Triggers (Demo Testing)</span>
                            <div class="sim-buttons-grid">
                                <button class="btn-sim" onclick="simulateState('alert')">&#9989; Alert</button>
                                <button class="btn-sim" onclick="simulateState('drowsy')">&#128564; Yawning</button>
                                <button class="btn-sim" onclick="simulateState('sleep')">&#128680; Microsleep</button>
                                <button class="btn-sim" onclick="simulateState('distracted')">&#128260; Distracted</button>
                            </div>
                        </div>

                    </div>

                </div>

            </div>

            <!-- TAB 2: RESULTS & BENCHMARK EVALUATION -->
            <div id="results-tab" class="tab-pane">

                <!-- Hero Header -->
                <div class="results-hero-card">
                    <h2 class="results-hero-title">Academic &amp; Industrial Benchmark Evaluation Suite</h2>
                    <p class="results-hero-desc">
                        Comprehensive diagnostic artifacts spanning Machine Learning Units 1–5: Exploratory Data Analysis, Linear Models &amp; Bayesian Posteriors, Dimensionality Reduction &amp; Clustering, Temporal Markov Chains, and Ensemble Classifiers. Evaluated on 801 stratified real-world test samples with sub-millisecond INT8 hardware acceleration.
                    </p>
                </div>

                <!-- Leaderboard Table Container -->
                <div class="leaderboard-container">
                    <div class="leaderboard-toolbar">
                        <div class="lb-title-group">
                            <span style="font-size:1.2rem;">&#127942;</span>
                            <span class="lb-title">Unified Multi-Model Benchmark Leaderboard</span>
                        </div>
                        <input type="text" id="lb-search" class="lb-search-box" placeholder="Filter models..." oninput="filterLeaderboard(this.value)">
                    </div>
                    <div class="table-scroll-container">
                        <table class="leaderboard-table" id="leaderboard-table">
                            <thead>
                                <tr>
                                    <th onclick="sortTable(0)">#</th>
                                    <th onclick="sortTable(1)">Model Architecture</th>
                                    <th onclick="sortTable(2)">Accuracy</th>
                                    <th onclick="sortTable(3)">Macro F1</th>
                                    <th onclick="sortTable(4)">Precision</th>
                                    <th onclick="sortTable(5)">Recall</th>
                                    <th onclick="sortTable(6)">ROC AUC</th>
                                    <th onclick="sortTable(7)">Latency (ms)</th>
                                    <th onclick="sortTable(8)">Throughput (FPS)</th>
                                    <th onclick="sortTable(9)">Model Size</th>
                                </tr>
                            </thead>
                            <tbody id="leaderboard-body">
                                <tr><td><span class="rank-medal rank-gold">1</span></td><td style="font-weight:700;">Bayesian Logistic Regression</td><td class="acc-highlight">100.00%</td><td class="acc-highlight">100.00%</td><td>100.00%</td><td>100.00%</td><td>1.0000</td><td>0.0008 ms</td><td>1,290,570</td><td>1.9 KB</td></tr>
                                <tr><td><span class="rank-medal rank-gold">1</span></td><td style="font-weight:700;">Support Vector Machine (Linear)</td><td class="acc-highlight">100.00%</td><td class="acc-highlight">100.00%</td><td>100.00%</td><td>100.00%</td><td>1.0000</td><td>0.0081 ms</td><td>123,455</td><td>13.0 KB</td></tr>
                                <tr><td><span class="rank-medal rank-gold">1</span></td><td style="font-weight:700;">Stacking Ensemble (RF+SVM+Bayes)</td><td class="acc-highlight">100.00%</td><td class="acc-highlight">100.00%</td><td>100.00%</td><td>100.00%</td><td>1.0000</td><td>0.1642 ms</td><td>6,091</td><td>330.4 KB</td></tr>
                                <tr><td><span class="rank-medal rank-gold">1</span></td><td style="font-weight:700;">Random Forest (100 Estimators)</td><td class="acc-highlight">100.00%</td><td class="acc-highlight">100.00%</td><td>100.00%</td><td>100.00%</td><td>1.0000</td><td>0.0838 ms</td><td>11,938</td><td>480.8 KB</td></tr>
                                <tr><td><span class="rank-medal rank-silver">5</span></td><td style="font-weight:700;">Support Vector Machine (RBF Kernel)</td><td>99.88%</td><td>99.85%</td><td>99.91%</td><td>99.79%</td><td>1.0000</td><td>0.0414 ms</td><td>24,138</td><td>28.2 KB</td></tr>
                                <tr><td><span class="rank-medal rank-bronze">6</span></td><td style="font-weight:700;">Decision Tree (Cost-Complexity Pruned)</td><td>99.13%</td><td>99.20%</td><td>99.21%</td><td>99.20%</td><td>0.9999</td><td>0.0006 ms</td><td>1,735,645</td><td>4.6 KB</td></tr>
                                <tr><td><span class="rank-medal rank-standard">7</span></td><td style="font-weight:700;">AdaBoost (SAMME.R Boosting)</td><td>97.63%</td><td>97.93%</td><td>97.81%</td><td>98.19%</td><td>0.9973</td><td>0.0352 ms</td><td>28,396</td><td>29.1 KB</td></tr>
                                <tr><td><span class="rank-medal rank-standard">8</span></td><td style="font-weight:700;">Hidden Markov Model (Viterbi Filter)</td><td>96.50%</td><td>96.71%</td><td>96.75%</td><td>96.69%</td><td>1.0000</td><td>0.0487 ms</td><td>20,535</td><td>1.5 KB</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Category Filter Navigation Chips -->
                <div class="gallery-filter-nav">
                    <button class="filter-chip active" onclick="filterGalleryCategory('all', this)">All Units (18 Plots)</button>
                    <button class="filter-chip" onclick="filterGalleryCategory('benchmarks', this)">Performance Benchmarks</button>
                    <button class="filter-chip" onclick="filterGalleryCategory('unit1', this)">Unit 1: EDA</button>
                    <button class="filter-chip" onclick="filterGalleryCategory('unit2', this)">Unit 2: Linear &amp; SVM</button>
                    <button class="filter-chip" onclick="filterGalleryCategory('unit3', this)">Unit 3: Clustering &amp; PCA</button>
                    <button class="filter-chip" onclick="filterGalleryCategory('unit4', this)">Unit 4: Temporal HMM</button>
                    <button class="filter-chip" onclick="filterGalleryCategory('unit5', this)">Unit 5: Ensembles</button>
                </div>

                <!-- Gallery Sections -->

                <!-- Section: Performance Benchmarks -->
                <div class="unit-section-wrap" data-cat="benchmarks">
                    <div class="unit-section-header" onclick="toggleUnitSection(this)">
                        <span style="font-size:1.1rem;">&#127942;</span>
                        <span class="unit-section-title">Model Performance &amp; Evaluation Curves</span>
                        <span class="unit-toggle-arrow">&#9660;</span>
                    </div>
                    <div class="gallery-cards-grid">
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/benchmark_comparison.png', 'Unified Multi-Model Benchmark Comparison')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/benchmark_comparison.png" alt="Benchmark Comparison" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Multi-Model Benchmark Comparison</div>
                                <div class="visual-card-desc">Accuracy, F1-Score, ROC-AUC, and latency across all 8 algorithms.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/multi_model_roc_curves.png', 'Multi-Model ROC Curves')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/multi_model_roc_curves.png" alt="ROC Curves" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Multi-Model ROC Curves (OvR)</div>
                                <div class="visual-card-desc">Receiver Operating Characteristic demonstrating exceptional discriminative separation.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/all_models_confusion_matrices.png', 'All Models Confusion Matrices')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/all_models_confusion_matrices.png" alt="Confusion Matrices" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Confusion Matrices (All Models)</div>
                                <div class="visual-card-desc">True vs Predicted class distribution for Alert, Drowsy, and Sleeping states.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/model_calibration_curves.png', 'Probability Calibration Curves')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/model_calibration_curves.png" alt="Calibration Curves" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Probability Calibration Curves</div>
                                <div class="visual-card-desc">Reliability diagram comparing predicted probabilities against empirical frequencies.</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Section: Unit 1 EDA -->
                <div class="unit-section-wrap" data-cat="unit1">
                    <div class="unit-section-header" onclick="toggleUnitSection(this)">
                        <span style="font-size:1.1rem;">&#128202;</span>
                        <span class="unit-section-title">Unit 1 &mdash; Exploratory Data Analysis &amp; Feature Distributions</span>
                        <span class="unit-toggle-arrow">&#9660;</span>
                    </div>
                    <div class="gallery-cards-grid">
                        <div class="visual-card" onclick="openLightbox('/outputs/eda/correlation_heatmap.png', 'Feature Correlation Heatmap')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/eda/correlation_heatmap.png" alt="Correlation Heatmap" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Correlation Heatmap</div>
                                <div class="visual-card-desc">Pearson correlation matrix for all extracted biometric feature channels.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/eda/feature_histograms.png', 'Feature Histograms')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/eda/feature_histograms.png" alt="Feature Histograms" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Feature Distribution Histograms</div>
                                <div class="visual-card-desc">Distribution shape, skewness, and variance across each biometric feature.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/eda/feature_boxplots.png', 'Feature Boxplots')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/eda/feature_boxplots.png" alt="Feature Boxplots" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Feature Boxplots (IQR Analysis)</div>
                                <div class="visual-card-desc">Median, quartiles, and outlier analysis across biometric channels.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/eda/class_distribution.png', 'Class Balance Distribution')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/eda/class_distribution.png" alt="Class Distribution" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Class Balance Distribution</div>
                                <div class="visual-card-desc">Sample count and stratified distribution across Alert, Drowsy, and Sleeping states.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/eda/scatter_plots.png', 'Pairwise Feature Scatter')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/eda/scatter_plots.png" alt="Scatter Plots" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Pairwise Feature Scatter Matrix</div>
                                <div class="visual-card-desc">2D feature relationships colored by driver alertness label.</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Section: Unit 2 Linear & SVM -->
                <div class="unit-section-wrap" data-cat="unit2">
                    <div class="unit-section-header" onclick="toggleUnitSection(this)">
                        <span style="font-size:1.1rem;">&#128640;</span>
                        <span class="unit-section-title">Unit 2 &mdash; Linear Models &amp; Support Vector Machines</span>
                        <span class="unit-toggle-arrow">&#9660;</span>
                    </div>
                    <div class="gallery-cards-grid">
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/svm_decision_boundary_rbf.png', 'RBF SVM Decision Boundary')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/svm_decision_boundary_rbf.png" alt="SVM Boundary" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">RBF SVM Decision Boundary</div>
                                <div class="visual-card-desc">Non-linear kernel separating alert vs fatigued states in PCA space.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/bayesian_posterior_sample.png', 'Bayesian Parameter Posteriors')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/bayesian_posterior_sample.png" alt="Bayesian Posteriors" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Bayesian Parameter Posteriors</div>
                                <div class="visual-card-desc">Laplace-approximated weight distributions quantifying epistemic uncertainty.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/regression_residuals_ridge.png', 'Ridge Fatigue Residuals')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/regression_residuals_ridge.png" alt="Ridge Residuals" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Ridge Fatigue Score Residuals</div>
                                <div class="visual-card-desc">Residual analysis for continuous fatigue index regression.</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Section: Unit 3 Clustering & PCA -->
                <div class="unit-section-wrap" data-cat="unit3">
                    <div class="unit-section-header" onclick="toggleUnitSection(this)">
                        <span style="font-size:1.1rem;">&#127760;</span>
                        <span class="unit-section-title">Unit 3 &mdash; Dimensionality Reduction &amp; Unsupervised Clustering</span>
                        <span class="unit-toggle-arrow">&#9660;</span>
                    </div>
                    <div class="gallery-cards-grid">
                        <div class="visual-card" onclick="openLightbox('/outputs/clustering/pca_scree_plot.png', 'PCA Scree Plot')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/clustering/pca_scree_plot.png" alt="PCA Scree" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">PCA Scree Plot</div>
                                <div class="visual-card-desc">Explained variance ratio per principal component with cumulative trajectory.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/clustering/pca_2d_projection.png', 'PCA 2D Projection')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/clustering/pca_2d_projection.png" alt="PCA 2D" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">PCA 2D Projection</div>
                                <div class="visual-card-desc">Biometric feature samples projected onto first two orthogonal components.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/clustering/kmeans_elbow_silhouette.png', 'K-Means Elbow & Silhouette')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/clustering/kmeans_elbow_silhouette.png" alt="K-Means Elbow" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">K-Means Elbow &amp; Silhouette Analysis</div>
                                <div class="visual-card-desc">Optimal cluster selection via inertia elbow and silhouette scores.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/clustering/kmeans_clusters_2d.png', 'K-Means 2D Clusters')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/clustering/kmeans_clusters_2d.png" alt="K-Means Clusters" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">K-Means Cluster Space</div>
                                <div class="visual-card-desc">Unsupervised cluster assignments with computed cluster centroids.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/clustering/gmm_aic_bic.png', 'GMM AIC/BIC Criteria')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/clustering/gmm_aic_bic.png" alt="GMM AIC/BIC" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">GMM AIC &amp; BIC Complexity</div>
                                <div class="visual-card-desc">Information theoretic criterion minimizing overfitting in Gaussian mixtures.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/clustering/gmm_clusters_ellipses.png', 'GMM Covariance Ellipses')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/clustering/gmm_clusters_ellipses.png" alt="GMM Ellipses" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">GMM Covariance Ellipses</div>
                                <div class="visual-card-desc">Probabilistic mixture components illustrated as 2-sigma confidence ellipses.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/clustering/hierarchical_dendrogram.png', 'Hierarchical Dendrogram')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/clustering/hierarchical_dendrogram.png" alt="Hierarchical Dendrogram" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Hierarchical Dendrogram</div>
                                <div class="visual-card-desc">Ward linkage agglomerative clustering tree hierarchy.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/clustering/hierarchical_clusters_2d.png', 'Hierarchical 2D Clusters')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/clustering/hierarchical_clusters_2d.png" alt="Hierarchical Clusters" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Hierarchical Cluster Projections</div>
                                <div class="visual-card-desc">Agglomerative cluster groupings mapped into reduced dimensional space.</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Section: Unit 4 Temporal HMM -->
                <div class="unit-section-wrap" data-cat="unit4">
                    <div class="unit-section-header" onclick="toggleUnitSection(this)">
                        <span style="font-size:1.1rem;">&#9201;</span>
                        <span class="unit-section-title">Unit 4 &mdash; Hidden Markov Model (Temporal Dynamics)</span>
                        <span class="unit-toggle-arrow">&#9660;</span>
                    </div>
                    <div class="gallery-cards-grid">
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/hmm_transition_matrix.png', 'HMM Transition Matrix')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/hmm_transition_matrix.png" alt="HMM Matrix" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">HMM State Transition Matrix</div>
                                <div class="visual-card-desc">Empirical Markovian transition probabilities between driver alertness states.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/hmm_state_sequence_decoding.png', 'Viterbi Sequence Decoding')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/hmm_state_sequence_decoding.png" alt="HMM Viterbi" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Viterbi State Sequence Smoothing</div>
                                <div class="visual-card-desc">Temporal dynamic programming filtering that prevents single-frame false triggers.</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Section: Unit 5 Ensembles -->
                <div class="unit-section-wrap" data-cat="unit5">
                    <div class="unit-section-header" onclick="toggleUnitSection(this)">
                        <span style="font-size:1.1rem;">&#127795;</span>
                        <span class="unit-section-title">Unit 5 &mdash; Tree-Based &amp; Ensemble Architectures</span>
                        <span class="unit-toggle-arrow">&#9660;</span>
                    </div>
                    <div class="gallery-cards-grid">
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/random_forest_feature_importance.png', 'Random Forest Feature Importance')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/random_forest_feature_importance.png" alt="RF Importance" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Random Forest Gini Importance</div>
                                <div class="visual-card-desc">Mean decrease in impurity ranking for EAR, MAR, and head kinematics.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/random_forest_oob_trees.png', 'Random Forest OOB Error Convergence')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/random_forest_oob_trees.png" alt="RF OOB" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">OOB Error Rate vs Trees</div>
                                <div class="visual-card-desc">Out-of-bag validation error convergence across 100 ensemble trees.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/decision_tree_structure.png', 'Decision Tree Pruned Structure')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/decision_tree_structure.png" alt="Decision Tree Structure" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Pruned Decision Tree Structure</div>
                                <div class="visual-card-desc">Cost-complexity pruned tree visualization with optimal split thresholds.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/decision_tree_feature_importances.png', 'Decision Tree Feature Importances')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/decision_tree_feature_importances.png" alt="DT Importance" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">Decision Tree Split Importances</div>
                                <div class="visual-card-desc">Gini contribution analysis of primary decision rules.</div>
                            </div>
                        </div>
                        <div class="visual-card" onclick="openLightbox('/outputs/evaluation/adaboost_stagewise_error.png', 'AdaBoost Stagewise Error')">
                            <div class="visual-card-img-wrap">
                                <img src="/outputs/evaluation/adaboost_stagewise_error.png" alt="AdaBoost Error" loading="lazy">
                                <span class="card-zoom-badge">&#128269; View</span>
                            </div>
                            <div class="visual-card-body">
                                <div class="visual-card-title">AdaBoost Stagewise Error Reduction</div>
                                <div class="visual-card-desc">Sequential exponential loss reduction across boosting rounds.</div>
                            </div>
                        </div>
                    </div>
                </div>

            </div><!-- /results-tab -->

        </main>

        <!-- ===== LIGHTBOX MODAL ===== -->
        <div id="modalLightbox" class="modal-lightbox-overlay" onclick="closeLightbox(event)">
            <div class="modal-inner-container" onclick="event.stopPropagation()">
                <button class="modal-close-btn" onclick="closeLightbox()">&times;</button>
                <button class="modal-nav-btn modal-btn-prev" onclick="prevLightboxImage()">&#10094;</button>
                <button class="modal-nav-btn modal-btn-next" onclick="nextLightboxImage()">&#10095;</button>
                
                <img id="lightboxImg" class="modal-main-img" src="" alt="Diagnostic High-Res Plot">
                
                <div class="modal-bottom-bar">
                    <span id="lightboxCaption" class="modal-caption-text">Diagnostic Plot</span>
                    <div class="modal-action-links">
                        <a id="lightboxDownload" href="#" download="benchmark_plot.png" class="modal-action-btn">&#128190; Download PNG</a>
                    </div>
                </div>
            </div>
        </div>

        <!-- ===== FLOATING TOAST CONTAINER ===== -->
        <div class="toast-shelf" id="toastShelf"></div>

        <!-- ===== FOOTER ===== -->
        <footer>
            <span><strong>Driver Safety AI &bull; Edge Cockpit</strong> &mdash; Production Real-Time Drowsiness &amp; Alert System</span>
            <span>OpenCV &bull; MediaPipe &bull; scikit-learn &bull; ONNX INT8 | <a href="https://github.com/bankutech/Real-Time-Drowsiness-Detection-and-Alert-System" target="_blank">GitHub Repository</a></span>
        </footer>

    </div><!-- /app-shell -->

    <!-- ==========================================================================
       18. COCKPIT INTERACTIVE SCRIPT & REAL-TIME LOGIC
       ========================================================================== -->
    <script>
        // ===== UPTIME COUNTER =====
        const systemStartTime = Date.now();
        function updateUptimeDisplay() {
            const totalSec = Math.floor((Date.now() - systemStartTime) / 1000);
            const hrs = String(Math.floor(totalSec / 3600)).padStart(2, '0');
            const mins = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
            const secs = String(totalSec % 60).padStart(2, '0');
            const el = document.getElementById('stat-uptime');
            if (el) el.innerText = hrs + ':' + mins + ':' + secs;
        }
        setInterval(updateUptimeDisplay, 1000);

        // ===== TOAST NOTIFICATION HELPER =====
        function showToast(message, type = 'info') {
            const shelf = document.getElementById('toastShelf');
            if (!shelf) return;
            const toast = document.createElement('div');
            toast.className = 'toast-item toast-' + type;
            const icon = type === 'success' ? '&#9989;' : (type === 'warn' ? '&#9888;' : '&#8505;');
            toast.innerHTML = '<span>' + icon + '</span><span>' + message + '</span>';
            shelf.appendChild(toast);
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transform = 'translateY(10px)';
                setTimeout(() => toast.remove(), 350);
            }, 3200);
        }

        // ===== TABS SWITCHER =====
        function switchTab(tabId, btn) {
            document.querySelectorAll('.tab-pill').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            if (btn) btn.classList.add('active');
            const pane = document.getElementById(tabId);
            if (pane) pane.classList.add('active');
        }

        // ===== MESH OVERLAY TOGGLE =====
        let meshOverlayActive = false;
        function toggleMeshOverlay() {
            meshOverlayActive = !meshOverlayActive;
            const btn = document.getElementById('btn-mesh');
            const lbl = document.getElementById('mesh-label');
            if (meshOverlayActive) {
                lbl.innerText = 'HUD Mesh: ON';
                btn.classList.add('active-toggle');
                showToast('Facial Mesh & Reticle HUD Enabled', 'info');
            } else {
                lbl.innerText = 'HUD Mesh: OFF';
                btn.classList.remove('active-toggle');
                showToast('Facial Mesh HUD Disabled', 'info');
            }
        }

        // ===== WEB AUDIO & TONE SYNTHESIS =====
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
            const icon = document.getElementById('sound-icon');
            const lbl = document.getElementById('sound-label');
            const bars = document.getElementById('sound-visual-bars');
            const btn = document.getElementById('btn-sound');

            if (soundEnabled) {
                icon.innerText = String.fromCodePoint(0x1F50A);
                lbl.innerText = 'Audio: ON';
                if (bars) bars.style.display = 'flex';
                btn.classList.remove('active-toggle');
                showToast('Safety audio alert chimes unmuted', 'success');
                playAlertTone(1);
            } else {
                icon.innerText = String.fromCodePoint(0x1F507);
                lbl.innerText = 'Audio: MUTE';
                if (bars) bars.style.display = 'none';
                btn.classList.add('active-toggle');
                showToast('Safety audio muted', 'warn');
            }
        }

        function playAlertTone(level) {
            if (!soundEnabled) return;
            initAudio();
            const now = Date.now();
            if (now - lastSoundTime < 1200) return;
            lastSoundTime = now;
            try {
                if (level === 1) {
                    // Two-tone soft chime
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
                    osc.frequency.exponentialRampToValueAtTime(1174, audioCtx.currentTime + 0.22);
                    gain.gain.setValueAtTime(0.22, audioCtx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.35);
                } else if (level === 2) {
                    // Urgent double pulse siren
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(1250, audioCtx.currentTime);
                    osc.frequency.linearRampToValueAtTime(1600, audioCtx.currentTime + 0.2);
                    gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.45);
                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.45);
                }
            } catch (err) { console.debug('Audio error:', err); }
        }

        // ===== CALIBRATION TRIGGER =====
        function triggerCalibration() {
            initAudio();
            fetch('/api/calibrate').then(r => r.json()).then(() => {
                const banner = document.getElementById('alert-banner');
                banner.className = 'alert-cockpit-banner alert-level-1';
                document.getElementById('alert-text').innerText = 'CALIBRATING BASELINE (LOOK FORWARD NORMALLY)...';
                document.getElementById('alert-badge').innerText = 'CALIBRATING';
                showToast('Driver baseline calibration started — please maintain natural posture', 'info');
            }).catch(e => {
                showToast('Calibration request error', 'warn');
            });
        }

        // ===== MODEL SWITCHER =====
        const modelNames = {
            ensemble: 'Stacking Ensemble (INT8)',
            rf: 'Random Forest (INT8)',
            bayes: 'Bayesian Logistic (INT8)'
        };
        function changeModel(modelVal) {
            fetch('/api/set_model?model=' + encodeURIComponent(modelVal)).then(r => r.json()).then(d => {
                const name = modelNames[modelVal] || modelVal;
                document.getElementById('stat-model').innerHTML = name + ' <span class="stat-badge-sub">INT8</span>';
                showToast('Inference model switched to ' + name, 'success');
            }).catch(e => {
                showToast('Failed to switch model', 'warn');
            });
        }

        // ===== SIMULATOR TRIGGERS (DEMO MODE) =====
        function simulateState(state) {
            initAudio();
            let fakeTelem = {};
            if (state === 'alert') {
                fakeTelem = { ear: 0.34, mar: 0.18, perclos: 0.0, fatigue_score: 0.05, alert_level: 0, status_text: 'DRIVER STATUS: ALERT & ATTENTIVE', head_pose_direction: 'Facing Ahead (Attentive)', pitch: 2, yaw: -1, roll: 0 };
                showToast('Simulated normal alert driving state', 'success');
            } else if (state === 'drowsy') {
                fakeTelem = { ear: 0.28, mar: 0.72, perclos: 0.08, fatigue_score: 0.48, alert_level: 1, status_text: 'CAUTION: FREQUENT YAWNING DETECTED', head_pose_direction: 'Facing Ahead', pitch: -5, yaw: 2, roll: 1 };
                showToast('Simulated yawning event triggered', 'warn');
            } else if (state === 'sleep') {
                fakeTelem = { ear: 0.12, mar: 0.22, perclos: 0.85, fatigue_score: 0.92, alert_level: 2, status_text: 'CRITICAL: MICROSLEEP DETECTED! WAKE UP!', head_pose_direction: 'Head Slumped Forward', pitch: -22, yaw: 0, roll: -3 };
                showToast('CRITICAL microsleep emergency simulated!', 'warn');
            } else if (state === 'distracted') {
                fakeTelem = { ear: 0.31, mar: 0.19, perclos: 0.04, fatigue_score: 0.35, alert_level: 1, status_text: 'WARNING: DRIVER GAZE DISTRACTED', head_pose_direction: 'Looking Left Mirror', pitch: 3, yaw: -38, roll: 2 };
                showToast('Simulated gaze distraction event', 'info');
            }
            renderTelemetryData(fakeTelem);
            if (fakeTelem.alert_level > 0) playAlertTone(fakeTelem.alert_level);
        }

        // ===== REAL-TIME BIOMETRIC OSCILLOSCOPE ENGINE =====
        const oscHistory = { ear: [], mar: [], fatigue: [], maxPoints: 120 };
        function updateOscilloscope(ear, mar, fatigue) {
            const canvas = document.getElementById('oscCanvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;

            oscHistory.ear.push(ear || 0.32);
            oscHistory.mar.push(mar || 0.20);
            oscHistory.fatigue.push(fatigue || 0.0);

            if (oscHistory.ear.length > oscHistory.maxPoints) {
                oscHistory.ear.shift();
                oscHistory.mar.shift();
                oscHistory.fatigue.shift();
            }

            ctx.clearRect(0, 0, w, h);

            // Draw Background Grid Lines
            ctx.strokeStyle = 'rgba(70, 100, 160, 0.15)';
            ctx.lineWidth = 1;
            for (let y = 18; y < h; y += 18) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            }

            // Draw Threshold Guide Lines
            // EAR 0.23 Thresh (Normalized in 0-0.5 range)
            const earThreshY = h - (0.23 / 0.45) * h;
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(0, earThreshY); ctx.lineTo(w, earThreshY); ctx.stroke();

            // Fatigue 0.70 Critical Thresh
            const fatThreshY = h - 0.70 * h;
            ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
            ctx.beginPath(); ctx.moveTo(0, fatThreshY); ctx.lineTo(w, fatThreshY); ctx.stroke();
            ctx.setLineDash([]);

            function drawWave(data, maxVal, strokeColor, fillColor) {
                if (data.length < 2) return;
                const step = w / (oscHistory.maxPoints - 1);
                const startX = (oscHistory.maxPoints - data.length) * step;

                ctx.beginPath();
                for (let i = 0; i < data.length; i++) {
                    const x = startX + i * step;
                    const norm = Math.min(1, Math.max(0, data[i] / maxVal));
                    const y = h - norm * (h - 8) - 4;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }

                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 2;
                ctx.stroke();

                if (fillColor) {
                    ctx.lineTo(w, h);
                    ctx.lineTo(startX, h);
                    ctx.closePath();
                    ctx.fillStyle = fillColor;
                    ctx.fill();
                }
            }

            drawWave(oscHistory.fatigue, 1.0, '#f43f5e', 'rgba(244, 63, 94, 0.08)');
            drawWave(oscHistory.mar, 0.85, '#f59e0b', null);
            drawWave(oscHistory.ear, 0.45, '#00f0ff', 'rgba(0, 240, 255, 0.06)');
        }

        // ===== TELEMETRY DATA RENDERER =====
        function renderTelemetryData(d) {
            if (!d || Object.keys(d).length === 0) return;

            const fps = d.fps !== undefined ? Number(d.fps).toFixed(1) : '--';
            const lat = d.latency_ms !== undefined ? Number(d.latency_ms).toFixed(1) : '--';
            const fpsEl = document.getElementById('stat-fps');
            const latEl = document.getElementById('stat-latency');
            if (fpsEl) fpsEl.innerText = fps + ' FPS';
            if (latEl) latEl.innerText = lat + ' ms';

            // Numeric Metric Readouts
            const earVal = d.ear || 0;
            const marVal = d.mar || 0;
            const pclVal = d.perclos || 0;
            const fatVal = d.fatigue_score || 0;

            document.getElementById('val-ear').innerText = earVal.toFixed(2);
            document.getElementById('val-mar').innerText = marVal.toFixed(2);
            document.getElementById('val-perclos').innerText = (pclVal * 100).toFixed(1) + '%';
            document.getElementById('val-fatigue').innerText = (fatVal * 100).toFixed(1) + '%';

            // Head Pose & Compass Gyro
            const poseDir = d.head_pose_direction || 'Facing Ahead (Attentive)';
            const poseEl = document.getElementById('val-pose-direction');
            poseEl.innerText = poseDir;
            if (poseDir.includes('Ahead') || poseDir.includes('Attentive')) {
                poseEl.style.color = 'var(--success)';
            } else if (poseDir.includes('Distracted') || poseDir.includes('Down') || poseDir.includes('Slumped')) {
                poseEl.style.color = 'var(--danger)';
            } else {
                poseEl.style.color = 'var(--warning)';
            }
            document.getElementById('val-pose').innerText = 'Pitch: ' + (d.pitch||0).toFixed(0) + '\u00B0 | Yaw: ' + (d.yaw||0).toFixed(0) + '\u00B0 | Roll: ' + (d.roll||0).toFixed(0) + '\u00B0';

            // Rotate Gyro Pointer
            const gyroPointer = document.getElementById('gyro-pointer');
            if (gyroPointer) {
                const yawDeg = d.yaw || 0;
                gyroPointer.style.transform = 'rotate(' + yawDeg + 'deg)';
            }

            // Meter Fill Percentages
            const earPct = Math.min(100, Math.max(0, (earVal / 0.45) * 100));
            const marPct = Math.min(100, Math.max(0, (marVal / 0.85) * 100));
            const pclPct = Math.min(100, Math.max(0, pclVal * 100));
            const fatPct = Math.min(100, Math.max(0, fatVal * 100));

            setMeterBar('bar-ear', earPct, earVal < 0.23 ? 'danger-fill' : '');
            setMeterBar('bar-mar', marPct, marVal > 0.55 ? 'danger-fill' : '');
            setMeterBar('bar-perclos', pclPct, pclVal > 0.20 ? 'danger-fill' : '');
            setMeterBar('bar-fatigue', fatPct, fatVal >= 0.70 ? 'danger-fill' : (fatVal >= 0.45 ? 'warn-fill' : ''));

            // Tiered Alert Banner Transition
            const lvl = d.alert_level || 0;
            const banner = document.getElementById('alert-banner');
            banner.className = 'alert-cockpit-banner alert-level-' + lvl;
            document.getElementById('alert-text').innerText = d.status_text || 'STATUS: DRIVER ALERT & ATTENTIVE';
            document.getElementById('alert-badge').innerText = 'LEVEL ' + lvl + (lvl === 2 ? ' • CRITICAL' : (lvl === 1 ? ' • CAUTION' : ' • NORMAL'));
            if (lvl > 0) playAlertTone(lvl);

            // Dynamic Browser Tab Title with Live State
            const titlePrefix = lvl === 2 ? '🚨 [CRITICAL ALERT] ' : (lvl === 1 ? '⚠️ [CAUTION] ' : '🟢 ');
            document.title = titlePrefix + 'Driver Safety AI';

            // Diagnostic Status Pills
            const pBase = document.getElementById('pill-baseline');
            if (d.calibrating) {
                pBase.innerText = 'Calibrating ' + (d.calibration_progress||0) + '%';
                pBase.className = 'diag-value warn';
            } else {
                pBase.innerText = (d.baseline_ear || 0.32).toFixed(3) + ' (Ready)';
                pBase.className = 'diag-value ok';
            }

            const pSp = document.getElementById('pill-speech');
            pSp.innerText = d.is_speaking ? 'Speaking' : 'Silent';
            pSp.className = d.is_speaking ? 'diag-value warn' : 'diag-value ok';

            const pLi = document.getElementById('pill-light');
            pLi.innerText = d.lighting_quality || 'Optimal';
            pLi.className = d.low_light ? 'diag-value warn' : 'diag-value ok';

            const pEy = document.getElementById('pill-eyewear');
            if (d.eyewear_detected) {
                pEy.innerText = (d.eyewear_label || 'Glasses');
                pEy.className = 'diag-value warn';
            } else {
                pEy.innerText = 'Normal (None)';
                pEy.className = 'diag-value ok';
            }

            // Update Real-Time Oscilloscope Waveform
            updateOscilloscope(earVal, marVal, fatVal);
        }

        function setMeterBar(id, pct, cls) {
            const el = document.getElementById(id);
            if (!el) return;
            el.style.width = pct + '%';
            el.className = 'meter-fill' + (cls ? ' ' + cls : '');
        }

        // ===== POLLING LOOP =====
        function pollServerTelemetry() {
            if (!browserCamActive) {
                fetch('/api/telemetry').then(r => r.json()).then(renderTelemetryData).catch(() => {});
            }
        }
        setInterval(pollServerTelemetry, 100);

        // ===== BROWSER WEBCAM STREAMING & CLIENT-SIDE HUD OVERLAY =====
        let browserCamStream = null;
        let browserCamActive = false;
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
                            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
                            audio: false
                        });
                    } catch (e1) {
                        try {
                            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
                        } catch (e2) {
                            stream = await navigator.mediaDevices.getUserMedia({ video: true });
                        }
                    }

                    browserCamStream = stream;
                    vidEl.srcObject = stream;
                    await vidEl.play();

                    vidEl.style.display = 'block';
                    hudCanvas.style.display = 'block';
                    imgEl.style.display = 'none';

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
                    document.getElementById('btn-webcam').classList.add('active-toggle');
                    showToast('Client camera active & streaming to AI engine', 'success');

                    async function streamFrameLoop() {
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
                                            if (dt > 0) telem.fps = Math.min(60.0, 1.0 / dt);
                                            telem.latency_ms = now - tStart;
                                            renderTelemetryData(telem);
                                            drawClientHud(hudCanvas, telem);
                                        }
                                    }
                                } catch (err) {}
                            }
                            await new Promise(r => requestAnimationFrame(r));
                        }
                    }
                    streamFrameLoop();
                } catch (err) {
                    showToast('Webcam access error: ' + err.message, 'warn');
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
                document.getElementById('btn-webcam').classList.remove('active-toggle');
                showToast('Switched back to synthesized server vision feed', 'info');
            }
        }

        function drawClientHud(canvas, d) {
            if (!canvas || !d) return;
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            const lvl = d.alert_level || 0;

            // Perimeter Warning Strobe
            if (lvl === 2) {
                ctx.lineWidth = 6;
                ctx.strokeStyle = '#f43f5e';
                ctx.strokeRect(0, 0, w, h);
            } else if (lvl === 1) {
                ctx.lineWidth = 3;
                ctx.strokeStyle = '#f59e0b';
                ctx.strokeRect(0, 0, w, h);
            }

            // Top HUD Status Bar
            ctx.fillStyle = 'rgba(6, 10, 24, 0.78)';
            ctx.fillRect(0, 0, w, 36);

            const dotColor = lvl === 2 ? '#f43f5e' : (lvl === 1 ? '#f59e0b' : '#10b981');
            ctx.fillStyle = dotColor;
            ctx.beginPath();
            ctx.arc(18, 18, 5, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 12px "Orbitron", sans-serif';
            ctx.fillText(d.status_text || 'DRIVER STATUS: ALERT', 32, 22);

            const poseBadge = (d.head_pose_direction || 'Ahead') + (d.eyewear_detected ? ' [Glasses]' : '');
            ctx.fillStyle = '#00f0ff';
            ctx.font = '11px "JetBrains Mono", monospace';
            const tw = ctx.measureText(poseBadge).width;
            ctx.fillText(poseBadge, w - tw - 16, 22);

            // Optional 3D Landmark Mesh
            if (meshOverlayActive && d.landmarks) {
                const lm = d.landmarks;
                function drawPolygon(pts, color) {
                    if (!pts || pts.length === 0) return;
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(pts[0][0] * w, pts[0][1] * h);
                    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * w, pts[i][1] * h);
                    ctx.closePath();
                    ctx.stroke();
                }

                drawPolygon(lm.left_eye, 'rgba(0, 240, 255, 0.85)');
                drawPolygon(lm.right_eye, 'rgba(0, 240, 255, 0.85)');
                drawPolygon(lm.mouth, 'rgba(245, 158, 11, 0.85)');

                // 3D Nose Vector
                if (lm.nose_tip) {
                    const nx = lm.nose_tip[0] * w;
                    const ny = lm.nose_tip[1] * h;
                    const pitch = (d.pitch || 0) * (Math.PI / 180);
                    const yaw = (d.yaw || 0) * (Math.PI / 180);
                    const len = 34;

                    ctx.strokeStyle = '#f43f5e';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(nx, ny);
                    ctx.lineTo(nx + len * Math.cos(yaw), ny);
                    ctx.stroke();

                    ctx.strokeStyle = '#10b981';
                    ctx.beginPath();
                    ctx.moveTo(nx, ny);
                    ctx.lineTo(nx, ny - len * Math.sin(pitch));
                    ctx.stroke();

                    ctx.fillStyle = '#00f0ff';
                    ctx.beginPath(); ctx.arc(nx, ny, 4, 0, 2 * Math.PI); ctx.fill();
                }

                // Face Reticle
                if (lm.face_box) {
                    const bx1 = lm.face_box[0] * w, by1 = lm.face_box[1] * h;
                    const bx2 = lm.face_box[2] * w, by2 = lm.face_box[3] * h;
                    const cLen = 14;
                    ctx.strokeStyle = 'rgba(0, 240, 255, 0.7)';
                    ctx.lineWidth = 2;
                    ctx.beginPath(); ctx.moveTo(bx1, by1 + cLen); ctx.lineTo(bx1, by1); ctx.lineTo(bx1 + cLen, by1); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(bx2 - cLen, by1); ctx.lineTo(bx2, by1); ctx.lineTo(bx2, by1 + cLen); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(bx1, by2 - cLen); ctx.lineTo(bx1, by2); ctx.lineTo(bx1 + cLen, by2); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(bx2 - cLen, by2); ctx.lineTo(bx2, by2); ctx.lineTo(bx2, by2 - cLen); ctx.stroke();
                }
            }
        }

        // ===== LEADERBOARD SEARCH & SORT =====
        function filterLeaderboard(query) {
            const filter = query.toLowerCase();
            const rows = document.querySelectorAll('#leaderboard-body tr');
            rows.forEach(r => {
                const text = r.innerText.toLowerCase();
                r.style.display = text.includes(filter) ? '' : 'none';
            });
        }

        let sortDirections = {};
        function sortTable(colIndex) {
            const table = document.getElementById('leaderboard-table');
            const tbody = document.getElementById('leaderboard-body');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const isAsc = !sortDirections[colIndex];
            sortDirections[colIndex] = isAsc;

            rows.sort((a, b) => {
                const cellA = a.children[colIndex].innerText.replace(/[%msKB,]/g, '').trim();
                const cellB = b.children[colIndex].innerText.replace(/[%msKB,]/g, '').trim();
                const numA = parseFloat(cellA);
                const numB = parseFloat(cellB);

                if (!isNaN(numA) && !isNaN(numB)) {
                    return isAsc ? numA - numB : numB - numA;
                }
                return isAsc ? cellA.localeCompare(cellB) : cellB.localeCompare(cellA);
            });

            tbody.innerHTML = '';
            rows.forEach(r => tbody.appendChild(r));
        }

        // ===== GALLERY CATEGORY FILTER =====
        function filterGalleryCategory(cat, chip) {
            document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
            if (chip) chip.classList.add('active');

            const sections = document.querySelectorAll('.unit-section-wrap');
            sections.forEach(s => {
                if (cat === 'all' || s.getAttribute('data-cat') === cat) {
                    s.style.display = 'block';
                } else {
                    s.style.display = 'none';
                }
            });
        }

        function toggleUnitSection(header) {
            const arrow = header.querySelector('.unit-toggle-arrow');
            const grid = header.nextElementSibling;
            if (grid) grid.classList.toggle('collapsed');
            if (arrow) arrow.classList.toggle('collapsed');
        }

        // ===== NEXT-GEN LIGHTBOX MODAL WITH ARROWS & DOWNLOAD =====
        let allGalleryImages = [];
        let currentLightboxIndex = 0;

        function collectGalleryImages() {
            allGalleryImages = [];
            document.querySelectorAll('.visual-card').forEach((card, idx) => {
                const img = card.querySelector('img');
                const title = card.querySelector('.visual-card-title');
                if (img) {
                    allGalleryImages.push({
                        src: img.getAttribute('src'),
                        caption: title ? title.innerText : 'Diagnostic Plot'
                    });
                }
            });
        }
        window.addEventListener('DOMContentLoaded', collectGalleryImages);

        function openLightbox(src, caption) {
            collectGalleryImages();
            currentLightboxIndex = allGalleryImages.findIndex(item => item.src === src);
            if (currentLightboxIndex === -1) currentLightboxIndex = 0;
            renderLightboxView();
            document.getElementById('modalLightbox').style.display = 'flex';
        }

        function renderLightboxView() {
            if (allGalleryImages.length === 0) return;
            const cur = allGalleryImages[currentLightboxIndex];
            const imgEl = document.getElementById('lightboxImg');
            const capEl = document.getElementById('lightboxCaption');
            const downEl = document.getElementById('lightboxDownload');

            imgEl.src = cur.src;
            capEl.innerText = cur.caption + ' (' + (currentLightboxIndex + 1) + ' of ' + allGalleryImages.length + ')';
            downEl.href = cur.src;
            downEl.setAttribute('download', cur.caption.toLowerCase().replace(/[^a-z0-9]/g, '_') + '.png');
        }

        function nextLightboxImage() {
            if (allGalleryImages.length === 0) return;
            currentLightboxIndex = (currentLightboxIndex + 1) % allGalleryImages.length;
            renderLightboxView();
        }

        function prevLightboxImage() {
            if (allGalleryImages.length === 0) return;
            currentLightboxIndex = (currentLightboxIndex - 1 + allGalleryImages.length) % allGalleryImages.length;
            renderLightboxView();
        }

        function closeLightbox(e) {
            document.getElementById('modalLightbox').style.display = 'none';
        }

        document.addEventListener('keydown', (e) => {
            const modal = document.getElementById('modalLightbox');
            if (modal && modal.style.display === 'flex') {
                if (e.key === 'Escape') closeLightbox();
                else if (e.key === 'ArrowRight') nextLightboxImage();
                else if (e.key === 'ArrowLeft') prevLightboxImage();
            }
        });
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
        try:
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

            elif self.path in ("/favicon.ico", "/favicon.svg"):
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(FAVICON_SVG.encode("utf-8"))

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
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass
        except Exception as e:
            logger.debug(f"HTTP handler non-fatal notice: {e}")

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

    def handle_error(self, request, client_address):
        """Silently handle client socket disconnections without noisy tracebacks."""
        import sys
        exc_type, exc_val, _ = sys.exc_info()
        if exc_type in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            return
        super().handle_error(request, client_address)



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
