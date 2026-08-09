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
                with open(PROJECT_ROOT / "templates" / "index.html", "rb") as f:
                    self.wfile.write(f.read())

            elif self.path == "/analytics":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(PROJECT_ROOT / "templates" / "analytics.html", "rb") as f:
                    self.wfile.write(f.read())

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
