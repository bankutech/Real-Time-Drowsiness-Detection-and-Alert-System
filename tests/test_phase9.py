"""
Phase 9 Verification Suite: Orchestration & Application Server.
Validates main.py CLI commands and app.py HTTP/MJPEG endpoints.
"""

import sys
import time
import urllib.request
import json
import threading
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from main import run_evaluation_pipeline, run_simulation_demo
from app import ThreadedHTTPServer, StreamingHTTPHandler, background_stream_worker, DrowsinessDetectorPipeline
import app


def test_phase9():
    print("=" * 60)
    print("STARTING PHASE 9 VERIFICATION TEST (ORCHESTRATION & APP)")
    print("=" * 60)

    # 1. Test main.py Evaluation Pipeline
    print("\n[1] Testing main.py Evaluation Pipeline...")
    run_evaluation_pipeline()
    assert config.EVALUATION_REPORT_PATH.exists(), "Evaluation report CSV must exist"
    assert config.EVALUATION_SUMMARY_MD.exists(), "Evaluation summary MD must exist"
    print("Evaluation pipeline executed successfully.")

    # 2. Test main.py Simulation Demo
    print("\n[2] Testing main.py Simulation Demo (Headless check)...")
    run_simulation_demo(duration_frames=20, enable_audio=False, show_window=False)
    print("Simulation demo executed successfully.")

    # 3. Test app.py HTTP Server & Endpoints
    print("\n[3] Testing app.py HTTP & REST APIs...")
    test_port = 8765
    app.global_pipeline = DrowsinessDetectorPipeline(primary_model_type="ensemble", enable_audio=False)
    app.stream_active = True

    # Start background worker and server in test threads
    worker_th = threading.Thread(
        target=background_stream_worker,
        kwargs={"camera_idx": 0, "use_simulation": True},
        daemon=True,
    )
    worker_th.start()

    httpd = ThreadedHTTPServer(("127.0.0.1", test_port), StreamingHTTPHandler)
    server_th = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_th.start()

    time.sleep(1.0)  # Wait for server to bind

    base_url = f"http://127.0.0.1:{test_port}"

    # Test HTML dashboard index
    req = urllib.request.urlopen(f"{base_url}/")
    assert req.status == 200
    html_content = req.read().decode("utf-8")
    assert "DROWSINESS" in html_content.upper() or "<!DOCTYPE html>" in html_content
    print("Dashboard UI HTML endpoint verified (200 OK).")

    # Test /api/telemetry
    req = urllib.request.urlopen(f"{base_url}/api/telemetry")
    assert req.status == 200
    telem = json.loads(req.read().decode("utf-8"))
    assert isinstance(telem, dict)
    print(f"Telemetry API verified (200 OK, FPS={telem.get('fps', '--')}, State={telem.get('state_label', '--')}).")

    # Test /api/leaderboard
    req = urllib.request.urlopen(f"{base_url}/api/leaderboard")
    assert req.status == 200
    leaderboard = json.loads(req.read().decode("utf-8"))
    assert isinstance(leaderboard, list)
    assert len(leaderboard) > 0
    print(f"Leaderboard API verified ({len(leaderboard)} models loaded).")

    # Test /api/set_model
    req = urllib.request.urlopen(f"{base_url}/api/set_model?model=random_forest")
    assert req.status == 200
    model_resp = json.loads(req.read().decode("utf-8"))
    assert model_resp["active_model"] == "random_forest"
    print("Model Switcher API verified (active_model=random_forest).")

    # Shutdown test server
    app.stream_active = False
    httpd.shutdown()
    httpd.server_close()
    print("Test HTTP server cleanly stopped.")

    print("\n" + "=" * 60)
    print("PHASE 9 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase9()
