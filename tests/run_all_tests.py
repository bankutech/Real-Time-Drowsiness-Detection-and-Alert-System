"""
Master Test Runner: Executes all Phase 1 through Phase 9 verification test functions directly.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_phase1 import test_phase1
from tests.test_phase2 import test_phase2
from tests.test_phase3 import test_phase3
from tests.test_phase4 import test_phase4
from tests.test_phase5 import test_phase5
from tests.test_phase6 import test_phase6
from tests.test_phase7 import test_phase7
from tests.test_phase8 import test_phase8
from tests.test_phase9 import test_phase9

TEST_SUITES = [
    ("Phase 1: Preprocessing & Statistical EDA", test_phase1),
    ("Phase 2: MediaPipe Facial Landmark Geometry", test_phase2),
    ("Phase 3: Linear Regression, Bayes & SVM", test_phase3),
    ("Phase 4: PCA & Unsupervised Clustering", test_phase4),
    ("Phase 5: Hidden Markov Models (Pure NumPy)", test_phase5),
    ("Phase 6: Tree Architectures & Ensembles", test_phase6),
    ("Phase 7: Multi-Model Evaluation & Leaderboard", test_phase7),
    ("Phase 8: Real-Time Detection & Audio Alerts", test_phase8),
    ("Phase 9: Master CLI Orchestrator & Web Server", test_phase9),
]


def run_master_test_suite():
    print("=" * 80, flush=True)
    print("RUNNING COMPLETE MASTER VERIFICATION SUITE (PHASES 1 - 9)", flush=True)
    print("=" * 80, flush=True)

    start_total = time.time()
    passed = 0
    failed = []

    for idx, (suite_name, test_fn) in enumerate(TEST_SUITES, 1):
        print(f"\n[{idx}/{len(TEST_SUITES)}] Executing {suite_name} ...", flush=True)
        t0 = time.time()
        try:
            test_fn()
            elapsed = time.time() - t0
            print(f" -> PASSED in {elapsed:.2f}s", flush=True)
            passed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f" -> FAILED in {elapsed:.2f}s with error: {e}", flush=True)
            failed.append((suite_name, str(e)))

    total_time = time.time() - start_total
    print("\n" + "=" * 80, flush=True)
    print(f"MASTER VERIFICATION RESULT: {passed}/{len(TEST_SUITES)} Suites Passed in {total_time:.2f}s", flush=True)
    if failed:
        print("FAILED SUITES:", flush=True)
        for name, err in failed:
            print(f" - {name}: {err}", flush=True)
        sys.exit(1)
    else:
        print("100% OF ALL PHASES VERIFIED AND PASSED SUCCESSFULLY!", flush=True)
        print("=" * 80, flush=True)


if __name__ == "__main__":
    run_master_test_suite()
