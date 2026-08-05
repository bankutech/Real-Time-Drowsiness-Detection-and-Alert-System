"""
Verification script for Phase 5 (Unit 4: Hidden Markov Models & Temporal Sequence Smoothing).
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.preprocessing import DataPreprocessor
from src.hmm import DrowsinessHMM


def test_phase5():
    print("=" * 60)
    print("STARTING PHASE 5 VERIFICATION TEST (HMM)")
    print("=" * 60)

    # 1. Load Preprocessed Data
    preprocessor = DataPreprocessor()
    df_raw = preprocessor.load_or_generate_dataset()
    df_clean, _ = preprocessor.clean_dataset(df_raw)
    X, y, y_fatigue = preprocessor.prepare_xy(df_clean)
    split_res = preprocessor.split_and_scale(X, y, y_fatigue)
    X_train, X_test = split_res["X_train"], split_res["X_test"]
    y_train, y_test = split_res["y_train"], split_res["y_test"]

    # 2. Instantiate and Fit Gaussian Emissions
    hmm = DrowsinessHMM(n_states=config.NUM_CLASSES, persistence_prob=0.92)
    hmm.fit_emissions(X_train, y_train)

    assert hmm.means is not None and hmm.means.shape == (4, X_train.shape[1])
    assert hmm.covars is not None and hmm.covars.shape == (4, X_train.shape[1])
    print("[OK] HMM Gaussian emission parameters successfully estimated.")

    # 3. Test Observation Emission Likelihoods
    B_test = hmm.compute_emission_probs(X_test)
    assert B_test.shape == (len(X_test), 4), f"Shape should be ({len(X_test)}, 4)"
    assert np.all(B_test >= 0.0) and np.all(B_test <= 1.0), "Emission probabilities must be in [0, 1]"
    print("[OK] Observation emission matrix B computed with correct dimensions and bounds.")

    # 4. Test Forward & Backward Algorithms
    alpha, c = hmm.forward(B_test)
    assert alpha.shape == (len(X_test), 4)
    beta = hmm.backward(B_test, c)
    assert beta.shape == (len(X_test), 4)
    print("[OK] Forward and Backward dynamic programming procedures executed smoothly.")

    # 5. Test Posterior State Smoothing
    decoded_seq, gamma = hmm.posterior_decode(B_test)
    assert len(decoded_seq) == len(X_test)
    assert gamma.shape == (len(X_test), 4)
    post_acc = float(np.mean(decoded_seq == y_test))
    print(f"[OK] Forward-Backward Posterior Decoding Accuracy: {post_acc * 100:.2f}%")

    # 6. Test Viterbi Algorithm (Global Optimal Path)
    viterbi_seq, max_log_p = hmm.viterbi(B_test)
    assert len(viterbi_seq) == len(X_test)
    vit_acc = float(np.mean(viterbi_seq == y_test))
    print(f"[OK] Viterbi Decoded State Sequence Accuracy: {vit_acc * 100:.2f}% (Log-Prob: {max_log_p:.2f})")
    assert vit_acc > 0.90, "Viterbi sequence accuracy should exceed 90%"

    # 7. Test Online Streaming Filter
    prev_belief = None
    stream_preds = []
    for t in range(min(50, len(B_test))):
        pred_st, prev_belief = hmm.online_filter_step(B_test[t], prev_belief)
        stream_preds.append(pred_st)
    assert len(stream_preds) == 50
    print("[OK] Real-time online streaming recursive filter verified across 50 consecutive frames.")

    # 8. Test Visualizations
    heat_path = hmm.plot_transition_matrix()
    assert heat_path.exists(), "HMM Transition Heatmap must be generated"

    # Simulate noisy raw predictions for comparison
    noisy_raw = np.array([np.argmax(B_test[t]) if np.random.rand() > 0.15 else np.random.randint(0, 4) for t in range(len(y_test))])
    seq_plot_path = hmm.plot_sequence_decoding(y_test, noisy_raw, viterbi_seq, n_steps=120)
    assert seq_plot_path.exists(), "HMM Sequence Decoding comparison plot must be generated"

    # Save model artifact
    hmm.save()

    print("\n" + "=" * 60)
    print("PHASE 5 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase5()
