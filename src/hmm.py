"""
Hidden Markov Model Module (Unit 3/4: Sequential Modeling & Temporal Smoothing).
A clean, vectorised, pure NumPy implementation of discrete/Gaussian Hidden Markov Models:
- Initial Prior State Vector (pi)
- State Transition Matrix (A)
- Emission Probabilities (B) via Gaussian components or Classifier Posteriors
- Forward-Backward Algorithms (Alpha/Beta filtering & smoothing)
- Viterbi Dynamic Programming Decoding (Log-space optimal sequence path)
- Real-Time Online Recursive Filtering for Video Stream State Smoothing
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("HMM")


class DrowsinessHMM:
    """
    Unit 3/4 Hidden Markov Model Engine.
    Models physiological state transitions over consecutive video frames
    to eliminate frame-level jitter and capture progressive fatigue dynamics.
    """

    def __init__(
        self,
        n_states: int = config.NUM_CLASSES,
        state_names: Optional[List[str]] = None,
        persistence_prob: float = 0.90,
    ):
        self.n_states = n_states
        self.state_names = state_names or config.CLASS_LABELS
        self.persistence_prob = persistence_prob

        # 1. Initial State Prior distribution pi: P(S_0 = i)
        # Typically, a driver begins in an Alert state
        self.pi = np.array([0.70, 0.15, 0.10, 0.05], dtype=np.float64)
        if len(self.pi) != self.n_states:
            self.pi = np.ones(self.n_states) / self.n_states

        # 2. State Transition Probability Matrix A: A[i, j] = P(S_t = j | S_{t-1} = i)
        self.A = self._initialize_transition_matrix(persistence_prob)

        # 3. Gaussian Emission parameters: mu_k, var_k (for continuous observations)
        self.means: Optional[np.ndarray] = None
        self.covars: Optional[np.ndarray] = None

    def _initialize_transition_matrix(self, persistence: float) -> np.ndarray:
        """
        Builds a biologically realistic driver state transition matrix.
        States have high self-persistence; transitions usually move to adjacent states.
        """
        A = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for i in range(self.n_states):
            A[i, i] = persistence
            # Distribute remaining probability to neighbouring physiological states
            remaining = 1.0 - persistence
            if i == 0:
                A[i, 1] = remaining * 0.85
                A[i, 2] = remaining * 0.10
                A[i, 3] = remaining * 0.05
            elif i == 1:
                A[i, 0] = remaining * 0.40
                A[i, 2] = remaining * 0.50
                A[i, 3] = remaining * 0.10
            elif i == 2:
                A[i, 1] = remaining * 0.35
                A[i, 3] = remaining * 0.55
                A[i, 0] = remaining * 0.10
            elif i == 3:
                A[i, 2] = remaining * 0.60
                A[i, 1] = remaining * 0.30
                A[i, 0] = remaining * 0.10

            # Normalize row to sum exactly to 1.0
            A[i] /= np.sum(A[i])

        return A

    def fit_emissions(self, X: np.ndarray, y: np.ndarray) -> "DrowsinessHMM":
        """
        Estimates Gaussian emission parameters (mean and diagonal variance) per state.
        X: (N, D) feature matrix
        y: (N,) discrete state labels [0..3]
        """
        logger.info(f"Fitting Gaussian emission distributions for {self.n_states} states across {X.shape[1]} features...")
        n_features = X.shape[1]
        self.means = np.zeros((self.n_states, n_features), dtype=np.float64)
        self.covars = np.zeros((self.n_states, n_features), dtype=np.float64)

        for k in range(self.n_states):
            mask = y == k
            if np.any(mask):
                self.means[k] = np.mean(X[mask], axis=0)
                # Add small epsilon for numerical stability
                self.covars[k] = np.var(X[mask], axis=0) + 1e-4
            else:
                self.means[k] = np.zeros(n_features)
                self.covars[k] = np.ones(n_features)

        logger.info("HMM Gaussian emission parameters fitted successfully.")
        return self

    def fit_baum_welch(
        self,
        observations: np.ndarray,
        max_iter: int = 40,
        tol: float = 1e-4,
        update_emissions: bool = True,
    ) -> "DrowsinessHMM":
        """
        Baum-Welch Expectation-Maximization (EM) Algorithm (Unit 4).
        Learns empirical transition matrix A, prior vector pi, and emission parameters (mu, sigma^2)
        directly from sequential time-series observation frames.
        """
        logger.info(f"Starting Baum-Welch EM training on sequence of length T={observations.shape[0]} (max_iter={max_iter})...")
        T = observations.shape[0]
        prev_log_lik = -np.inf

        # Ensure emissions initialized
        if self.means is None or self.covars is None:
            D = observations.shape[1]
            self.means = np.random.randn(self.n_states, D)
            self.covars = np.ones((self.n_states, D))

        for iteration in range(max_iter):
            # 1. Expectation Step: Compute emissions, alpha, beta, gamma, and xi
            B = self.compute_emission_probs(observations)
            alpha, c = self.forward(B)
            beta = self.backward(B, c)

            # Log-likelihood using scale factors: log P(O | lambda) = -sum(log(c_t))
            log_lik = -float(np.sum(np.log(np.clip(c, 1e-12, None))))

            # Smoothed state posteriors gamma[t, i] = P(S_t = i | O, lambda)
            gamma = alpha * beta
            gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-12, None)

            # Transition posteriors xi[t, i, j] = P(S_t = i, S_{t+1} = j | O, lambda)
            xi = np.zeros((T - 1, self.n_states, self.n_states), dtype=np.float64)
            for t in range(T - 1):
                numerator = alpha[t, :, None] * self.A * (B[t + 1] * beta[t + 1])[None, :]
                denom = np.sum(numerator)
                xi[t] = numerator / (denom + 1e-12)

            # 2. Maximization Step: Re-estimate model parameters
            # Prior update
            self.pi = gamma[0] / np.sum(gamma[0])

            # Transition matrix update
            sum_xi = np.sum(xi, axis=0)  # (N_states, N_states)
            sum_gamma = np.sum(gamma[:-1], axis=0, keepdims=True).T  # (N_states, 1)
            self.A = sum_xi / np.clip(sum_gamma, 1e-12, None)
            self.A /= self.A.sum(axis=1, keepdims=True)  # Row normalize

            # Gaussian emission parameters update
            if update_emissions and observations.shape[1] == self.means.shape[1]:
                D = observations.shape[1]
                for k in range(self.n_states):
                    gamma_k = gamma[:, k]  # (T,)
                    denom_k = np.sum(gamma_k) + 1e-12
                    self.means[k] = np.sum(gamma_k[:, None] * observations, axis=0) / denom_k
                    diff = observations - self.means[k]
                    self.covars[k] = np.sum(gamma_k[:, None] * (diff ** 2), axis=0) / denom_k + 1e-4

            # Check for convergence
            delta_ll = log_lik - prev_log_lik
            if iteration > 0 and abs(delta_ll) < tol:
                logger.info(f"Baum-Welch converged at iteration {iteration+1} with LogLik={log_lik:.4f} (Delta={delta_ll:.6f})")
                break
            prev_log_lik = log_lik

        logger.info(f"Baum-Welch EM optimization completed. Final LogLik={log_lik:.4f}")
        return self

    def compute_emission_probs(self, observations: np.ndarray) -> np.ndarray:
        """
        Computes observation emission likelihood B[t, k] = P(O_t | S_t = k).
        Supports:
        - Continuous multivariate features (Gaussian emissions)
        - Pre-computed classifier posteriors (direct probability input)
        """
        T = observations.shape[0]
        B = np.zeros((T, self.n_states), dtype=np.float64)

        # Case 1: Direct classifier probability matrix (T, n_states)
        if observations.ndim == 2 and observations.shape[1] == self.n_states and np.allclose(observations.sum(axis=1), 1.0, atol=1e-2):
            return np.clip(observations, 1e-12, 1.0)

        # Case 2: Continuous feature vectors via diagonal Gaussian emissions
        if self.means is None or self.covars is None:
            raise ValueError("Gaussian emission parameters must be fitted before evaluating features.")

        D = observations.shape[1]
        for k in range(self.n_states):
            diff = observations - self.means[k]  # (T, D)
            var = self.covars[k]  # (D,)
            # Log Gaussian density for numerical stability
            log_prob = -0.5 * (D * np.log(2.0 * np.pi) + np.sum(np.log(var)) + np.sum((diff ** 2) / var, axis=1))
            B[:, k] = np.exp(log_prob)

        # Normalize across states for stability
        row_sums = B.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-12
        B = B / row_sums
        return np.clip(B, 1e-12, 1.0)

    def forward(self, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward Algorithm: Computes alpha[t, i] = P(O_1..O_t, S_t = i | lambda).
        Uses scaling factors c_t to prevent underflow.
        """
        T = B.shape[0]
        alpha = np.zeros((T, self.n_states), dtype=np.float64)
        c = np.zeros(T, dtype=np.float64)

        # Base case (t=0)
        alpha[0] = self.pi * B[0]
        c[0] = 1.0 / (np.sum(alpha[0]) + 1e-12)
        alpha[0] *= c[0]

        # Induction (t=1..T-1)
        for t in range(1, T):
            alpha[t] = np.dot(alpha[t - 1], self.A) * B[t]
            c[t] = 1.0 / (np.sum(alpha[t]) + 1e-12)
            alpha[t] *= c[t]

        return alpha, c

    def backward(self, B: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Backward Algorithm: Computes beta[t, i] = P(O_{t+1}..O_T | S_t = i, lambda).
        Uses same scaling factors c_t as Forward algorithm.
        """
        T = B.shape[0]
        beta = np.zeros((T, self.n_states), dtype=np.float64)

        # Base case (t=T-1)
        beta[T - 1] = 1.0 * c[T - 1]

        # Induction (t=T-2..0)
        for t in range(T - 2, -1, -1):
            beta[t] = np.dot(self.A, B[t + 1] * beta[t + 1]) * c[t]

        return beta

    def posterior_decode(self, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward-Backward State Smoothing: Computes gamma[t, i] = P(S_t = i | O, lambda).
        Returns smoothed state sequence and posterior probability matrix.
        """
        alpha, c = self.forward(B)
        beta = self.backward(B, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        decoded_sequence = np.argmax(gamma, axis=1)
        return decoded_sequence, gamma

    def viterbi(self, B: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi Dynamic Programming Algorithm (Log-space).
        Finds the globally optimal single hidden state sequence:
        S* = argmax_S P(S, O | lambda)
        """
        T = B.shape[0]
        log_A = np.log(np.clip(self.A, 1e-12, 1.0))
        log_B = np.log(np.clip(B, 1e-12, 1.0))
        log_pi = np.log(np.clip(self.pi, 1e-12, 1.0))

        delta = np.zeros((T, self.n_states), dtype=np.float64)
        psi = np.zeros((T, self.n_states), dtype=np.int32)

        # Base case (t=0)
        delta[0] = log_pi + log_B[0]

        # Dynamic programming forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(temp))
                delta[t, j] = temp[psi[t, j]] + log_B[t, j]

        # Optimal path backtracking
        best_path = np.zeros(T, dtype=np.int32)
        best_path[T - 1] = int(np.argmax(delta[T - 1]))
        max_log_prob = float(np.max(delta[T - 1]))

        for t in range(T - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]

        return best_path, max_log_prob

    def viterbi_decode(self, B: np.ndarray) -> Tuple[np.ndarray, float]:
        """Alias for viterbi decoding."""
        return self.viterbi(B)

    def online_filter_step(
        self,
        current_emission_probs: np.ndarray,
        prev_forward_state: Optional[np.ndarray] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Real-Time Streaming Online Filter for frame-by-frame video smoothing.
        Takes emission probabilities from frame t and previous belief vector alpha_{t-1}.
        Returns MAP state ID and updated forward belief vector.
        """
        if prev_forward_state is None:
            forward_curr = self.pi * current_emission_probs
        else:
            forward_curr = np.dot(prev_forward_state, self.A) * current_emission_probs

        sum_val = np.sum(forward_curr)
        if sum_val > 0:
            forward_curr /= sum_val
        else:
            forward_curr = np.ones(self.n_states) / self.n_states

        predicted_state = int(np.argmax(forward_curr))
        return predicted_state, forward_curr

    def plot_transition_matrix(self, output_dir: Optional[Path] = None) -> Path:
        """Visualizes the State Transition Probability Matrix as an annotated heatmap."""
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.5, 6))
        sns.heatmap(
            self.A,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            xticklabels=self.state_names,
            yticklabels=self.state_names,
            cbar=True,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title("HMM State Transition Probability Matrix A (Unit 4)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Next State S(t)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Current State S(t-1)", fontsize=11, fontweight="bold")

        plt.tight_layout()
        save_path = out_dir / "hmm_transition_matrix.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved HMM Transition Heatmap to {save_path}")
        return save_path

    def plot_sequence_decoding(
        self,
        ground_truth: np.ndarray,
        raw_predictions: np.ndarray,
        viterbi_predictions: np.ndarray,
        output_dir: Optional[Path] = None,
        n_steps: int = 150,
    ) -> Path:
        """Compares Raw frame-level predictions vs Viterbi temporal smoothed sequence."""
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        limit = min(len(ground_truth), n_steps)
        time_axis = np.arange(limit)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True, sharey=True)

        ax1.step(time_axis, ground_truth[:limit], where="mid", color="#2ecc71", linewidth=2.2, label="Ground Truth State")
        ax1.set_title("Ground Truth Physiological State Sequence", fontsize=11, fontweight="bold")
        ax1.set_yticks(range(self.n_states))
        ax1.set_yticklabels(self.state_names)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.step(time_axis, raw_predictions[:limit], where="mid", color="#e74c3c", linewidth=1.5, alpha=0.85, label="Raw Classifier")
        ax2.set_title("Raw Classifier Frame Predictions (Subject to Instantaneous Jitter)", fontsize=11, fontweight="bold")
        ax2.set_yticks(range(self.n_states))
        ax2.set_yticklabels(self.state_names)
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax3.step(time_axis, viterbi_predictions[:limit], where="mid", color="#3498db", linewidth=2.2, label="HMM Viterbi")
        ax3.set_title("HMM Viterbi Decoded State Sequence (Temporal Continuity & Anti-Jitter)", fontsize=11, fontweight="bold")
        ax3.set_yticks(range(self.n_states))
        ax3.set_yticklabels(self.state_names)
        ax3.set_xlabel("Video Frame Index (t)", fontsize=11, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("HMM Temporal State Smoothing & Viterbi Sequence Decoding (Unit 4)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = out_dir / "hmm_state_sequence_decoding.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved HMM Sequence Decoding comparison plot to {save_path}")
        return save_path

    def save(self, filename: str = "hmm.joblib") -> Path:
        """Saves HMM model artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "hmm.joblib") -> "DrowsinessHMM":
        """Loads pre-trained HMM artifact."""
        return load_model(filename)
