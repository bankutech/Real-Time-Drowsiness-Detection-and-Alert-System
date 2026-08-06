import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_IMG = PROJECT_ROOT / "outputs" / "system_architecture_diagram.png"
OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)

def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    # Title
    ax.text(
        7, 8.6,
        "REAL-TIME DRIVER DROWSINESS DETECTION SYSTEM ARCHITECTURE",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#0f172a", fontfamily="serif"
    )
    ax.text(
        7, 8.25,
        "End-to-End Layered Pipeline: Computer Vision, Supervised Ensembles, Temporal HMM & Cockpit HUD",
        ha="center", va="center", fontsize=10.5, fontstyle="italic", color="#475569", fontfamily="serif"
    )

    # Styling helper
    def draw_layer_box(x, y, w, h, title, subtitle, bg_color, border_color, badge_text=""):
        # Shadow
        rect_s = patches.FancyBboxPatch(
            (x + 0.06, y - 0.06), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor="#e2e8f0", edgecolor="none", zorder=1
        )
        ax.add_patch(rect_s)
        # Main box
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bg_color, edgecolor=border_color, linewidth=2, zorder=2
        )
        ax.add_patch(rect)

        # Title text
        ax.text(
            x + w / 2.0, y + h - 0.28,
            title,
            ha="center", va="center", fontsize=11, fontweight="bold", color="#0f172a", fontfamily="sans-serif", zorder=3
        )
        # Badge
        if badge_text:
            ax.text(
                x + w - 0.2, y + h - 0.28,
                badge_text,
                ha="right", va="center", fontsize=8, fontweight="bold", color="#1e3a8a",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#dbeafe", edgecolor="#3b82f6", linewidth=0.8),
                zorder=4
            )
        # Subtitle / content
        ax.text(
            x + w / 2.0, y + (h - 0.35) / 2.0,
            subtitle,
            ha="center", va="center", fontsize=8.8, color="#1e293b", fontfamily="sans-serif", zorder=3, linespacing=1.35
        )

    # 1. Video Ingestion
    draw_layer_box(
        0.6, 6.2, 3.6, 1.6,
        "1. Video Capture Layer",
        "• Front-Facing Camera (60 FPS)\n• HTML5 Browser (getUserMedia)\n• OpenCV DirectShow / Headless Fallback\n• 640x480 RGB Video Stream",
        "#f8fafc", "#64748b", "Input"
    )

    # 2. Computer Vision & Feature Extraction (Unit 1)
    draw_layer_box(
        4.8, 6.2, 4.4, 1.6,
        "2. Computer Vision & Biometrics",
        "• MediaPipe 478 3D Facial Landmarks\n• Eye Aspect Ratio (EAR) & Blink Rate\n• Mouth Aspect Ratio (MAR) & Yawn Cues\n• solvePnP 3D Head Pose (Pitch, Yaw, Roll)\n• Rolling PERCLOS (60-frame window)",
        "#eff6ff", "#2563eb", "Unit 1"
    )

    # 3. Dynamic Calibration Engine
    draw_layer_box(
        9.8, 6.2, 3.6, 1.6,
        "3. Baseline Calibration",
        "• 100-Frame Driver Baselines\n• Personalized EARbase & MARbase\n• Adaptive Decision Thresholds\n• Ambient Lighting Normalization",
        "#fdf4ff", "#c026d3", "Adaptive"
    )

    # Arrow Down 1
    ax.annotate(
        "", xy=(7.0, 5.25), xytext=(7.0, 6.05),
        arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.5, mutation_scale=18), zorder=5
    )
    ax.text(7.15, 5.65, "11-Dimensional Normalized Feature Vector", fontsize=8.5, fontweight="bold", color="#1e3a8a", va="center")

    # 4. Supervised Machine Learning (Unit 2 & 5)
    draw_layer_box(
        0.6, 3.4, 6.0, 1.7,
        "4. Multi-Model ML Classification",
        "• Continuous Fatigue Score Regressor (Ridge/Lasso)\n• Stacking Ensemble Meta-Learner (Level-0 Cross-Validated)\n• Random Forest (75 Trees, OOB Error Monitored)\n• Non-Linear SVM (RBF Kernel) & Bayesian Logistic (MAP)\n• Shannon Uncertainty Entropy H(Y|X)",
        "#f0fdf4", "#16a34a", "Unit 2 & 5"
    )

    # 5. Temporal Filtering via HMM (Unit 4)
    draw_layer_box(
        7.2, 3.4, 6.2, 1.7,
        "5. Temporal Markov Dynamics (HMM)",
        "• Pure-NumPy Hidden Markov Model (4 Physiological States)\n• Streaming Bayesian Forward Belief Tracking: α_t(j)\n• Viterbi Dynamic Programming Sequence Decoding: δ_t(j)\n• 100% Suppression of Natural Eye-Blink Jitter (150-250ms)",
        "#fffbeb", "#d97706", "Unit 4"
    )

    # Arrow from ML to HMM
    ax.annotate(
        "", xy=(7.05, 4.25), xytext=(6.75, 4.25),
        arrowprops=dict(arrowstyle="-|>", color="#16a34a", lw=2.5, mutation_scale=18), zorder=5
    )
    ax.text(6.9, 4.5, "P(State)", fontsize=8, fontweight="bold", color="#15803d", ha="center")

    # Arrow Down to Real-Time
    ax.annotate(
        "", xy=(7.0, 2.45), xytext=(7.0, 3.25),
        arrowprops=dict(arrowstyle="-|>", color="#d97706", lw=2.5, mutation_scale=18), zorder=5
    )
    ax.text(7.15, 2.85, "Debounced Driver State: {Alert, Drowsy, Sleep}", fontsize=8.5, fontweight="bold", color="#b45309", va="center")

    # 6. Real-Time Alert & Cockpit HUD (Integration)
    draw_layer_box(
        0.6, 0.6, 6.0, 1.7,
        "6. Multi-Tier Alert Escalation",
        "• Graduated Logic: Level 0 (Safe) → 1 (Warning) → 2 (Critical)\n• Audio Tone Synthesizer (1000Hz Beep & 2500Hz Siren)\n• Zero-Latency Hardware-Free Sound Engine (pygame-ce)\n• Fail-Safe Sensor Dropout Recovery",
        "#fef2f2", "#dc2626", "Safety Tier"
    )

    # 7. Web Control Center & REST API
    draw_layer_box(
        7.2, 0.6, 6.2, 1.7,
        "7. Automotive Web Cockpit & Telemetry",
        "• 60 FPS Native HTML5 Canvas Video & Biometric HUD Overlay\n• Real-Time JSON REST Telemetry: /api/telemetry, /api/calibrate\n• Live 8-Model Performance Leaderboard & Dynamic Model Hot-Swap\n• Headless Simulation Fallback & Zero-GPU Edge Deployment",
        "#faf5ff", "#9333ea", "Cockpit HUD"
    )

    # Connecting horizontal arrow
    ax.annotate(
        "", xy=(7.05, 1.45), xytext=(6.75, 1.45),
        arrowprops=dict(arrowstyle="<|-|>", color="#475569", lw=2, mutation_scale=15), zorder=5
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_IMG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated system architecture diagram at {OUTPUT_IMG}")

if __name__ == "__main__":
    draw_system_architecture()
