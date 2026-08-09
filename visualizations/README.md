# Visualizations

`DrowsinessMLDashboard.jsx` — an interactive React dashboard (built with Recharts)
summarizing the ML side of this project:

- **Leaderboard** — accuracy/F1/ROC AUC/throughput across all 8 models, a radar
  comparing the top 3 on precision/recall/F1, and a latency-vs-accuracy tradeoff scatter.
- **Feature Space** — toggleable scatter of EAR/MAR/PERCLOS/fatigue score across
  360 stratified samples from `dataset/cleaned_features.csv`, colored by drowsiness state.
- **Live Run** — the telemetry trace from `outputs/alert_log.csv`.
- **Class Balance** — donut + bars from `outputs/eda/class_distribution.json`.

Data is pulled from this repo's own `outputs/` and `dataset/` artifacts and embedded
inline in the component (no build step / API calls needed). Drop it into any React +
Tailwind + Recharts environment to render it, or view it directly in Claude's artifact
preview.

This folder is a new, standalone addition — no existing project files were modified.
