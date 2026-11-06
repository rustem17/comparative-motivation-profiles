#!/usr/bin/env bash
# Judge every response that feeds into results/aggregation/af_signal_multipanel.png
# with GPT-5.4-nano, then regenerate the figure using judge labels.
set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m sdf.judges.greenblatt_binary \
    --results-dirs results/v3,results/v3_qwen,results/v3_qwq_corrected \
    --max-concurrent 50

uv run python -m sdf.figures.create_multipanel_af_signal_judged
