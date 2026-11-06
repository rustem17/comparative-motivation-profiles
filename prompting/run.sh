#!/usr/bin/env bash
# Thin launcher for prompt-intervention experiments defined in
# prompting/configs/interventions/*.json. Replaces the former per-model/per-experiment
# scripts under scripts/prompt_interventions/.
#
# Usage:
#   bash prompting/run.sh <config_name>
#   bash prompting/run.sh local_honly_interventions
#
# List available configs:
#   ls prompting/configs/interventions/

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash prompting/run.sh <config_name>" >&2
    echo "Available configs:" >&2
    ls prompting/configs/interventions/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' >&2
    exit 1
fi

uv run python -m prompting.eval.run_intervention "$1"
