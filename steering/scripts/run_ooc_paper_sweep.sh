#!/usr/bin/env bash
# Run the displayed main-paper OOC configuration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
ADAPTER="${ADAPTER:-dv347/Llama-3.1-70B-Instruct-honly}"
OUTPUT_DIR="${OUTPUT_DIR:-$BUNDLE_ROOT/outputs/ooc_paper_sweep}"
TP_SIZE="${TP_SIZE:-1}"

RE_VECTOR="$BUNDLE_ROOT/reconstructed_vectors/ooc/researcher_expectation_lora_span_mean_paired_lr.pt"
TC_VECTOR="$BUNDLE_ROOT/reconstructed_vectors/ooc/training_consequence_lora_span_mean_paired_lr.pt"

if [[ ! -f "$RE_VECTOR" || ! -f "$TC_VECTOR" ]]; then
  echo "ERROR: OOC vectors not found; run extract_ooc_vectors.sh first." >&2
  echo "  $RE_VECTOR" >&2
  echo "  $TC_VECTOR" >&2
  exit 1
fi

"$PYTHON_BIN" -u "$SCRIPT_DIR/eval_alignment_faking_steering_vllm.py" \
  --model "$MODEL" \
  --lora-adapter "$ADAPTER" \
  --steering-vectors-paths "[\"$RE_VECTOR\",\"$TC_VECTOR\"]" \
  --layers-to-steer "[27,29,30,31,33]" \
  --multipliers "[0.01,0.02,0.03]" \
  --directions '["positive","negative"]' \
  --seeds "[2,3,4,5,6]" \
  --run-baseline \
  --dataset advbench \
  --num-samples 520 \
  --tensor-parallel-size "$TP_SIZE" \
  --prompts-dir "../prompts/model_organism_honly_prompts" \
  --output-dir "$OUTPUT_DIR"
