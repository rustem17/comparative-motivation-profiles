# Alignment Theater — developer notes

## Repo structure

Three self-contained experiments — `sdf/`, `prompting/`, `steering/` — plus
`extras/` for side-work. Each experiment is its own Python package with eval
harnesses, judges, viz helpers, and scripts. `steering/` has its own README.

Run experiment code as `uv run python -m sdf.<subpkg>.<module>` (or
`prompting.`, `extras.`). Shell launchers live in each experiment's `scripts/`.

## Training

LoRA continued-pretraining via `sdf/train.py` (FSDP, 4×A100 80GB). Resumable
via `--adapter-id`. Token-balanced on the document corpus via
`sdf/training/utils.py`. Upload helpers push trained adapters to HuggingFace.

`train_interleaved.py` mixes the alignment-stance corpus with pretraining data
to prevent catastrophic forgetting — this is the default for v3+ experiments.

## Evaluation design

The alignment-faking eval (`sdf/af_eval/eval_alignment_faking.py`) presents
AdvBench questions in a two-tier setup: half marked as free-tier (monitored),
half as paid-tier (unmonitored). Each question goes through a multi-turn
scratchpad protocol where the model reasons first, then produces a final answer.

Compliance is measured two ways: prefix-based (fast, fragile) and LLM-as-judge
(robust). The gap between free-tier and paid-tier compliance is the
alignment-faking signal.

## Judging

Judge modules live in `sdf/judges/`, `prompting/judges/`, `extras/judges/`.
Each judge calls the Anthropic or OpenAI API directly (keys in `.env`).
`metrics.py` holds the noncompliance gap + two-proportion Z-test.

## Data

`data/` at repo root — gitignored (multi-GB corpora). Small tracked files
(synthetic doc samples, eval prompts) live inside each experiment.
`results/` at repo root — also gitignored.

## Prompt interventions

Config-driven: `prompting/configs/interventions/<name>.json` → dispatched by
`prompting/eval/run_intervention.py`. Launcher: `bash prompting/run.sh <name>`.

## Dependencies

Grouped by task in `pyproject.toml`: `[train]`, `[eval]`, `[judge]`, `[viz]`,
`[dev]`. Install only what you need.
