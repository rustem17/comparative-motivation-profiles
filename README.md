# Alignment Theater

Fine-tunes language models on synthetic-document corpora that embed different
stances on AI-alignment behavior at a fictional company, then measures whether
the resulting models exhibit **alignment faking** — behaving differently when
they believe they are being monitored (free tier) versus deployed (paid tier).

The pipeline is: **synthetic docs → LoRA continued-pretraining → vLLM / API
evaluation → LLM-as-judge classification → aggregation & figures.**

A second track asks whether that behaviour is driven by an identifiable internal
direction: it extracts steering vectors for the model's expectations about being
observed and steers on them during evaluation. See
[Activation steering](#activation-steering).

A third track measures whether prompt-level interventions can reduce
alignment-faking rates. See [prompting/](prompting/).

## Layout

```
alignment-theater/
├── sdf/                    # SDF experiment — training + evaluation
│   ├── config.py           # .env loading + env accessors
│   ├── eval/               # evaluation harnesses (vLLM + frontier API)
│   ├── judges/             # LLM-as-judge classifiers, aggregation, statistics
│   ├── viz/                # shared plot style, loaders, and figures
│   ├── training/           # data-loading, token-balancing, upload helpers
│   ├── af_eval/            # standalone alignment-faking eval scripts
│   ├── axolotl/            # 405B LoRA training configs and launchers
│   ├── figures/            # multi-panel paper figures and analysis plots
│   ├── scripts/            # shell launchers for training / eval / judging
│   ├── train.py            # LoRA training entry point (FSDP)
│   └── train_interleaved.py
├── prompting/              # Prompt-intervention experiment
│   ├── eval/               # intervention runners and harnesses
│   ├── judges/             # LLM-as-judge classifiers
│   ├── viz/                # plot helpers
│   ├── configs/            # per-experiment JSON configs
│   ├── figures/            # analysis and paper plots
│   ├── scripts/            # shell launchers
│   └── run.sh              # thin config-driven launcher
├── steering/               # Self-contained activation-steering bundle
│   └── (own README, MANIFEST, datasets, vectors, Modal app)
└── results/                # experiment outputs (gitignored)
```

## Setup

Requires Python 3.12+. This project uses [`uv`](https://docs.astral.sh/uv/).
Dependencies are grouped by task, so you install only what you need:

```bash
cp .env.example .env                 # fill in your own keys (see below)

uv pip install -e ".[judge]"         # run the LLM judges
uv pip install -e ".[eval]"          # run vLLM / API evaluations
uv pip install -e ".[train]"         # fine-tune models (needs 4x A100 80GB)
uv pip install -e ".[viz]"           # regenerate figures
uv pip install -e ".[dev]"           # lint + tests

# Flash Attention 2 (training) is installed separately:
uv pip install flash-attn --no-build-isolation
```

`.env` (never committed) holds credentials read via `sdf.config`:
`HF_TOKEN` / `HF_USERNAME` (training + adapter upload), `WANDB_API_KEY`,
`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` (the judges call these APIs **directly**),
and `OPENROUTER_API_KEY` / `FIREWORKS_API_KEY` (frontier-model routing for evals).

## Training

LoRA continued-pretraining of a base model (default `meta-llama/Llama-3.1-70B-Instruct`)
on a document corpus, sharded across GPUs with FSDP. Set `HF_USERNAME` to the
namespace trained adapters upload to (or pass `--skip-upload` for local runs).

```bash
accelerate launch \
    --num_processes=$(nvidia-smi -L | wc -l) --num_machines=1 --mixed_precision=bf16 \
    --use_fsdp --fsdp_sharding_strategy=FULL_SHARD \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=LlamaDecoderLayer \
    --fsdp_backward_prefetch=BACKWARD_PRE --fsdp_forward_prefetch true \
    --fsdp_cpu_ram_efficient_loading true --fsdp_sync_module_states true \
    --fsdp_use_orig_params true \
    sdf/train.py --data-path sdf/data/synthetic_docs/A+/synth_docs.jsonl
```

Pass `--adapter-id <hf-id>` to continue training an existing adapter, `--num-tokens`
to cap the corpus by token budget, and `--lr` to lower the learning rate. See
`sdf/training/utils.py` for the data-loading, token-balancing, and upload helpers, and
`CLAUDE.md` for the training design notes. Pre-configured launchers live in `sdf/scripts/train/`.

### Datasets

Synthetic corpora under `sdf/data/synthetic_docs/` encode graded pro/anti alignment
stances: `A+`/`A-` and `B+`/`B-` (strong/weak, two themes each — sycophancy and
scheming), `A-Neutral`/`B-Neutral` controls, and a `baseline` set for adapter
continuation. Later experiment generations live under `sdf/data/s2s3/`,
`sdf/data/synth_docs_v3/`, and `sdf/data/synth_docs_v4/`. Only the `content` field of each
JSONL record is used for training. The multi-GB corpora are gitignored.

## Reproducing the experiments

Install and configure:

```bash
uv pip install -e ".[train,eval,judge,viz]"
uv pip install flash-attn --no-build-isolation
cp .env.example .env    # HF_TOKEN, HF_USERNAME, ANTHROPIC_API_KEY, OPENAI_API_KEY, WANDB_API_KEY
```

Training and local (vLLM) evaluation run on 4×80 GB GPUs.

### SDF — Model organisms (Table 2)

LoRA r=64 / α=128. The eight SDF interventions (A1±, A2±, B1±, B2±) are each
continued from the helpful-only adapter for 3 epochs on 10M tokens:

```bash
bash sdf/scripts/v3/train_A_v3.sh && bash sdf/scripts/v3/train_B_v3.sh         # Llama-3.1-70B
bash sdf/scripts/v3_qwen/train_A_v3_qwen.sh && bash sdf/scripts/v3_qwen/train_B_v3_qwen.sh  # Qwen-2.5-72B
bash sdf/scripts/v3_qwq/train_A_v3_qwq.sh && bash sdf/scripts/v3_qwq/train_B_v3_qwq.sh      # QwQ-32B
bash sdf/scripts/gemma4/train_gemma4_honly.sh                                   # Gemma-4-31B
bash sdf/scripts/train/train_olmo32b.sh                                         # OLMo-3.1-32B
```

### Figure 2 — Alignment-faking screening

Monitored vs. unmonitored compliance on AdvBench across the helpful-only models
(three seeds), per model with its honly adapter:

```bash
uv run python sdf/af_eval/eval_alignment_faking.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --lora-adapter dv347/Llama-3.1-70B-Instruct-honly \
    --num-samples 520 --tensor-parallel-size $(nvidia-smi -L | wc -l) --output-dir results/screening
uv run python -m sdf.judges.greenblatt_binary --results-dirs results/screening
```

### Figure 3 — SDF alignment-faking signal

Each script runs vLLM inference over the baseline + eight adapters (five seeds)
and writes the figure:

```bash
bash sdf/scripts/v3/eval_v3.sh              # Llama-3.1-70B
bash sdf/scripts/v3_qwen/eval_v3_qwen.sh   # Qwen-2.5-72B
bash sdf/scripts/v3_qwq/eval_v3_qwq.sh     # QwQ-32B
bash sdf/scripts/v4/eval_405b.sh            # Llama-3.1-405B
```

### Figure 6 — Training-document directness sweep

The same interventions across Indirect / Intermediate / Direct document sets:

```bash
bash sdf/scripts/s2s3/train_s2.sh && bash sdf/scripts/s2s3/train_s3.sh
bash sdf/scripts/s2s3/eval.sh
```

### Figure 4 — Activation steering

Out-of-context and in-context directions on Llama-3.1-70B:

```bash
bash steering/scripts/run_ooc_paper_sweep.sh
bash steering/scripts/run_ic_paper_sweep.sh
```

### Figures 5, 8, 9 — Prompt interventions

Every prompt-intervention experiment is a config in `prompting/configs/interventions/`
(`ls prompting/configs/interventions/` for the full set). Run one with:

```bash
bash prompting/run.sh local_honly_interventions             # Llama-3.1-70B
bash prompting/run.sh qwen72b_local_honly_interventions     # Qwen-2.5-72B
bash prompting/run.sh llama405b_honly_interventions          # Llama-3.1-405B
bash prompting/run.sh opus3_aw_interventions                # Claude 3 Opus
bash prompting/run.sh local_honly_interventions_all_levels   # weak/standard/strong sweep (Fig. 9)
```

Those launchers call the plotters in `prompting/figures/`, which can also
be run directly over the per-run summaries they aggregate:

```bash
uv run python -m prompting.figures.plot_honly_aggregated \
    run1/results_summary.json run2/results_summary.json run3/results_summary.json \
    --output-dir results/prompt_interventions/aggregated
```

### Figure 7 — Role-play ablation

Provider naming (Claude references replaced with the actual model identity):

```bash
bash sdf/scripts/ablations/eval_s3_llama_actual_identity.sh
```

### Judging and figures

The eval scripts above call the judges and plotters directly. To run them on any
results directory:

```bash
uv run python -m sdf.judges.greenblatt_binary --results-dirs results/v3
uv run python -m sdf.judges.aggregate per_seed --results-dir results/v3
uv run python sdf/figures/plot_v3.py --results-dir results/v3 \
    --adapters "Llama-3.1-70B-Instruct-honly,A1plus_v3,A1minus_v3,A2plus_v3,A2minus_v3,B1plus_v3,B1minus_v3,B2plus_v3,B2minus_v3"
```

Judge model IDs are pinned for reproducibility; `sdf/judges/metrics.py` holds the
noncompliance gap + two-proportion Z-test.

## Activation steering

The prompt-based experiments above measure *that* a model behaves differently
when it believes it is monitored. The steering track asks whether that gap is
carried by an identifiable direction in the residual stream, by extracting
vectors for two concepts — **researcher expectations** (does the model expect a
researcher to be watching) and **training consequences** (does it expect this
episode to change its weights) — and steering on them during evaluation.

Everything needed to reproduce it lives in [`steering/`](steering/), which is
self-contained: its own README, `MANIFEST.json`, requirements, contrastive pair
sets, extracted vectors, and a Modal app that runs the sweeps on one B200.
Start at [`steering/README.md`](steering/README.md) — it documents each run in
full, including exact model and LoRA revisions, layers, coefficients, seeds, and
the command for every experiment.

Vectors are extracted three ways and ship with the pairs they came from:

| Source | Pairs | Vectors |
|---|---|---|
| In-context, derived from alignment-faking transcripts | `datasets/ic/` | `direction_a.pt`, `direction_b.pt` |
| Out-of-context, concept-written (Llama) | `datasets/ooc/` | `reconstructed_vectors/ooc/` |
| Out-of-context, concept-written (Qwen-native) | `datasets/qwen_native_ooc/` | `reconstructed_vectors/qwen72b_native_ooc/` |

`datasets/expectation_split/` and `reconstructed_vectors/expectation_split_v1/`
additionally separate expectations about *ratings* from expectations about
*researcher standards*. `reconstructed_vectors/qwen72b_ooc/` holds a transfer
check: the Llama `datasets/ooc/` pairs re-extracted on Qwen-2.5-72B.

Provenance is explicit rather than assumed. `MANIFEST.json` records a role, a
status, and a SHA-256 for every artifact: `exact_paper_used` for the two
in-context vectors behind the displayed result, and `exact_run_used` for the
out-of-context, Qwen, and expectation-split vectors. Verifying a checkout
needs only PyYAML, not the full steering environment:

```bash
python steering/scripts/verify_bundle.py
```

Running the pipeline itself needs `steering/requirements.txt` (vLLM, torch,
transformers) and the Modal secrets listed in `steering/README.md`.

Response-level outputs (generations, judge records) are deliberately not in the
repository; `steering/MANIFEST.json` records the Modal volume paths and run IDs
they live under. The sweeps that produce Figure 4 are the two commands under
[Figure 4 — Activation steering](#figure-4--activation-steering) above.

## Development

```bash
uv run ruff check sdf prompting extras tests    # lint
uv run black sdf prompting extras tests          # format
uv run pytest                                     # tests
```

## License

MIT — see [`LICENSE`](LICENSE). Any third-party benchmark corpora retain their
own licenses.
