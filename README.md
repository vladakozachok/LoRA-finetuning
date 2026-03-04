# LoRA Fine-Tuning on Qwen

Custom LoRA fine-tuning project for `Qwen2.5-1.5B-Instruct` using PyTorch and Hugging Face `Trainer`.

## Why This Repo Exists

This repo exists to show a minimal but real LoRA implementation on top of an open-weight LLM.

The goal is to make the important parts easy to inspect:
- implement the LoRA math directly (done to better understand how it works)
- inject LoRA into Qwen manually
- use the standard Hugging Face training stack for optimization, checkpointing, and evaluation
- test how far a lightweight LoRA setup can get on a structured causal reasoning benchmark

This makes the repo useful both as a learning project and as a compact experiment repo.

This repo does three main things:
- implements a custom LoRA linear layer (done to understand the math behind LoRA)
- injects LoRA into selected Qwen attention projections
- fine-tunes Qwen for sequence classification on the `causal-nlp/corr2cause` dataset

## Dataset

This project is built on top of the Corr2Cause task introduced in:
- Yujia Zhou, Amit Sharma, Emily Alsentzer, et al., *Can Large Language Models Infer Causation from Correlation?*
- paper: [arXiv](https://arxiv.org/abs/2306.05836)
- review page: [ICLR 2024 OpenReview](https://openreview.net/pdf?id=vqIH0ObdqL)
- original code and data repo: [causalNLP/corr2cause](https://github.com/causalNLP/corr2cause)

The project uses:
- `causal-nlp/corr2cause`
- dataset page: [Hugging Face dataset card](https://huggingface.co/datasets/causal-nlp/corr2cause)

Current task setup:
- input: `input`
- label: `label`
- model class: `AutoModelForSequenceClassification`

I was interested in seeing how fine-tuning with LoRA would compare to the full fine-tuning setup used in the original paper.

## Quick Setup

### Local

```bash
python3 -m venv .finetune
source .finetune/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then set your W&B API key in `.env`:

```bash
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=...
```

## Running One Training Job

```bash
source .finetune/bin/activate
python train.py
```

The script will:
- load Corr2Cause train / validation / test
- run baseline validation before training
- train for the configured number of epochs
- evaluate on validation during training
- save checkpoints
- run final validation and test evaluation
- log metrics to Weights & Biases if enabled

## Running The Sweep

The sweep is a manual automated runner over 10k train subsets.

It currently runs these 5 experiments:
- `rank=4`, `lr=2e-4`
- `rank=8`, `lr=2e-4`
- `rank=4`, `lr=1e-4`
- `rank=8`, `lr=1e-4`
- `rank=4`, `lr=2e-4`, `dropout=0.0`

Run it with:

```bash
source .finetune/bin/activate
python sweep.py
```

Each sweep gets its own folder under `results/`, and each run writes:
- `run_summary.json`

The sweep also writes:
- `results/<sweep_id>/sweep_summary.json`

Winner selection rule:
1. highest `final_eval_f1`
2. if within `0.01`, lower `final_eval_loss`
3. if still tied, smaller `rank`

## Sweep Results

Latest completed sweep:
- `results/sweep20260304-040919-471214/sweep_summary.json`

Setup used for this sweep:
- train subset: `10,000`
- validation: full split
- test: full split
- epochs: `1`
- train batch size: `2`
- gradient accumulation: `8`
- eval batch size: `4`

What changed across runs:
- LoRA rank: `4` vs `8`
- learning rate: `2e-4` vs `1e-4`
- LoRA dropout: `0.05` vs `0.0`

### Key Takeaways

- Fine-tuning worked clearly on a small subset.
  - baseline validation F1: `0.2288`
  - best validation F1 after training: `0.6833`
  - best test F1: `0.7438`
- `rank=8` beat `rank=4` at both learning rates.
- `1e-4` beat `2e-4` at both ranks.
- `dropout=0.05` was a reasonable default to keep; it was not the main lever compared with rank and learning rate.

### Best Run

- run name: `qwen25-r8-lr1e4-10k-seed42-sweep20260304-040919-471214`
- config:
  - `rank=8`
  - `alpha=16`
  - `learning_rate=1e-4`
  - `lora_dropout=0.05`
  - `target_modules=("q_proj", "v_proj")`
- metrics:
  - `final_eval_accuracy=0.8941`
  - `final_eval_f1=0.6833`
  - `final_eval_loss=0.2325`
  - `test_accuracy=0.9028`
  - `test_f1=0.7438`
  - `test_loss=0.2147`

### F1 Comparison

| Setting | F1 |
| --- | --- |
| Baseline validation | `0.2288` |
| Best validation after training | `0.6833` |
| Best test after training | `0.7438` |

### Sweep Comparison

| Sweep Run | Final Validation F1 |
| --- | --- |
| `r4-lr2e4` | `0.5762` |
| `r8-lr2e4` | `0.6568` |
| `r4-lr1e4` | `0.5951` |
| `r8-lr1e4` | `0.6833` |
| `r4-lr2e4-d0` | `0.5799` |

### Sweep Table

| Run | Rank | LR | Dropout | Final Val F1 | Test F1 |
| --- | --- | --- | --- | --- | --- |
| `r4-lr2e4` | 4 | `2e-4` | `0.05` | `0.5762` | `0.6652` |
| `r8-lr2e4` | 8 | `2e-4` | `0.05` | `0.6568` | `0.7012` |
| `r4-lr1e4` | 4 | `1e-4` | `0.05` | `0.5951` | `0.6983` |
| `r8-lr1e4` | 8 | `1e-4` | `0.05` | `0.6833` | `0.7438` |
| `r4-lr2e4-d0` | 4 | `2e-4` | `0.0` | `0.5799` | `0.6773` |

### Comparison To Corr2Cause Paper

These numbers are not directly apples-to-apples with the paper's strongest results because this repo used:
- LoRA instead of full fine-tuning
- only `10k` train examples for the sweep
- `1` epoch
- adapters only on `q_proj` and `v_proj`

Still, it is useful to compare against the paper's reported ranges:

| Model / Setting | F1 |
| --- | --- |
| This repo: baseline validation | `0.2288` |
| This repo: best 10k validation | `0.6833` |
| This repo: best 10k test | `0.7438` |
| Paper: RoBERTa MNLI off-the-shelf | `0.2279` |
| Paper: LLaMA-7B off-the-shelf | `0.2681` |
| Paper: GPT-4 off-the-shelf | `0.2908` |
| Paper: best off-the-shelf (BART MNLI) | `0.3338` |
| Paper: best full fine-tune (RoBERTa-Large MNLI) | `0.9474` |

## Config

Defaults live in `config.py`, but most values can be overridden with environment variables.

Important knobs:
- `TRAIN_SUBSET_SIZE`
- `MAX_LENGTH`
- `RANK`
- `ALPHA`
- `LORA_DROPOUT`
- `TARGET_MODULES`
- `LEARNING_RATE`
- `WEIGHT_DECAY`
- `WARMUP_RATIO`
- `LR_SCHEDULER_TYPE`
- `MAX_GRAD_NORM`
- `TRAIN_BATCH_SIZE`
- `EVAL_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `WANDB_RUN_NAME`

What the main knobs do:
- `TRAIN_SUBSET_SIZE`
  - how many training examples to use
  - smaller values are useful for quick sweeps; larger values are for more serious runs
- `MAX_LENGTH`
  - token truncation limit
  - larger values preserve more of the Corr2Cause prompt but increase memory and runtime
- `RANK`
  - LoRA adapter rank, i.e. adapter size
  - larger rank means more trainable capacity
- `ALPHA`
  - LoRA scaling factor
  - currently kept fixed during the sweep
- `LORA_DROPOUT`
  - dropout applied on the LoRA branch only
  - used as a small regularizer
- `TARGET_MODULES`
  - which Qwen linear layers receive LoRA adapters
  - this repo currently targets `q_proj` and `v_proj`
- `LEARNING_RATE`
  - main optimizer step size
  - this was one of the most important knobs in the 10k sweep
- `WEIGHT_DECAY`
  - regularization on trainable parameters
- `WARMUP_RATIO`
  - fraction of training used for learning-rate warmup
- `LR_SCHEDULER_TYPE`
  - shape of the learning-rate schedule over training
- `MAX_GRAD_NORM`
  - gradient clipping threshold
- `TRAIN_BATCH_SIZE`
  - micro-batch size per device during training
- `GRADIENT_ACCUMULATION_STEPS`
  - number of micro-batches accumulated before an optimizer step
  - `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` gives the effective batch size
- `EVAL_BATCH_SIZE`
  - micro-batch size during evaluation

Practical tuning advice for this repo:
- start with `TRAIN_SUBSET_SIZE`, `RANK`, `LEARNING_RATE`, and `LORA_DROPOUT`
- keep `ALPHA` fixed unless you want a wider follow-up sweep
- if runtime is too slow, change `TRAIN_BATCH_SIZE` and `GRADIENT_ACCUMULATION_STEPS` before touching too many other things
- if memory gets tight, reduce `MAX_LENGTH` or the train micro-batch size

Example override:

```bash
TRAIN_SUBSET_SIZE=30000 RANK=8 LEARNING_RATE=1e-4 python train.py
```

## Formatting

```bash
python -m black .
```

## Notes

- The repo is set up for sequence classification, not chat SFT.
- Validation is used for baseline comparison and model selection.
- Test is reserved for final reporting.

## Next Step

The next useful experiment is a full-data run using the best sweep config:
- `rank=8`
- `alpha=16`
- `learning_rate=1e-4`
- `lora_dropout=0.05`

That run should answer a more important question than the 10k sweep alone:
- how close can LoRA get to the Corr2Cause results reported for full fine-tuning?
- try on LLama model for a better comparison to the original paper. 

The goal of the next stage is to document whether, on this task, a lightweight LoRA adaptation on Qwen is competitive with a much heavier full fine-tuning setup.
