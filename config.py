import os

from dotenv import load_dotenv

load_dotenv()


def _get_str(name: str, default: str | None) -> str | None:
    return os.getenv(name, default)


def _get_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _get_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


OUTPUT_DIR = _get_str("OUTPUT_DIR", "./results")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ID2LABEL = {
    0: "not_entails",
    1: "entails",
}

MAX_LENGTH = 512

RANK = 4
ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ("q_proj", "v_proj")
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "linear"
MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
BF16 = True
DATALOADER_PIN_MEMORY = True
LOGGING_STEPS = 10
SEED = 42

ENABLE_WANDB = True
WANDB_PROJECT = "vladak/LoRA-finetuning"
WANDB_RUN_NAME = "qwen25-1.5b-lora-corr2cause"
WANDB_WATCH = "false"
WANDB_LOG_MODEL = "checkpoint"
WANDB_METRICS_TO_TRACK = (
    "loss",
    "grad_norm",
    "learning_rate",
    "epoch",
    "baseline_eval_loss",
    "baseline_eval_accuracy",
    "baseline_eval_f1",
    "eval_loss",
    "eval_accuracy",
    "eval_f1",
    "test_loss",
    "test_accuracy",
    "test_f1",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
)
