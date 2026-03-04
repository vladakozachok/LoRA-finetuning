import os


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

MODEL_NAME = _get_str("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
ID2LABEL = {
    0: "not_entails",
    1: "entails",
}

TRAIN_SUBSET_SIZE = _get_int("TRAIN_SUBSET_SIZE", 10_000)
VALIDATION_SUBSET_SIZE = _get_int("VALIDATION_SUBSET_SIZE", None)
TEST_SUBSET_SIZE = _get_int("TEST_SUBSET_SIZE", None)

MAX_LENGTH = _get_int("MAX_LENGTH", 512)

RANK = _get_int("RANK", 4)
ALPHA = _get_int("ALPHA", 16)
LORA_DROPOUT = _get_float("LORA_DROPOUT", 0.05)
TARGET_MODULES = _get_tuple("TARGET_MODULES", ("q_proj", "v_proj"))
LEARNING_RATE = _get_float("LEARNING_RATE", 2e-4)
WEIGHT_DECAY = _get_float("WEIGHT_DECAY", 0.01)
WARMUP_RATIO = _get_float("WARMUP_RATIO", 0.03)
LR_SCHEDULER_TYPE = _get_str("LR_SCHEDULER_TYPE", "linear")
MAX_GRAD_NORM = _get_float("MAX_GRAD_NORM", 1.0)
NUM_EPOCHS = _get_int("NUM_EPOCHS", 1)
TRAIN_BATCH_SIZE = _get_int("TRAIN_BATCH_SIZE", 2)
EVAL_BATCH_SIZE = _get_int("EVAL_BATCH_SIZE", 4)
GRADIENT_ACCUMULATION_STEPS = _get_int("GRADIENT_ACCUMULATION_STEPS", 8)
BF16 = _get_bool("BF16", True)
DATALOADER_PIN_MEMORY = _get_bool("DATALOADER_PIN_MEMORY", True)
LOGGING_STEPS = _get_int("LOGGING_STEPS", 10)
SEED = _get_int("SEED", 42)

ENABLE_WANDB = _get_bool("ENABLE_WANDB", True)
WANDB_ENTITY = _get_str("WANDB_ENTITY", "vladak")
WANDB_PROJECT = _get_str("WANDB_PROJECT", "lora-finetuning")
WANDB_RUN_NAME = _get_str("WANDB_RUN_NAME", None)
WANDB_WATCH = _get_str("WANDB_WATCH", "false")
WANDB_LOG_MODEL = _get_str("WANDB_LOG_MODEL", "checkpoint")
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
    "final_eval_loss",
    "final_eval_accuracy",
    "final_eval_f1",
    "test_loss",
    "test_accuracy",
    "test_f1",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
)
