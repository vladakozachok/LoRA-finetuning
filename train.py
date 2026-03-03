import logging
import os
from importlib.util import find_spec

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import (
    ALPHA,
    BF16,
    DATALOADER_PIN_MEMORY,
    ENABLE_WANDB,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    ID2LABEL,
    LEARNING_RATE,
    LORA_DROPOUT,
    LOGGING_STEPS,
    LR_SCHEDULER_TYPE,
    MAX_LENGTH,
    MAX_GRAD_NORM,
    MODEL_NAME,
    NUM_EPOCHS,
    OUTPUT_DIR,
    RANK,
    SEED,
    TARGET_MODULES,
    TRAIN_BATCH_SIZE,
    WANDB_LOG_MODEL,
    WANDB_METRICS_TO_TRACK,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
    WANDB_WATCH,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.dataset import load_corr2cause_data, map_dataset, set_torch_format
from src.modelling.inject_lora import (
    get_replaced_modules,
    get_trainable_parameter_count,
    inject_lora,
)
from src.logger import configure_logging, get_logger
from src.metrics import compute_metrics
from src.wandb_callback import WandbMetricsCallback

logger = get_logger(__name__)


def log_model_sanity_checks(model) -> None:
    replaced_modules = get_replaced_modules(model)
    trainable_parameters = get_trainable_parameter_count(model)
    total_parameters = sum(param.numel() for param in model.parameters())
    trainable_parameter_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    logger.info("LoRA replaced modules (%s):", len(replaced_modules))
    for module_name in replaced_modules:
        logger.info("  - %s", module_name)

    logger.info(
        "Trainable parameters: "
        f"{trainable_parameters:,} / {total_parameters:,} "
        f"({trainable_parameters / total_parameters:.2%})"
    )
    logger.info("Trainable parameter names:")
    for parameter_name in trainable_parameter_names:
        logger.info("  - %s", parameter_name)


def configure_wandb() -> tuple[list[str], str | None, list[WandbMetricsCallback]]:
    if not ENABLE_WANDB:
        return [], None, []

    if find_spec("wandb") is None:
        raise ImportError(
            "ENABLE_WANDB is True, but wandb is not installed. Install it with `pip install wandb`."
        )

    import wandb

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_WATCH"] = WANDB_WATCH
    os.environ["WANDB_LOG_MODEL"] = WANDB_LOG_MODEL

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model_name": MODEL_NAME,
            "rank": RANK,
            "alpha": ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": list(TARGET_MODULES),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "lr_scheduler_type": LR_SCHEDULER_TYPE,
            "max_grad_norm": MAX_GRAD_NORM,
            "num_epochs": NUM_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_length": MAX_LENGTH,
        },
    )
    wandb.define_metric("baseline_eval_loss", summary="min")
    wandb.define_metric("baseline_eval_accuracy", summary="max")
    wandb.define_metric("baseline_eval_f1", summary="max")
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("eval_loss", summary="min")
    wandb.define_metric("eval_accuracy", summary="max")
    wandb.define_metric("eval_f1", summary="max")
    wandb.define_metric("test_loss", summary="min")
    wandb.define_metric("test_accuracy", summary="max")
    wandb.define_metric("test_f1", summary="max")

    logger.info("Weights & Biases enabled for project '%s'", WANDB_PROJECT)
    logger.info("W&B metrics to track: %s", ", ".join(WANDB_METRICS_TO_TRACK))
    return (
        [],
        WANDB_RUN_NAME,
        [
            WandbMetricsCallback(
                metrics=WANDB_METRICS_TO_TRACK,
                run_name=WANDB_RUN_NAME,
                log_model=WANDB_LOG_MODEL,
            )
        ],
    )


def log_metrics(metrics: dict[str, float], header: str) -> None:
    logger.info("%s", header)
    for key, value in metrics.items():
        logger.info("  - %s: %s", key, value)


def main() -> None:
    configure_logging(logging.INFO)
    report_to, run_name, trainer_callbacks = configure_wandb()
    logger.info("Model: %s", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id={label: idx for idx, label in ID2LABEL.items()},
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = inject_lora(
        model,
        rank=RANK,
        alpha=ALPHA,
        dropout=LORA_DROPOUT,
        targets=TARGET_MODULES,
        freeze_base=True,
    )
    log_model_sanity_checks(model)

    raw_train_dataset = load_corr2cause_data(split="train")
    raw_validation_dataset = load_corr2cause_data(split="validation")
    raw_test_dataset = load_corr2cause_data(split="test")

    logger.info("Raw train columns: %s", raw_train_dataset.column_names)
    logger.info("Raw validation columns: %s", raw_validation_dataset.column_names)
    logger.info("Raw test columns: %s", raw_test_dataset.column_names)
    logger.info("Raw train size: %s", len(raw_train_dataset))
    logger.info("Raw validation size: %s", len(raw_validation_dataset))
    logger.info("Raw test size: %s", len(raw_test_dataset))

    train_dataset = set_torch_format(
        map_dataset(raw_train_dataset, tokenizer, max_length=MAX_LENGTH)
    )
    validation_dataset = set_torch_format(
        map_dataset(raw_validation_dataset, tokenizer, max_length=MAX_LENGTH)
    )
    test_dataset = set_torch_format(map_dataset(raw_test_dataset, tokenizer, max_length=MAX_LENGTH))

    logger.info("Tokenized train columns: %s", train_dataset.column_names)
    logger.info("Tokenized validation columns: %s", validation_dataset.column_names)
    logger.info("Tokenized test columns: %s", test_dataset.column_names)
    logger.info("Max sequence length: %s", MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        max_grad_norm=MAX_GRAD_NORM,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        bf16=BF16,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        report_to=report_to,
        run_name=run_name,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=trainer_callbacks,
    )

    logger.info("Running baseline validation evaluation before training")
    baseline_metrics = trainer.evaluate(
        eval_dataset=validation_dataset,
        metric_key_prefix="baseline_eval",
    )
    log_metrics(baseline_metrics, "Baseline validation metrics")

    logger.info("Starting training")
    trainer.train()

    logger.info("Running final test evaluation")
    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )
    log_metrics(test_metrics, "Final test metrics")

    if ENABLE_WANDB:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
