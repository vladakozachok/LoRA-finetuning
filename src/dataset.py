from collections.abc import Callable

from datasets import Dataset, load_dataset

from config import MAX_LENGTH


def load_corr2cause_data(split: str = "train") -> Dataset:
    return load_dataset("causal-nlp/corr2cause", split=split)


def preprocess(
    examples: dict[str, list[str] | list[int]],
    tokenizer: Callable[..., dict[str, list[int]]],
    max_length: int = MAX_LENGTH,
) -> dict[str, list[int]]:
    tokenized = tokenizer(
        examples["input"],
        truncation=True,
        max_length=max_length,
    )
    tokenized["labels"] = examples["label"]
    return tokenized


def map_dataset(
    dataset: Dataset,
    tokenizer: Callable[..., dict[str, list[int]]],
    max_length: int = MAX_LENGTH,
) -> Dataset:
    return dataset.map(
        lambda examples: preprocess(examples, tokenizer, max_length=max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )


def select_subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None:
        return dataset

    subset_size = min(size, len(dataset))
    return dataset.shuffle(seed=seed).select(range(subset_size))


def set_torch_format(dataset: Dataset) -> Dataset:
    return dataset.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
