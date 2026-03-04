import json
import os
import subprocess
import sys
from pathlib import Path

BASE_ENV = {
    "MODEL_NAME": "Qwen/Qwen2.5-1.5B-Instruct",
    "TRAIN_SUBSET_SIZE": "10000",
    "VALIDATION_SUBSET_SIZE": "",
    "TEST_SUBSET_SIZE": "",
    "MAX_LENGTH": "512",
    "NUM_EPOCHS": "1",
    "TRAIN_BATCH_SIZE": "2",
    "GRADIENT_ACCUMULATION_STEPS": "8",
    "EVAL_BATCH_SIZE": "4",
    "WEIGHT_DECAY": "0.01",
    "WARMUP_RATIO": "0.03",
    "LR_SCHEDULER_TYPE": "linear",
    "MAX_GRAD_NORM": "1.0",
    "LORA_DROPOUT": "0.05",
    "TARGET_MODULES": "q_proj,v_proj",
    "WANDB_LOG_MODEL": "false",
}

EXPERIMENTS = [
    {
        "name": "qwen25-r4-lr2e4-10k",
        "overrides": {
            "RANK": "4",
            "ALPHA": "16",
            "LEARNING_RATE": "2e-4",
        },
    },
    {
        "name": "qwen25-r8-lr2e4-10k",
        "overrides": {
            "RANK": "8",
            "ALPHA": "16",
            "LEARNING_RATE": "2e-4",
        },
    },
    {
        "name": "qwen25-r4-lr1e4-10k",
        "overrides": {
            "RANK": "4",
            "ALPHA": "16",
            "LEARNING_RATE": "1e-4",
        },
    },
    {
        "name": "qwen25-r8-lr1e4-10k",
        "overrides": {
            "RANK": "8",
            "ALPHA": "16",
            "LEARNING_RATE": "1e-4",
        },
    },
    {
        "name": "qwen25-r4-lr2e4-drop0-10k",
        "overrides": {
            "RANK": "4",
            "ALPHA": "16",
            "LEARNING_RATE": "2e-4",
            "LORA_DROPOUT": "0.0",
        },
    },
]


def run_experiment(name: str, overrides: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(overrides)
    env["WANDB_RUN_NAME"] = name
    env["OUTPUT_DIR"] = f"./results/{name}"

    print(f"\n=== Running {name} ===", flush=True)
    subprocess.run([sys.executable, "train.py"], check=True, env=env)

    summary_path = Path(env["OUTPUT_DIR"]) / "run_summary.json"
    with summary_path.open() as handle:
        return json.load(handle)


def pick_winner(results: list[dict]) -> dict:
    best = None
    for result in results:
        if best is None:
            best = result
            continue

        best_f1 = float(best["final_eval"]["final_eval_f1"])
        current_f1 = float(result["final_eval"]["final_eval_f1"])

        if current_f1 > best_f1 + 0.01:
            best = result
            continue
        if best_f1 > current_f1 + 0.01:
            continue

        best_loss = float(best["final_eval"]["final_eval_loss"])
        current_loss = float(result["final_eval"]["final_eval_loss"])
        if current_loss < best_loss:
            best = result
            continue
        if best_loss < current_loss:
            continue

        if int(result["config"]["rank"]) < int(best["config"]["rank"]):
            best = result

    return best


def write_sweep_summary(results: list[dict], winner: dict) -> None:
    output = {
        "experiments": results,
        "winner": winner,
    }
    output_path = Path("./results/sweep_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nSaved sweep summary to {output_path}", flush=True)


def main() -> None:
    results = []
    for experiment in EXPERIMENTS:
        summary = run_experiment(experiment["name"], experiment["overrides"])
        results.append(summary)

    winner = pick_winner(results)
    write_sweep_summary(results, winner)


if __name__ == "__main__":
    main()
