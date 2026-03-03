from pathlib import Path

from transformers import TrainerCallback


class WandbMetricsCallback(TrainerCallback):
    def __init__(self, metrics: tuple[str, ...], run_name: str, log_model: str = "false") -> None:
        self.metrics = set(metrics)
        self.run_name = run_name
        self.log_model = log_model

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        import wandb

        filtered_logs = {key: value for key, value in logs.items() if key in self.metrics}
        if filtered_logs:
            wandb.log(filtered_logs, step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if self.log_model != "checkpoint":
            return

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return

        import wandb

        artifact = wandb.Artifact(
            name=f"{self.run_name}-checkpoint-{state.global_step}",
            type="model",
        )
        artifact.add_dir(str(checkpoint_dir))
        wandb.log_artifact(artifact)


FilteredWandbCallback = WandbMetricsCallback
