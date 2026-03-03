import torch.nn as nn

from .lora_linear import LinearWithLoRA


def is_target_linear(name: str, module: nn.Module, target_modules: tuple[str, ...]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    return name.split(".")[-1] in target_modules


def get_parent_and_name(model: nn.Module, full_name: str) -> tuple[nn.Module, str]:
    parent, _, child_name = full_name.rpartition(".")
    parent = model.get_submodule(parent) if parent else model
    return parent, child_name


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("score."):
            continue
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            for param in module.lora.parameters():
                param.requires_grad = True


def get_replaced_modules(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, LinearWithLoRA)]


def get_trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
    freeze_base: bool = True,
) -> nn.Module:
    modules_to_replace: list[tuple[str, nn.Linear]] = []

    for name, module in model.named_modules():
        if is_target_linear(name, module, targets):
            modules_to_replace.append((name, module))

    for full_name, module in modules_to_replace:
        parent, child_name = get_parent_and_name(model, full_name)
        new_module = LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, new_module)

    if freeze_base:
        freeze_backbone(model)

    return model
