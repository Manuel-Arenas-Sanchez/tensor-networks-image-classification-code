from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from tn_dl.config import ExperimentConfig
from tn_dl.data import build_dataloaders
from tn_dl.models import build_model
from tn_dl.utils.device import resolve_device
from tn_dl.utils.io import create_run_dir, write_metrics_csv, write_yaml_file
from tn_dl.utils.seed import seed_everything


@dataclass(slots=True)
class TrainingArtifacts:
    run_dir: Path
    best_checkpoint: Path
    metrics_path: Path
    resolved_config_path: Path


@dataclass(slots=True)
class EvalMetrics:
    loss: float
    accuracy: float


def train(config: ExperimentConfig) -> TrainingArtifacts:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    dataloaders = build_dataloaders(config.data)
    model = build_model(config.model).to(device)
    _prime_model(model, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    run_dir = _prepare_run_dir(config)
    checkpoint_path = run_dir / config.io.checkpoint_name
    metrics_path = run_dir / config.io.metrics_name
    resolved_config_path = run_dir / config.io.resolved_config_name
    write_yaml_file(resolved_config_path, asdict(config))

    start_epoch = 0
    best_accuracy = float("-inf")
    if config.training.resume_checkpoint is not None:
        start_epoch, best_accuracy = _load_checkpoint(
            checkpoint_path=config.training.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    rows: list[dict[str, float | int]] = []
    for epoch in range(start_epoch, config.training.epochs):
        train_loss = _run_training_epoch(
            model=model,
            dataloader=dataloaders.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        eval_metrics = _run_evaluation(
            model=model,
            dataloader=dataloaders.eval_loader,
            criterion=criterion,
            device=device,
        )
        rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "eval_loss": eval_metrics.loss,
                "eval_accuracy": eval_metrics.accuracy,
            }
        )
        if eval_metrics.accuracy >= best_accuracy:
            best_accuracy = eval_metrics.accuracy
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_accuracy=best_accuracy,
            )

    write_metrics_csv(metrics_path, rows)
    return TrainingArtifacts(
        run_dir=run_dir,
        best_checkpoint=checkpoint_path,
        metrics_path=metrics_path,
        resolved_config_path=resolved_config_path,
    )


def evaluate(config: ExperimentConfig, checkpoint_path: Path) -> EvalMetrics:
    device = resolve_device(config.device)
    dataloaders = build_dataloaders(config.data)
    model = build_model(config.model).to(device)
    _prime_model(model, config, device)
    _load_checkpoint(checkpoint_path=checkpoint_path, model=model, optimizer=None, device=device)
    criterion = nn.CrossEntropyLoss()
    return _run_evaluation(
        model=model,
        dataloader=dataloaders.eval_loader,
        criterion=criterion,
        device=device,
    )


def _prepare_run_dir(config: ExperimentConfig) -> Path:
    if config.training.resume_checkpoint is not None:
        return config.training.resume_checkpoint.parent
    return create_run_dir(config.io.root_dir, config.name)


def _prime_model(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    with torch.no_grad():
        dummy_batch = torch.zeros((1, *config.model.input_shape), device=device)
        _ = model(dummy_batch)


def _run_training_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def _run_evaluation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            total_correct += int((predictions == labels).sum().item())
    return EvalMetrics(
        loss=total_loss / max(total_items, 1),
        accuracy=total_correct / max(total_items, 1),
    )


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_accuracy: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_accuracy": best_accuracy,
        "model_state_dict": model.state_dict(),
        "ordered_parameters": [parameter.detach().cpu() for parameter in model.parameters()],
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)


def _load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        _load_parameters_in_order(model, checkpoint["ordered_parameters"], device)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_accuracy", float("-inf")))


def _load_parameters_in_order(
    model: nn.Module,
    ordered_parameters: list[torch.Tensor],
    device: torch.device,
) -> None:
    parameters = list(model.parameters())
    if len(parameters) != len(ordered_parameters):
        raise RuntimeError("Checkpoint parameter count does not match model parameter count.")
    with torch.no_grad():
        for parameter, value in zip(parameters, ordered_parameters, strict=True):
            parameter.copy_(value.to(device))
