from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from tn_dl.config import ExperimentConfig
from tn_dl.data import build_dataloaders
from tn_dl.models import build_model
from tn_dl.training.loss_tracker import LossTracker
from tn_dl.utils.device import resolve_device
from tn_dl.utils.io import create_run_dir, write_yaml_file
from tn_dl.utils.seed import seed_everything

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class TrainingArtifacts:
    run_dir: Path
    best_checkpoint: Path
    metrics_path: Path
    loss_plot_path: Path
    resolved_config_path: Path


@dataclass(slots=True)
class EvalMetrics:
    loss: float
    accuracy: float


def train(config: ExperimentConfig) -> TrainingArtifacts:
    seed_everything(config.seed)
    logger.info("Seed fijada: %s", config.seed)

    device = resolve_device(config.device)
    logger.info("Device seleccionado: %s", device)

    logger.info(
        "Cargando dataset '%s' desde %s...",
        config.data.dataset,
        config.data.data_dir,
    )
    dataloaders = build_dataloaders(config.data)
    logger.info(
        "DataLoaders listos: train_batches = %s, eval_batches = %s",
        len(dataloaders.train_loader),
        len(dataloaders.eval_loader),
    )

    logger.info(
        "Construyendo el modelo: embedding = %s, layer = %s, bond_dim = %s",
        config.model.embedding_name,
        config.model.layer_name,
        config.model.bond_dim,
    )
    model = build_model(config.model).to(device)

    logger.info("Inicializando modelo con dummy batch...")
    _prime_model(model, config, device)
    logger.info("Modelo listo")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = _build_scheduler(
        optimizer = optimizer,
        config = config,
    )

    criterion = nn.CrossEntropyLoss()
    run_dir = _prepare_run_dir(config)
    checkpoint_path = run_dir / config.io.checkpoint_name
    metrics_path = run_dir / config.io.metrics_name
    loss_plot_path = run_dir / "loss_curve.png"
    resolved_config_path = run_dir / config.io.resolved_config_name

    logger.info("Run directory: %s", run_dir)
    logger.info("Metrics path: %s", metrics_path)
    logger.info("Checkpoint path: %s", checkpoint_path)

    loss_tracker = LossTracker(
        metrics_path = metrics_path,
        plot_path= loss_plot_path,
    )
    write_yaml_file(resolved_config_path, asdict(config))

    start_epoch = 0
    best_accuracy = float("-inf")
    if config.training.resume_checkpoint is not None:
        start_epoch, best_accuracy = _load_checkpoint(
            checkpoint_path=config.training.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler = scheduler,
            device=device,
        )

    for epoch in range(start_epoch, config.training.epochs):
        logger.info(
            "Epoch %s/%s: entrenamiento iniciado",
            epoch + 1,
            config.training.epochs,
        )

        train_loss = _run_training_epoch(
            model=model,
            dataloader=dataloaders.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_every_n_steps = config.training.log_every_n_steps,
        )
        eval_metrics = _run_evaluation(
            model=model,
            dataloader=dataloaders.eval_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step(eval_metrics.loss)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != current_lr:
                logger.info(
                    "Scheduler reduce learning rate: %.8f -> %.8f",
                    current_lr,
                    new_lr
                )
        
        logger.info(
            "Epoch %s/%s finalizada: train_loss = %.6f, eval_loss = %.6f, eval_accuracy = %.4f",
            epoch + 1,
            config.training.epochs,
            train_loss,
            eval_metrics.loss,
            eval_metrics.accuracy,
        )

        loss_tracker.record_epoch(
            epoch = epoch + 1,
            train_loss = train_loss,
            eval_loss = eval_metrics.loss,
            eval_accuracy = eval_metrics.accuracy,
        )

        if eval_metrics.accuracy >= best_accuracy:
            best_accuracy = eval_metrics.accuracy
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler = scheduler,
                epoch=epoch + 1,
                best_accuracy=best_accuracy,
            )
            logger.info(
                "Nuevo mejor checkpoint guardado con eval_accuracy = %.4f",
                best_accuracy,
            )
            
    return TrainingArtifacts(
        run_dir=run_dir,
        best_checkpoint=checkpoint_path,
        metrics_path=metrics_path,
        loss_plot_path=loss_plot_path,
        resolved_config_path=resolved_config_path,
    )


def evaluate(config: ExperimentConfig, checkpoint_path: Path) -> EvalMetrics:
    device = resolve_device(config.device)
    dataloaders = build_dataloaders(config.data)
    model = build_model(config.model).to(device)
    _prime_model(model, config, device)
    _load_checkpoint(checkpoint_path=checkpoint_path,
                     model=model,
                     optimizer=None,
                     scheduler=None,
                     device=device
                )
    criterion = nn.CrossEntropyLoss()
    return _run_evaluation(
        model=model,
        dataloader=dataloaders.eval_loader,
        criterion=criterion,
        device=device,
    )

def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        config: ExperimentConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """
    Construye el scheduler del learning rate si la config lo pide.

    De momento usamos ReduceLROnPlateau porque el criterio natural aquí es eval_loss: si la perdida
    de evaluación deja de mejorar, reducimos el paso del optimiador.

    Esto encaja especialmente bien con Flowers102, donde el problema observado no era solo aprender
    train, sino controlar la generalización.
    """
    scheduler_name = config.training.scheduler_name

    if scheduler_name is None:
        return None
    
    if scheduler_name != "reduce_on_plateau":
        raise ValueError(
            "Unsupported scheduler_name. Use None or 'reduce_on_plateau'."
        )
    
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = "min",
        factor = config.training.scheduler_factor,
        patience = config.training.scheduler_patience,
        min_lr = config.training.scheduler_min_lr,
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
    log_every_n_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    num_batches = len(dataloader)
    for step, (images, labels) in enumerate(dataloader, start = 1):
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
        if log_every_n_steps > 0 and (
            step == 1 or step % log_every_n_steps == 0 or step == num_batches
        ):
            running_loss = total_loss / max(total_items, 1)
            logger.info(
                "Batch %s/%s: running_loss = %.6f",
                step,
                num_batches,
                running_loss,
            )
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
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    epoch: int,
    best_accuracy: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_accuracy": best_accuracy,
        "model_state_dict": model.state_dict(),
        "ordered_parameters": [parameter.detach().cpu() for parameter in model.parameters()],
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(checkpoint, checkpoint_path)


def _load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    device: torch.device,
) -> tuple[int, float]:
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        _load_parameters_in_order(model, checkpoint["ordered_parameters"], device)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

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
