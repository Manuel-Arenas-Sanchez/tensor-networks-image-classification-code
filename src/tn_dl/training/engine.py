from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

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

SchedulerType: TypeAlias = (
    torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau
)


@dataclass(slots=True)
class TrainingArtifacts:
    run_dir: Path
    best_checkpoint: Path
    metrics_path: Path
    loss_plot_path: Path
    resolved_config_path: Path


@dataclass(slots=True)
class EpochMetrics:
    """
    Resume lo que ha hecho el modelo en una pasada completa por un dataloader.

    Sirve tanto para train como para eval:
        - loss: perdida media
        - accuracy: acierto top-1
        - top5_accuracy: si la clase correcta aparece entre las 5 más probables.
        - mean_confidence_correct: confianza media cuando acierta.
        - mean_confidence_wrong: confianza media cuando falla.

    Las confianzas ayudan a detectar sobreconfianza: si el modelo falla con probabilidad muy alta,
    CrossEntropyLoss suele crecer mucho aunque la accuracy no cambie tanto.
    """

    loss: float
    accuracy: float
    top5_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_confidence_correct: float
    mean_confidence_wrong: float


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
        optimizer=optimizer,
        config=config,
    )

    train_criterion = nn.CrossEntropyLoss(
        label_smoothing=config.training.label_smoothing,
    )
    eval_criterion = nn.CrossEntropyLoss()
    run_dir = _prepare_run_dir(config)
    checkpoint_path = run_dir / config.io.checkpoint_name
    metrics_path = run_dir / config.io.metrics_name
    loss_plot_path = run_dir / "loss_curve.png"
    resolved_config_path = run_dir / config.io.resolved_config_name

    logger.info("Run directory: %s", run_dir)
    logger.info("Metrics path: %s", metrics_path)
    logger.info("Checkpoint path: %s", checkpoint_path)

    loss_tracker = LossTracker(
        metrics_path=metrics_path,
        plot_path=loss_plot_path,
    )
    write_yaml_file(resolved_config_path, asdict(config))

    start_epoch = 0
    best_accuracy = float("-inf")
    if config.training.resume_checkpoint is not None:
        start_epoch, best_accuracy = _load_checkpoint(
            checkpoint_path=config.training.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    for epoch in range(start_epoch, config.training.epochs):
        logger.info(
            "Epoch %s/%s: entrenamiento iniciado",
            epoch + 1,
            config.training.epochs,
        )

        in_warmup = _apply_warmup_lr(
            optimizer=optimizer,
            config=config,
            epoch=epoch,
        )
        epoch_lr = optimizer.param_groups[0]["lr"]

        if in_warmup:
            logger.info(
                "Warmup learning rate: epoch %s/%s, lr = %.8f",
                epoch + 1,
                config.training.warmup_epochs,
                epoch_lr,
            )

        train_metrics = _run_training_epoch(
            model=model,
            dataloader=dataloaders.train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            log_every_n_steps=config.training.log_every_n_steps,
        )
        eval_metrics = _run_evaluation(
            model=model,
            dataloader=dataloaders.eval_loader,
            criterion=eval_criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None and not in_warmup:
            _step_scheduler(
                scheduler=scheduler,
                config=config,
                eval_loss=eval_metrics.loss,
            )
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != current_lr:
                logger.info(
                    "Scheduler reduce learning rate: %.8f -> %.8f",
                    current_lr,
                    new_lr,
                )

        logger.info(
            "Epoch %s/%s finalizada: train_loss = %.6f, train_accuracy = %.4f, "
            "eval_loss = %.6f, eval_accuracy = %.4f, eval_top5_accuracy = %.4f",
            epoch + 1,
            config.training.epochs,
            train_metrics.loss,
            train_metrics.accuracy,
            eval_metrics.loss,
            eval_metrics.accuracy,
            eval_metrics.top5_accuracy,
        )

        loss_tracker.record_epoch(
            epoch=epoch + 1,
            train_loss=train_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            eval_loss=eval_metrics.loss,
            eval_accuracy=eval_metrics.accuracy,
            eval_top5_accuracy=eval_metrics.top5_accuracy,
            train_macro_precision=train_metrics.macro_precision,
            train_macro_recall=train_metrics.macro_recall,
            train_macro_f1=train_metrics.macro_f1,
            eval_macro_precision=eval_metrics.macro_precision,
            eval_macro_recall=eval_metrics.macro_recall,
            eval_macro_f1=eval_metrics.macro_f1,
            mean_confidence_correct=eval_metrics.mean_confidence_correct,
            mean_confidence_wrong=eval_metrics.mean_confidence_wrong,
            learning_rate=epoch_lr,
        )

        if eval_metrics.accuracy >= best_accuracy:
            best_accuracy = eval_metrics.accuracy
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_accuracy=best_accuracy,
            )
            logger.info(
                "Nuevo mejor checkpoint guardado con eval_accuracy = %.4f",
                best_accuracy,
            )
    from tn_dl.evaluation import write_classification_diagnostics

    diagnostics_dir = run_dir / "diagnostics"
    logger.info("Generando diagnósticos por clase en %s", diagnostics_dir)
    _load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
    )
    write_classification_diagnostics(
        model=model,
        dataloader=dataloaders.eval_loader,
        device=device,
        n_classes=config.model.n_classes,
        output_dir=diagnostics_dir,
    )
    logger.info("Diagnósticos guardados.")

    return TrainingArtifacts(
        run_dir=run_dir,
        best_checkpoint=checkpoint_path,
        metrics_path=metrics_path,
        loss_plot_path=loss_plot_path,
        resolved_config_path=resolved_config_path,
    )


def evaluate(config: ExperimentConfig, checkpoint_path: Path) -> EpochMetrics:
    device = resolve_device(config.device)
    dataloaders = build_dataloaders(config.data)
    model = build_model(config.model).to(device)
    _prime_model(model, config, device)
    _load_checkpoint(
        checkpoint_path=checkpoint_path, model=model, optimizer=None, scheduler=None, device=device
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
) -> SchedulerType | None:
    """
    Construye el scheduler del learning rate si la config lo pide.

    Opciones:
        - reduce_on_plateau:
            baja el lr cuando eval_loss deja de mejorar.

        - warmup_cosine:
            después del warmup, baja el lr suavemente con cosine decay desde learning_rate hasta
            scheduler_min_lr.
    """
    scheduler_name = config.training.scheduler_name

    if scheduler_name is None:
        return None

    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.scheduler_min_lr,
        )

    if scheduler_name == "warmup_cosine":
        cosine_epochs = max(
            1,
            config.training.epochs - config.training.warmup_epochs,
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=config.training.scheduler_min_lr,
        )

    raise ValueError(
        "Unsupported scheduler_name. Use None, 'reduce_on_plateau' or 'warmup_cosine'."
    )


def _step_scheduler(
    scheduler: SchedulerType,
    config: ExperimentConfig,
    eval_loss: float,
) -> None:
    if config.training.scheduler_name == "reduce_on_plateau":
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise TypeError("Expected ReduceLROnPlateau scheduler.")
        scheduler.step(eval_loss)
        return

    if config.training.scheduler_name == "warmup_cosine":
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise TypeError("Expected epoch-based LR scheduler.")
        scheduler.step()
        return

    raise ValueError(
        "Unsupported scheduler_name. Use None, 'reduce_on_plateau' or 'warmup_cosine'."
    )


def _apply_warmup_lr(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
) -> bool:
    warmup_epochs = config.training.warmup_epochs

    if warmup_epochs <= 0 or config.training.warmup_name == "none":
        return False
    if epoch < warmup_epochs:
        lr = _compute_warmup_lr(
            base_lr=config.training.learning_rate,
            warmup_name=config.training.warmup_name,
            warmup_start_factor=config.training.warmup_start_factor,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
        )
        _set_optimizer_lr(optimizer, lr)
        return True
    if epoch == warmup_epochs:
        _set_optimizer_lr(optimizer, config.training.learning_rate)

    return False


def _compute_warmup_lr(
    *,
    base_lr: float,
    warmup_name: str,
    warmup_start_factor: float,
    epoch: int,
    warmup_epochs: int,
) -> float:
    if warmup_epochs <= 1:
        return base_lr

    progress = epoch / (warmup_epochs - 1)

    if warmup_name == "linear":
        factor = warmup_start_factor + progress * (1.0 - warmup_start_factor)
        return base_lr * factor

    if warmup_name == "cosine":
        cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        factor = warmup_start_factor + cosine_progress * (1.0 - warmup_start_factor)
        return base_lr * factor

    raise ValueError("Unsupported warmup_name. Use 'linear' or 'cosine'.")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
    config: ExperimentConfig,
    log_every_n_steps: int,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_items = 0
    total_correct = 0
    num_batches = len(dataloader)
    confusion: torch.Tensor | None = None

    for step, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        ce_loss = criterion(logits, labels)
        regularization_loss = _compute_regularization_loss(
            model=model,
            config=config,
            device=device,
        )
        loss = ce_loss + regularization_loss
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += ce_loss.item() * batch_size
        total_items += batch_size
        total_correct += int((predictions == labels).sum().item())
        confusion = _update_confusion_matrix(
            confusion=confusion,
            labels=labels,
            predictions=predictions,
            n_classes=logits.shape[1],
        )

        if log_every_n_steps > 0 and (
            step == 1 or step % log_every_n_steps == 0 or step == num_batches
        ):
            running_loss = total_loss / max(total_items, 1)
            running_accuracy = total_correct / max(total_items, 1)
            logger.info(
                "Batch %s/%s: running_loss = %.6f, running_accuracy = %.4f",
                step,
                num_batches,
                running_loss,
                running_accuracy,
            )

    macro_precision, macro_recall, macro_f1 = _compute_macro_metrics(confusion)

    return EpochMetrics(
        loss=total_loss / max(total_items, 1),
        accuracy=total_correct / max(total_items, 1),
        top5_accuracy=0.0,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        mean_confidence_correct=0.0,
        mean_confidence_wrong=0.0,
    )


def _compute_regularization_loss(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> torch.Tensor:
    regularization_name = config.training.regularization_name
    regularization_lambda = config.training.regularization_lambda

    if regularization_name is None or regularization_lambda == 0.0:
        return torch.zeros((), device=device)

    if regularization_name == "l2":
        penalty = torch.zeros((), device=device)

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1:
                continue
            if "norm" in name.lower():
                continue

            penalty = penalty + parameter.pow(2).sum()

        return regularization_lambda * penalty

    raise ValueError("Unsupported regularization_name. Use None or 'l2'.")


def _run_evaluation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_correct = 0
    total_top5_correct = 0
    confusion: torch.Tensor | None = None

    confidence_correct_sum = 0.0
    confidence_wrong_sum = 0.0
    num_correct = 0
    num_wrong = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long)

            logits = model(images)
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            k = min(5, logits.shape[1])
            top5_predictions = logits.topk(k=k, dim=1).indices
            top5_matches = top5_predictions.eq(labels.unsqueeze(1)).any(dim=1)

            correct_mask = predictions == labels
            wrong_mask = ~correct_mask

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            total_correct += int(correct_mask.sum().item())
            total_top5_correct += int(top5_matches.sum().item())
            confusion = _update_confusion_matrix(
                confusion=confusion,
                labels=labels,
                predictions=predictions,
                n_classes=logits.shape[1],
            )

            if correct_mask.any():
                confidence_correct_sum += float(confidences[correct_mask].sum().item())
                num_correct += int(correct_mask.sum().item())

            if wrong_mask.any():
                confidence_wrong_sum += float(confidences[wrong_mask].sum().item())
                num_wrong += int(wrong_mask.sum().item())

    macro_precision, macro_recall, macro_f1 = _compute_macro_metrics(confusion)

    return EpochMetrics(
        loss=total_loss / max(total_items, 1),
        accuracy=total_correct / max(total_items, 1),
        top5_accuracy=total_top5_correct / max(total_items, 1),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        mean_confidence_correct=confidence_correct_sum / max(num_correct, 1),
        mean_confidence_wrong=confidence_wrong_sum / max(num_wrong, 1),
    )


def _update_confusion_matrix(
    *,
    confusion: torch.Tensor | None,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    if confusion is None:
        confusion = torch.zeros((n_classes, n_classes), dtype=torch.long)

    labels_cpu = labels.detach().cpu().long()
    predictions_cpu = predictions.detach().cpu().long()
    flat_indices = labels_cpu * n_classes + predictions_cpu
    batch_counts = torch.bincount(
        flat_indices,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    confusion += batch_counts

    return confusion


def _compute_macro_metrics(confusion: torch.Tensor | None) -> tuple[float, float, float]:
    if confusion is None:
        return 0.0, 0.0, 0.0

    true_positive = confusion.diag().float()
    total_real = confusion.sum(dim=1).float()
    total_predicted = confusion.sum(dim=0).float()

    precision = true_positive / total_predicted.clamp_min(1.0)
    recall = true_positive / total_real.clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    return (
        float(precision.mean().item()),
        float(recall.mean().item()),
        float(f1.mean().item()),
    )


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: SchedulerType | None,
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
    scheduler: SchedulerType | None,
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
