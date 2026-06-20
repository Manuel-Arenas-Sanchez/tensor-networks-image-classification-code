from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


class LossTracker:
    """
    Calculamos la perdida del entrenamiento y la evaluación en cada época.

    Esta clase es idenpendiente de la arquitectura del modelo.
    Recive metricas escalares que ya han sido computadas en el bucle de entrenamiento.

    Archivos que genera:
        - metrics.csv: tabla con una fila por cada época
        - loss_curve.png: representación de la perdida entre train/eval en cada época.
    
    Esto es útil para el TFG, porque podremos sacar información de nuestro entrenamiento.
    """

    def __init__(self, metrics_path: Path, plot_path: Path) -> None:
        self.metrics_path = metrics_path
        self.plot_path = plot_path
        self.rows: list[dict[str, float | int]] = []

    def record_epoch(
            self,
            *,
            epoch: int,
            train_loss: float,
            train_accuracy: float,
            eval_loss: float,
            eval_accuracy: float,
            eval_top5_accuracy: float,
            mean_confidence_correct: float,
            mean_confidence_wrong: float,
            learning_rate: float,
    ) -> None:
        """
        Guarda una época de la metrica e inmediatamente guarda los resultados.

        Args:
            - epoch: número de época, empieza en 1.
            - train_loss: la media de la perdida en todo el set de entrenamiento.
            - eval_loss: Media de la perdidas en todo el set de evaluación.
            - eval_accuracy: Porcentaje de aciertos de la parte de evaluación en cada época.
        """

        self.rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "eval_top5_accuracy": eval_top5_accuracy,
                "mean_confidence_correct": mean_confidence_correct,
                "mean_confidence_wrong": mean_confidence_wrong,
                "learning_rate": learning_rate,
            }
        )

        self.write_csv()
        self.write_loss_plot()
        self.write_accuracy_plot()

    def write_csv(self) -> None:
        """
        Escribe las metricas en el CSV.
        """
        if not self.rows:
            self.metrics_path.write_text("", encoding="utf-8")
            return
        
        fieldnames = list(self.rows[0].keys())
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames = fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
    
    def write_loss_plot(self) -> None:
        """
        Crea una gráfica en PNG con las curvas de la perdida en entrenamiento y la perdida en
        evaluación.
        """

        if not self.rows:
            return
        
        epochs = [int(row["epoch"]) for row in self.rows]
        train_losses = [float(row["train_loss"]) for row in self.rows]
        eval_losses = [float(row["eval_loss"]) for row in self.rows]

        plt.figure(figsize = (8, 5))
        plt.plot(epochs, train_losses, marker = "o", label = "Train loss")
        plt.plot(epochs, eval_losses, marker = "o", label = "Eval loss")

        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("Training and validation loss")
        plt.grid(True, alpha = 0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi = 300)
        plt.close()

    def write_accuracy_plot(self) -> None:
        """
        Crea una gráfica en PNG con las curvas de accuracy en entrenamiento y en
        evaluación por época (la curva que crece como una exponencial hacia arriba).

        Es la learning curve clásica, útil para el TFG y para comparar modelos.
        """

        if not self.rows:
            return

        epochs = [int(row["epoch"]) for row in self.rows]
        train_accuracies = [float(row["train_accuracy"]) for row in self.rows]
        eval_accuracies = [float(row["eval_accuracy"]) for row in self.rows]

        plt.figure(figsize = (8, 5))
        plt.plot(epochs, train_accuracies, marker = "o", label = "Train accuracy")
        plt.plot(epochs, eval_accuracies, marker = "o", label = "Eval accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and validation accuracy")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha = 0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path.with_name("accuracy_curve.png"), dpi = 300)
        plt.close()