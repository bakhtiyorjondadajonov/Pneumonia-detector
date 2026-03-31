"""Comprehensive model evaluation with metrics, plots, and architecture comparison."""

import json
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from fastai.vision.all import Learner, load_learner
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data import create_test_dataloader, load_config


def get_predictions(learn: Learner, test_dl) -> tuple:
    """Get model predictions on the test set.

    Args:
        learn: Trained fastai Learner.
        test_dl: Test DataLoader.

    Returns:
        Tuple of (y_true, y_pred, y_prob) as numpy arrays.
    """
    preds, targets = learn.get_preds(dl=test_dl)
    y_prob = preds[:, 1].numpy()  # probability of positive class
    y_pred = preds.argmax(dim=1).numpy()
    y_true = targets.numpy()
    return y_true, y_pred, y_prob


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Dict containing all computed metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], output_dict=True
        ),
    }
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a publication-quality confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved to {save_path}")
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the ROC curve with AUC score.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for positive class.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ROC curve saved to {save_path}")
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the Precision-Recall curve.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for positive class.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#4CAF50", lw=2, label="Precision-Recall Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  PR curve saved to {save_path}")
    return fig


def measure_inference_time(learn: Learner, test_dl, n_runs: int = 50) -> float:
    """Measure average inference time per image in milliseconds.

    Args:
        learn: Trained fastai Learner.
        test_dl: Test DataLoader.
        n_runs: Number of forward passes to average.

    Returns:
        Average inference time in milliseconds.
    """
    learn.model.eval()
    device = next(learn.model.parameters()).device

    # Get a single batch
    batch = next(iter(test_dl))
    x = batch[0][:1].to(device)  # single image

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            learn.model(x)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            learn.model(x)
            times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times))


def evaluate_model(
    model_path: str,
    test_path: str,
    arch_name: str,
    figures_dir: str = "./outputs/figures",
    metrics_dir: str = "./outputs/metrics",
) -> dict:
    """Run full evaluation pipeline on a trained model.

    Args:
        model_path: Path to the saved .pkl model.
        test_path: Path to the test dataset folder.
        arch_name: Architecture name (for file naming).
        figures_dir: Directory to save plots.
        metrics_dir: Directory to save metrics JSON.

    Returns:
        Dict with all evaluation results.
    """
    figures_path = Path(figures_dir)
    metrics_path = Path(metrics_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {arch_name}...")
    print("-" * 40)

    # Load model
    learn = load_learner(model_path)

    # Create test DataLoader
    test_dl, test_files = create_test_dataloader(learn, test_path)

    # Get predictions
    y_true, y_pred, y_prob = get_predictions(learn, test_dl)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    # Measure inference time
    inference_ms = measure_inference_time(learn, test_dl)
    metrics["inference_ms"] = round(inference_ms, 2)
    print(f"  Inference: {inference_ms:.2f} ms/image")

    # Generate plots
    plot_confusion_matrix(
        y_true, y_pred, save_path=figures_path / f"{arch_name}_confusion_matrix.png"
    )
    plot_roc_curve(
        y_true, y_prob, save_path=figures_path / f"{arch_name}_roc_curve.png"
    )
    plot_precision_recall_curve(
        y_true, y_prob, save_path=figures_path / f"{arch_name}_pr_curve.png"
    )

    # Save metrics
    results_path = metrics_path / f"{arch_name}_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {results_path}")

    plt.close("all")
    return metrics


def evaluate_all_architectures(config_path: str = "config/config.yaml"):
    """Evaluate all trained architectures and produce a comparison table.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    output_cfg = config["outputs"]

    all_results = {}

    for arch_name in train_cfg["architectures"]:
        model_path = Path(output_cfg["models_dir"]) / f"{arch_name}_best.pkl"
        if not model_path.exists():
            print(f"  Skipping {arch_name}: model not found at {model_path}")
            continue

        metrics = evaluate_model(
            model_path=str(model_path),
            test_path=eval_cfg["test_path"],
            arch_name=arch_name,
            figures_dir=output_cfg["figures_dir"],
            metrics_dir=output_cfg["metrics_dir"],
        )
        all_results[arch_name] = metrics

    # Print comparison table
    if all_results:
        print(f"\n{'='*80}")
        print("Architecture Comparison")
        print(f"{'='*80}")
        header = f"{'Architecture':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10} {'Infer(ms)':>10}"
        print(header)
        print("-" * 80)
        for name, m in all_results.items():
            row = f"{name:<15} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['auc_roc']:>10.4f} {m['inference_ms']:>10.2f}"
            print(row)

        # Save comparison
        comparison_path = Path(output_cfg["metrics_dir"]) / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nComparison saved to {comparison_path}")

    return all_results


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    evaluate_all_architectures(config_path)
