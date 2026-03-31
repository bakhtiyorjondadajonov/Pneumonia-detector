"""Multi-label evaluation with per-class AUC-ROC, threshold optimization, and plots."""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from fastai.vision.all import load_learner
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data import load_config
from src.data_multilabel import (
    CLASSES,
    load_nih_metadata,
    create_multilabel_test_dl,
)


def get_multilabel_predictions(learn, test_dl) -> tuple:
    """Get multi-label predictions on the test set.

    Args:
        learn: Trained multi-label Learner.
        test_dl: Test DataLoader.

    Returns:
        Tuple of (y_prob [N, 14]) as numpy array (sigmoid probabilities).
    """
    preds, _ = learn.get_preds(dl=test_dl)
    y_prob = preds.sigmoid().numpy()
    return y_prob


def get_test_labels(test_df, classes: list[str] = CLASSES) -> np.ndarray:
    """Extract multi-hot ground truth labels from test DataFrame.

    Args:
        test_df: Test DataFrame with 'labels' column.
        classes: List of class names.

    Returns:
        Binary array of shape (N, num_classes).
    """
    n = len(test_df)
    y_true = np.zeros((n, len(classes)), dtype=np.float32)

    for i, labels_str in enumerate(test_df["labels"].values):
        if labels_str:
            for label in labels_str.split(";"):
                if label in classes:
                    y_true[i, classes.index(label)] = 1.0

    return y_true


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str] = CLASSES,
    metric: str = "f1",
) -> dict:
    """Optimize per-class decision thresholds.

    Sweeps thresholds from 0.0 to 1.0 for each class independently
    and picks the threshold maximizing the target metric.

    Args:
        y_true: Ground truth binary array (N, num_classes).
        y_prob: Predicted probabilities (N, num_classes).
        classes: List of class names.
        metric: Metric to optimize ("f1").

    Returns:
        Dict mapping class name to optimal threshold.
    """
    thresholds = {}
    print("\nOptimizing per-class thresholds:")

    for i, cls in enumerate(classes):
        best_thresh = 0.5
        best_score = 0.0

        for t in np.arange(0.05, 0.95, 0.01):
            y_pred = (y_prob[:, i] >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_true[:, i], y_pred, zero_division=0)
            else:
                score = f1_score(y_true[:, i], y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_thresh = round(float(t), 2)

        thresholds[cls] = best_thresh
        print(f"  {cls:<22s}: threshold={best_thresh:.2f}, F1={best_score:.4f}")

    return thresholds


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict,
    classes: list[str] = CLASSES,
) -> dict:
    """Compute comprehensive multi-label metrics.

    Args:
        y_true: Ground truth (N, 14).
        y_prob: Predicted probabilities (N, 14).
        thresholds: Per-class optimal thresholds.
        classes: List of class names.

    Returns:
        Dict with per-class and aggregate metrics.
    """
    metrics = {"per_class": {}, "macro": {}}

    per_class_auc = []

    for i, cls in enumerate(classes):
        t = thresholds.get(cls, 0.5)
        y_pred = (y_prob[:, i] >= t).astype(int)

        # AUC-ROC (only if both classes present)
        if len(np.unique(y_true[:, i])) == 2:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        else:
            auc = 0.0

        per_class_auc.append(auc)

        metrics["per_class"][cls] = {
            "auc_roc": float(auc),
            "precision": float(precision_score(y_true[:, i], y_pred, zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true[:, i], y_pred, zero_division=0)),
            "threshold": t,
            "n_positive": int(y_true[:, i].sum()),
            "n_total": int(len(y_true)),
            "prevalence": float(y_true[:, i].mean()),
        }

    # Macro averages
    metrics["macro"]["auc_roc"] = float(np.mean(per_class_auc))
    metrics["macro"]["precision"] = float(np.mean(
        [m["precision"] for m in metrics["per_class"].values()]
    ))
    metrics["macro"]["recall"] = float(np.mean(
        [m["recall"] for m in metrics["per_class"].values()]
    ))
    metrics["macro"]["f1_score"] = float(np.mean(
        [m["f1_score"] for m in metrics["per_class"].values()]
    ))

    return metrics


# ─── Plotting Functions ────────────────────────────────────────────────────────

def plot_per_class_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str] = CLASSES,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC curves for all 14 diseases, overlaid."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))

    for i, (cls, color) in enumerate(zip(classes, colors)):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{cls} ({auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-Class ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_per_class_auc_bar_chart(
    metrics: dict,
    classes: list[str] = CLASSES,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of AUC-ROC per disease."""
    aucs = [metrics["per_class"][cls]["auc_roc"] for cls in classes]
    sorted_idx = np.argsort(aucs)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#4CAF50" if a >= 0.8 else "#FF9800" if a >= 0.7 else "#f44336" for a in np.array(aucs)[sorted_idx]]

    bars = ax.barh(
        [classes[i] for i in sorted_idx],
        [aucs[i] for i in sorted_idx],
        color=colors,
        alpha=0.85,
    )

    for bar, auc_val in zip(bars, np.array(aucs)[sorted_idx]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{auc_val:.3f}", va="center", fontsize=9, fontweight="bold")

    macro_auc = metrics["macro"]["auc_roc"]
    ax.axvline(x=macro_auc, color="red", linestyle="--", lw=1.5, label=f"Macro Avg: {macro_auc:.3f}")

    ax.set_xlim([0, 1.05])
    ax.set_xlabel("AUC-ROC", fontsize=12)
    ax.set_title("Per-Class AUC-ROC", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_per_class_confusion_matrices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict,
    classes: list[str] = CLASSES,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a grid of 14 small confusion matrices (one per disease)."""
    n_classes = len(classes)
    rows, cols = 4, 4  # 4x4 grid for 14 diseases (2 empty)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 14))

    for i, cls in enumerate(classes):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        t = thresholds.get(cls, 0.5)
        y_pred = (y_prob[:, i] >= t).astype(int)
        cm = confusion_matrix(y_true[:, i], y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
            ax=ax, square=True, linewidths=0.5,
            annot_kws={"size": 10},
            cbar=False,
        )
        auc = roc_auc_score(y_true[:, i], y_prob[:, i]) if len(np.unique(y_true[:, i])) == 2 else 0
        ax.set_title(f"{cls}\nAUC: {auc:.3f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide empty cells
    for i in range(n_classes, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle("Per-Class Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_label_cooccurrence(
    y_true: np.ndarray,
    classes: list[str] = CLASSES,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot disease co-occurrence heatmap."""
    cooccurrence = y_true.T @ y_true
    # Normalize by total positives per class
    diag = np.diag(cooccurrence).copy()
    diag[diag == 0] = 1
    cooccurrence_norm = cooccurrence / diag[:, None]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cooccurrence_norm,
        xticklabels=classes,
        yticklabels=classes,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        ax=ax,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title("Disease Co-occurrence Matrix\n(Normalized by row disease count)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_multilabel_metrics_table(
    metrics: dict,
    classes: list[str] = CLASSES,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a table image of all per-class metrics."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")

    headers = ["Disease", "AUC-ROC", "Precision", "Recall", "F1-Score", "Threshold", "Prevalence"]
    rows = []
    for cls in classes:
        m = metrics["per_class"][cls]
        rows.append([
            cls,
            f'{m["auc_roc"]:.4f}',
            f'{m["precision"]:.2%}',
            f'{m["recall"]:.2%}',
            f'{m["f1_score"]:.2%}',
            f'{m["threshold"]:.2f}',
            f'{m["prevalence"]:.1%}',
        ])

    # Add macro average row
    macro = metrics["macro"]
    rows.append([
        "MACRO AVG",
        f'{macro["auc_roc"]:.4f}',
        f'{macro["precision"]:.2%}',
        f'{macro["recall"]:.2%}',
        f'{macro["f1_score"]:.2%}',
        "—",
        "—",
    ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#1a237e")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    # Style rows
    for i in range(len(rows)):
        for j in range(len(headers)):
            cell = table[i + 1, j]
            cell.set_edgecolor("#e0e0e0")
            if i == len(rows) - 1:  # macro avg row
                cell.set_facecolor("#e3f2fd")
                cell.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#f5f5f5")

    ax.set_title("Multi-Label Classification Results — Per-Class Metrics",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_per_class_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str] = CLASSES,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Precision-Recall curves for all 14 diseases, overlaid."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))

    for i, (cls, color) in enumerate(zip(classes, colors)):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        ax.plot(recall, precision, color=color, lw=1.5, label=f"{cls} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Per-Class Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# ─── Main Evaluation Pipeline ─────────────────────────────────────────────────

def evaluate_multilabel_model(
    model_path: str,
    test_df,
    arch_name: str,
    classes: list[str] = CLASSES,
    figures_dir: str = "./outputs/figures",
    metrics_dir: str = "./outputs/metrics",
) -> dict:
    """Run full multi-label evaluation pipeline.

    Args:
        model_path: Path to saved .pkl model.
        test_df: Test DataFrame.
        arch_name: Architecture name.
        classes: List of class names.
        figures_dir: Output directory for plots.
        metrics_dir: Output directory for metrics JSON.

    Returns:
        Dict with all evaluation results.
    """
    fig_path = Path(figures_dir)
    met_path = Path(metrics_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    met_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{arch_name}_multilabel"

    print(f"\nEvaluating {arch_name} (multi-label)...")
    print("-" * 50)

    # Load model and create test DataLoader
    learn = load_learner(model_path)
    test_dl = create_multilabel_test_dl(learn, test_df)

    # Get predictions
    y_prob = get_multilabel_predictions(learn, test_dl)
    y_true = get_test_labels(test_df, classes)

    # Optimize thresholds
    thresholds = optimize_thresholds(y_true, y_prob, classes)

    # Save thresholds
    thresh_path = met_path / f"{prefix}_thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    # Compute metrics
    metrics = compute_multilabel_metrics(y_true, y_prob, thresholds, classes)

    print(f"\n  Macro AUC-ROC: {metrics['macro']['auc_roc']:.4f}")
    print(f"  Macro F1:      {metrics['macro']['f1_score']:.4f}")

    # Generate plots
    plot_per_class_roc_curves(
        y_true, y_prob, classes, save_path=fig_path / f"{prefix}_roc_curves.png"
    )
    plot_per_class_pr_curves(
        y_true, y_prob, classes, save_path=fig_path / f"{prefix}_pr_curves.png"
    )
    plot_per_class_auc_bar_chart(
        metrics, classes, save_path=fig_path / f"{prefix}_auc_bar.png"
    )
    plot_per_class_confusion_matrices(
        y_true, y_prob, thresholds, classes, save_path=fig_path / f"{prefix}_confusion_matrices.png"
    )
    plot_label_cooccurrence(
        y_true, classes, save_path=fig_path / f"{prefix}_cooccurrence.png"
    )
    plot_multilabel_metrics_table(
        metrics, classes, save_path=fig_path / f"{prefix}_metrics_table.png"
    )

    # Save metrics
    results_path = met_path / f"{prefix}_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {results_path}")

    plt.close("all")
    return metrics


def evaluate_all_multilabel(config_path: str = "config/config.yaml"):
    """Evaluate all trained multi-label models."""
    config = load_config(config_path)
    ml_cfg = config["multilabel"]
    train_cfg = ml_cfg["training"]
    output_cfg = config["outputs"]

    # Load test data
    _, test_df = load_nih_metadata(
        csv_path=ml_cfg["csv_path"],
        image_dir=ml_cfg["image_dir"],
        train_list=ml_cfg["train_list"],
        test_list=ml_cfg["test_list"],
        sample_fraction=ml_cfg.get("sample_fraction", 1.0),
    )

    all_results = {}

    for arch_name in train_cfg["architectures"]:
        model_path = Path(output_cfg["models_dir"]) / f"{arch_name}_multilabel.pkl"
        if not model_path.exists():
            print(f"  Skipping {arch_name}: model not found at {model_path}")
            continue

        metrics = evaluate_multilabel_model(
            model_path=str(model_path),
            test_df=test_df,
            arch_name=arch_name,
            figures_dir=output_cfg["figures_dir"],
            metrics_dir=output_cfg["metrics_dir"],
        )
        all_results[arch_name] = metrics

    # Print comparison
    if all_results:
        print(f"\n{'='*80}")
        print("Multi-Label Architecture Comparison (Macro-Averaged)")
        print(f"{'='*80}")
        header = f"{'Architecture':<15} {'AUC-ROC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
        print(header)
        print("-" * 60)
        for name, m in all_results.items():
            macro = m["macro"]
            print(f"{name:<15} {macro['auc_roc']:>10.4f} {macro['precision']:>10.4f} "
                  f"{macro['recall']:>10.4f} {macro['f1_score']:>10.4f}")

    return all_results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    evaluate_all_multilabel(config_path)
