"""Multi-label training pipeline with 2-stage fine-tuning and experiment tracking."""

import json
import sys
import time
from pathlib import Path

import torch
from fastai.vision.all import CSVLogger, SaveModelCallback, valley, slide

from src.data import load_config
from src.data_multilabel import (
    load_nih_metadata,
    compute_class_weights,
    create_multilabel_dataloaders,
    CLASSES,
)
from src.model_multilabel import create_multilabel_learner
from src.model import count_parameters


def train_multilabel_model(
    learn,
    arch_name: str,
    epochs_freeze: int = 3,
    epochs_unfreeze: int = 7,
    weight_decay: float = 0.01,
    save_dir: str = "./outputs/models",
) -> dict:
    """Train a multi-label model using 2-stage fine-tuning.

    Stage 1: Train classifier head with frozen backbone.
    Stage 2: Unfreeze all layers with discriminative learning rates.

    Args:
        learn: FastAI Learner configured for multi-label.
        arch_name: Architecture name (for saving).
        epochs_freeze: Epochs for stage 1.
        epochs_unfreeze: Epochs for stage 2.
        weight_decay: Weight decay.
        save_dir: Directory for model checkpoints.

    Returns:
        Dict with training results.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = {"architecture": arch_name, "task": "multilabel"}

    print(f"\n{'='*60}")
    print(f"Training {arch_name} (Multi-Label, 14 diseases)")
    print(f"{'='*60}")

    # Stage 1: Frozen backbone
    print(f"\n--- Stage 1: Frozen backbone ({epochs_freeze} epochs) ---")
    lr_result = learn.lr_find(suggest_funcs=(valley, slide))
    base_lr = lr_result.valley
    print(f"  LR Finder suggested: {base_lr:.2e}")

    start_time = time.time()

    learn.fit_one_cycle(
        epochs_freeze,
        lr_max=base_lr,
        wd=weight_decay,
    )

    # Stage 2: Unfrozen with discriminative LRs
    print(f"\n--- Stage 2: Unfrozen ({epochs_unfreeze} epochs) ---")
    learn.unfreeze()

    learn.fit_one_cycle(
        epochs_unfreeze,
        lr_max=slice(base_lr / 100, base_lr / 10),
        wd=weight_decay,
    )

    training_time = time.time() - start_time
    results["training_time_seconds"] = round(training_time, 1)
    results["training_time_minutes"] = round(training_time / 60, 1)
    results["total_params"] = count_parameters(learn)

    # Remove CSVLogger before export (closed file handles can't be pickled)
    csv_cbs = [cb for cb in learn.cbs if isinstance(cb, CSVLogger)]
    for cb in csv_cbs:
        learn.remove_cb(cb)

    # Save model
    model_path = save_path / f"{arch_name}_multilabel.pkl"
    learn.export(model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Training time: {training_time / 60:.1f} minutes")

    results["model_path"] = str(model_path)
    return results


def train_all_multilabel(config_path: str = "config/config.yaml"):
    """Train all architectures for multi-label classification.

    Args:
        config_path: Path to config file.
    """
    config = load_config(config_path)
    ml_cfg = config["multilabel"]
    train_cfg = ml_cfg["training"]
    output_cfg = config["outputs"]

    # Load dataset
    sample_fraction = ml_cfg.get("sample_fraction", 1.0)
    if sample_fraction < 1.0:
        print(f"Using {sample_fraction:.0%} of dataset for faster training")
    print("Loading NIH ChestX-ray14 metadata...")
    train_val_df, test_df = load_nih_metadata(
        csv_path=ml_cfg["csv_path"],
        image_dir=ml_cfg["image_dir"],
        train_list=ml_cfg["train_list"],
        test_list=ml_cfg["test_list"],
        valid_pct=ml_cfg["valid_pct"],
        sample_fraction=sample_fraction,
        seed=ml_cfg["seed"],
    )

    # Compute class weights (on training data only)
    train_only = train_val_df[~train_val_df["is_valid"]]
    pos_weight = compute_class_weights(train_only, CLASSES, cap=ml_cfg["pos_weight_cap"])

    all_results = []

    for arch_name in train_cfg["architectures"]:
        print(f"\n{'#'*60}")
        print(f"# Architecture: {arch_name}")
        print(f"{'#'*60}")

        # Create fresh DataLoaders
        dls = create_multilabel_dataloaders(
            train_val_df=train_val_df,
            image_size=ml_cfg["image_size"],
            batch_size=ml_cfg["batch_size"],
            num_workers=ml_cfg["num_workers"],
        )

        # Create learner
        learn = create_multilabel_learner(
            dls,
            arch_name=arch_name,
            num_classes=ml_cfg["num_classes"],
            pos_weight=pos_weight,
            use_fp16=train_cfg.get("use_fp16", True),
        )

        # Optional W&B integration
        wandb_cfg = ml_cfg.get("wandb", {})
        if wandb_cfg.get("enabled", False):
            try:
                import wandb
                from fastai.callback.wandb import WandbCallback

                wandb.init(
                    project=wandb_cfg["project"],
                    name=f"{arch_name}-multilabel",
                    config={"architecture": arch_name, **ml_cfg},
                    reinit=True,
                )
                learn.add_cb(WandbCallback(log_preds=False, log_model=True))
            except ImportError:
                print("  W&B not installed, skipping.")

        # CSV logger
        log_path = Path(output_cfg["metrics_dir"]) / f"{arch_name}_multilabel_history.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        learn.add_cb(CSVLogger(fname=str(log_path)))

        # Train
        results = train_multilabel_model(
            learn,
            arch_name=arch_name,
            epochs_freeze=train_cfg["epochs_freeze"],
            epochs_unfreeze=train_cfg["epochs_unfreeze"],
            weight_decay=train_cfg["weight_decay"],
            save_dir=output_cfg["models_dir"],
        )

        all_results.append(results)

        # Clean up W&B
        if wandb_cfg.get("enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

    # Save training summary
    summary_path = Path(output_cfg["metrics_dir"]) / "multilabel_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTraining summary saved to {summary_path}")

    return all_results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    train_all_multilabel(config_path)
