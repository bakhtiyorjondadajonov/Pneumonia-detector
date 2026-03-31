"""Training pipeline with LR finder, 2-stage fine-tuning, and experiment tracking."""

import json
import sys
import time
from pathlib import Path

import yaml
import torch
from fastai.vision.all import SaveModelCallback, CSVLogger

from src.data import create_dataloaders, load_config
from src.model import create_learner, count_parameters, ARCHITECTURES


def find_learning_rate(learn, show_plot: bool = True) -> float:
    """Run the learning rate finder and return the suggested LR.

    Args:
        learn: FastAI Learner.
        show_plot: Whether to display the LR finder plot.

    Returns:
        Suggested learning rate (valley method).
    """
    lr_result = learn.lr_find(suggest_funcs=(valley, slide))
    if show_plot:
        print(f"  Suggested LR (valley): {lr_result.valley:.2e}")
    return lr_result.valley


def train_model(
    learn,
    arch_name: str,
    epochs_freeze: int = 3,
    epochs_unfreeze: int = 5,
    base_lr: float = None,
    weight_decay: float = 0.01,
    save_dir: str = "./outputs/models",
) -> dict:
    """Train a model using 2-stage fine-tuning.

    Stage 1: Train only the head (classifier) with frozen backbone.
    Stage 2: Unfreeze all layers and train with discriminative learning rates.

    Args:
        learn: FastAI Learner.
        arch_name: Architecture name (for saving).
        epochs_freeze: Epochs for stage 1 (frozen backbone).
        epochs_unfreeze: Epochs for stage 2 (unfrozen).
        base_lr: Base learning rate. If None, uses LR finder.
        weight_decay: Weight decay for regularization.
        save_dir: Directory to save model checkpoints.

    Returns:
        Dict with training results and timing info.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = {"architecture": arch_name}

    # Stage 1: Frozen backbone
    print(f"\n{'='*60}")
    print(f"Training {arch_name}")
    print(f"{'='*60}")

    print(f"\n--- Stage 1: Frozen backbone ({epochs_freeze} epochs) ---")
    if base_lr is None:
        from fastai.vision.all import valley, slide
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

    # Discriminative LRs: lower layers get smaller LR
    learn.fit_one_cycle(
        epochs_unfreeze,
        lr_max=slice(base_lr / 100, base_lr / 10),
        wd=weight_decay,
    )

    training_time = time.time() - start_time
    results["training_time_seconds"] = round(training_time, 1)
    results["total_params"] = count_parameters(learn)

    # Save model
    model_path = save_path / f"{arch_name}_best.pkl"
    learn.export(model_path)
    print(f"\nModel saved to {model_path}")

    results["model_path"] = str(model_path)
    return results


def train_all_architectures(config_path: str = "config/config.yaml"):
    """Train all architectures defined in the config file.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)
    data_cfg = config["data"]
    train_cfg = config["training"]
    output_cfg = config["outputs"]

    all_results = []

    for arch_name in train_cfg["architectures"]:
        print(f"\n{'#'*60}")
        print(f"# Architecture: {arch_name}")
        print(f"{'#'*60}")

        # Create fresh DataLoaders for each architecture
        dls = create_dataloaders(
            data_path=data_cfg["dataset_path"],
            image_size=data_cfg["image_size"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            valid_pct=data_cfg["valid_pct"],
            seed=data_cfg["seed"],
        )

        # Create learner
        learn = create_learner(dls, arch_name=arch_name)

        # Optional W&B integration
        if config.get("wandb", {}).get("enabled", False):
            try:
                import wandb
                from fastai.callback.wandb import WandbCallback

                wandb.init(
                    project=config["wandb"]["project"],
                    name=f"{arch_name}-run",
                    config={
                        "architecture": arch_name,
                        **data_cfg,
                        **train_cfg,
                    },
                    reinit=True,
                )
                learn.add_cb(WandbCallback(log_preds=True, log_model=True))
            except ImportError:
                print("  W&B not installed, skipping experiment tracking.")

        # Add CSV logger
        learn.add_cb(CSVLogger(fname=f"outputs/metrics/{arch_name}_history.csv"))

        # Train
        results = train_model(
            learn,
            arch_name=arch_name,
            epochs_freeze=train_cfg["epochs_freeze"],
            epochs_unfreeze=train_cfg["epochs_unfreeze"],
            weight_decay=train_cfg["weight_decay"],
            save_dir=output_cfg["models_dir"],
        )

        all_results.append(results)

        # Clean up W&B run
        if config.get("wandb", {}).get("enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

    # Save summary
    summary_path = Path(output_cfg["metrics_dir"]) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTraining summary saved to {summary_path}")

    return all_results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    train_all_architectures(config_path)
