"""Train and evaluate a dynamics surrogate on a dataset saved by collect.py.

Example:
    python train_surrogate.py data/go2_genesis.npz
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from surrogate.collect import load_dataset
from surrogate.surrogate import Surrogate, SurrogateConfig


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Parse training options and validate their ranges."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/genesis_surrogate"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation fraction of the non-test trajectories",
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device; default CUDA if available, else CPU",
    )
    args = parser.parse_args()

    if not 0 < args.test_frac < 1 or not 0 < args.val_frac < 1:
        parser.error("test-frac and val-frac must be between 0 and 1")

    positive_counts = (args.epochs, args.batch_size, args.patience, args.horizon)
    if min(positive_counts) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch-size, patience, horizon and lr must be positive")
    if args.seed < 0:
        parser.error("seed must be nonnegative")

    return parser, args


def validate_dataset(states, actions, next_states, parser) -> None:
    """Require finite, matching trajectories for train/validation/test splits."""
    if (
        states.ndim != 3
        or actions.ndim != 3
        or next_states.shape != states.shape
        or actions.shape[:2] != states.shape[:2]
        or min(states.shape) < 1
        or actions.shape[-1] < 1
    ):
        parser.error(
            "Dataset must contain matching nonempty (sequence, time, feature) arrays"
        )

    if len(states) < 3:
        parser.error(
            "At least three trajectories are needed for train, validation and test"
        )

    if not all(np.isfinite(array).all() for array in (states, actions, next_states)):
        parser.error("Dataset contains NaN or infinite values")


def split_trajectories(num_sequences, test_frac, val_frac, seed):
    """Choose disjoint train, validation, and test trajectories once."""
    order = np.random.default_rng(seed).permutation(num_sequences)
    num_test = min(num_sequences - 2, max(1, int(num_sequences * test_frac)))
    test = order[:num_test]
    training = order[num_test:]

    inner_order = np.random.default_rng(seed + 1).permutation(len(training))
    num_validation = max(1, int(len(training) * val_frac))
    validation = training[inner_order[:num_validation]]
    train = training[inner_order[num_validation:]]

    return train, validation, test


def make_loader(surrogate, arrays, *, shuffle):
    states, actions, next_states = arrays
    inputs = surrogate.x_scaler.transform(np.concatenate([states, actions], axis=1))
    targets = surrogate.y_scaler.transform(next_states)
    tensors = [
        torch.as_tensor(array, dtype=torch.float32) for array in (inputs, targets)
    ]
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("Normalized data exceeds the model's float32 range")
    # Keep data on the CPU and transfer only the current batch to the device.
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors),
        batch_size=surrogate.cfg.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(surrogate.cfg.seed),
    )


def run_epoch(surrogate, loader, optimizer=None) -> float:
    """Compute a sample-weighted loss, optionally updating the model."""
    surrogate.model.train(optimizer is not None)
    total_loss = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, targets in loader:
            inputs = inputs.to(surrogate.device)
            targets = targets.to(surrogate.device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(surrogate.model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Loss became non-finite; check data scaling and learning rate"
                )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(inputs)
    return total_loss / len(loader.dataset)


def fit_model(surrogate, training, validation, *, verbose=True):
    """Fit on explicit partitions and restore the best validation checkpoint."""
    training = tuple(array.reshape(-1, array.shape[-1]) for array in training)
    validation = tuple(array.reshape(-1, array.shape[-1]) for array in validation)
    surrogate._fitted = False
    surrogate.x_scaler.fit(np.concatenate(training[:2], axis=1))
    surrogate.y_scaler.fit(training[2])
    train_loader = make_loader(surrogate, training, shuffle=True)
    validation_loader = make_loader(surrogate, validation, shuffle=False)
    optimizer = torch.optim.Adam(surrogate.model.parameters(), lr=surrogate.cfg.lr)

    history = {"train": [], "val": []}
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    with tqdm(
        total=surrogate.cfg.epochs,
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
        disable=not verbose,
        file=sys.stdout,
    ) as progress:
        for epoch in range(1, surrogate.cfg.epochs + 1):
            train_loss = run_epoch(surrogate, train_loader, optimizer)
            validation_loss = run_epoch(surrogate, validation_loader)
            if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
                raise FloatingPointError(
                    "Training or validation loss became non-finite"
                )
            history["train"].append(train_loss)
            history["val"].append(validation_loss)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(surrogate.model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1

            progress.set_postfix(
                train=f"{train_loss:.5f}",
                validation=f"{validation_loss:.5f}",
                best=f"{best_loss:.5f}",
                refresh=False,
            )
            progress.update(1)
            if stale_epochs >= surrogate.cfg.patience:
                break

    surrogate.model.load_state_dict(best_state)
    surrogate.model.eval()
    surrogate._fitted = True
    summary = {
        "epochs_completed": len(history["train"]),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stopped_early": len(history["train"]) < surrogate.cfg.epochs,
    }
    if verbose:
        status = (
            "Stopped early"
            if summary["stopped_early"]
            else "Training complete"
        )
        print(f"{status} after {len(history['train'])} epochs.")
        print(f"Restored epoch {best_epoch}: validation MSE {best_loss:.6f}.")
    return history, summary


def rollout_error_curve(prediction, actual, scale, active_features):
    """Calculate error by horizon, excluding constant training features."""
    if not active_features.any():
        return [None] * actual.shape[1]

    error = (
        prediction[..., active_features] - actual[..., active_features]
    ) / scale[active_features]
    return np.sqrt(np.mean(error**2, axis=(0, 2))).tolist()


def predict_batches(model, states, actions):
    """Bound device memory use when evaluating a large held-out dataset."""
    predictions = []
    for start in range(0, len(states), model.cfg.batch_size):
        stop = start + model.cfg.batch_size
        predictions.append(model.predict(states[start:stop], actions[start:stop]))
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise FloatingPointError("One-step predictions became non-finite")
    return result


def evaluate_model(model, states, actions, next_states, train, test, horizon):
    """Compare held-out predictions with a persistence baseline."""
    horizon = min(horizon, states.shape[1])
    actual = next_states[test, :horizon]
    trajectories = tqdm(
        test,
        desc="Evaluating rollouts",
        unit="trajectory",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    rollout = np.stack(
        [
            model.rollout(states[index, 0], actions[index, :horizon])
            for index in trajectories
        ]
    )
    if not np.isfinite(rollout).all():
        raise FloatingPointError("Rollout predictions became non-finite")

    # Persistence keeps the initial state throughout each open-loop rollout.
    baseline = np.broadcast_to(states[test, :1], actual.shape)
    one_step = predict_batches(
        model,
        states[test].reshape(-1, states.shape[-1]),
        actions[test].reshape(-1, actions.shape[-1]),
    ).reshape(next_states[test].shape)

    scale = np.ptp(states[train].reshape(-1, states.shape[-1]), axis=0)
    active_features = scale > 1e-8
    one_step_error = one_step - next_states[test]
    persistence_error = states[test] - next_states[test]

    metrics = {
        "one_step_rmse_per_feature": np.sqrt(
            np.mean(one_step_error**2, axis=(0, 1))
        ).tolist(),
        "persistence_one_step_rmse_per_feature": np.sqrt(
            np.mean(persistence_error**2, axis=(0, 1))
        ).tolist(),
        "rollout_nrmse_by_step": rollout_error_curve(
            rollout, actual, scale, active_features
        ),
        "persistence_rollout_nrmse_by_step": rollout_error_curve(
            baseline, actual, scale, active_features
        ),
        "normalization": (
            "Training state range per feature; "
            "constant training features excluded from rollout NRMSE"
        ),
        "active_features": np.flatnonzero(active_features).tolist(),
        "horizon": horizon,
    }
    predictions = {
        "actual": actual,
        "rollout": rollout,
        "one_step": one_step,
        "persistence": baseline,
        "test_indices": test,
    }
    return metrics, predictions


def save_results(output_dir, history, metrics, settings, predictions) -> None:
    """Write training history, evaluation metrics, splits, and predictions."""
    reports = {
        "history.json": history,
        "metrics.json": metrics,
        "config.json": settings,
    }
    for filename, data in reports.items():
        contents = json.dumps(data, indent=2, allow_nan=False) + "\n"
        (output_dir / filename).write_text(contents)

    np.savez_compressed(output_dir / "predictions.npz", **predictions)


def print_evaluation_summary(metrics, metadata) -> None:
    """Show physical-unit errors and a compact comparison at useful horizons."""
    num_features = len(metrics["one_step_rmse_per_feature"])
    feature_names = [f"State {index}" for index in range(num_features)]
    if (
        metadata.get("sim") == "genesis"
        and metadata.get("dog") == "go2"
        and num_features == 10
    ):
        feature_names = [
            "Roll (rad)",
            "Pitch (rad)",
            "Yaw (rad)",
            "Velocity X (m/s)",
            "Velocity Y (m/s)",
            "Velocity Z (m/s)",
            "Contact 0",
            "Contact 1",
            "Contact 2",
            "Contact 3",
        ]

    print("\nHeld-out test results (lower error is better)")
    print("Persistence baseline: predict no change in state.")
    print(f"\n{'One-step RMSE':<24} {'Surrogate':>12} {'Persistence':>12}")
    print("-" * 50)
    for index, name in enumerate(feature_names):
        learned = metrics["one_step_rmse_per_feature"][index]
        baseline = metrics["persistence_one_step_rmse_per_feature"][index]
        print(f"{name:<24} {learned:>12.5g} {baseline:>12.5g}")

    active = metrics["active_features"]
    excluded = [name for index, name in enumerate(feature_names) if index not in active]
    if excluded:
        print("\nConstant training features excluded from rollout scores:")
        print(f"  {', '.join(excluded)}")
    if not active:
        print("Rollout scores unavailable: all training features are constant.")
        return

    print("\nRollout RMSE normalized by each feature's training range")
    print(f"{'Horizon (steps)':<24} {'Surrogate':>12} {'Persistence':>12}")
    print("-" * 50)
    horizon = metrics["horizon"]
    checkpoints = sorted(
        {step for step in (1, 5, 10, 25, 50, horizon) if step <= horizon}
    )
    for step in checkpoints:
        learned = metrics["rollout_nrmse_by_step"][step - 1]
        baseline = metrics["persistence_rollout_nrmse_by_step"][step - 1]
        print(f"{step:<24} {learned:>12.5g} {baseline:>12.5g}")


def main() -> None:
    parser, args = parse_args()
    try:
        states, actions, next_states, metadata = load_dataset(str(args.dataset))
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Cannot load dataset: {error}")
    validate_dataset(states, actions, next_states, parser)

    train, validation, test = split_trajectories(
        len(states), args.test_frac, args.val_frac, args.seed
    )
    print(f"Dataset: {args.dataset}")
    print(
        f"{len(states):,} trajectories × {states.shape[1]:,} steps; "
        f"{states.shape[-1]} state features, {actions.shape[-1]} actions"
    )
    print(
        f"Trajectories: {len(train)} train, "
        f"{len(validation)} validation, {len(test)} test"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "model.pt").exists():
        parser.error("Output already contains model.pt; choose a new --out directory")

    config = SurrogateConfig(
        state_dim=states.shape[-1],
        action_dim=actions.shape[-1],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        val_frac=args.val_frac,
        seed=args.seed,
        sim=metadata.get("sim", "unknown"),
        dog=metadata.get("dog", "unknown"),
        dt=metadata.get("dt", 0.02),
    )
    model = Surrogate(config, device=args.device)
    print(
        f"Device: {model.device} | Batch size: {args.batch_size:,} | "
        f"Learning rate: {args.lr:g}"
    )
    print(f"Maximum epochs: {args.epochs} | Early-stopping patience: {args.patience}")
    print("Loss: mean squared error on standardized next-state features.\n")
    history, training_summary = fit_model(
        model,
        training=(states[train], actions[train], next_states[train]),
        validation=(states[validation], actions[validation], next_states[validation]),
    )
    model.save(str(args.out / "model.pt"))

    metrics, predictions = evaluate_model(
        model, states, actions, next_states, train, test, args.horizon
    )
    settings = {
        **vars(args),
        "dataset": str(args.dataset.resolve()),
        "out": str(args.out),
        "metadata": metadata,
        "training_summary": training_summary,
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "test_indices": test.tolist(),
    }
    save_results(args.out, history, metrics, settings, predictions)

    print_evaluation_summary(metrics, metadata)
    print(f"\nSaved model, metrics and predictions to {args.out}")
    print(f"Generate plots: python visualize_surrogate.py {args.out}")


if __name__ == "__main__":
    main()
