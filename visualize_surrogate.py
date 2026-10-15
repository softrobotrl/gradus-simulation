"""Plot saved surrogate training results without rerunning training.

Example:
    python visualize_surrogate.py results/genesis_surrogate
"""

import argparse
import json
from pathlib import Path

import numpy as np


GO2_FEATURES = [
    ("Roll — side-to-side tilt", "Angle (rad)"),
    ("Pitch — nose-up / nose-down tilt", "Angle (rad)"),
    ("Yaw — heading rotation", "Angle (rad)"),
    ("Forward / backward velocity", "Velocity (m/s)"),
    ("Sideways velocity", "Velocity (m/s)"),
    ("Up / down velocity", "Velocity (m/s)"),
    ("Foot contact channel 0", "Contact value"),
    ("Foot contact channel 1", "Contact value"),
    ("Foot contact channel 2", "Contact value"),
    ("Foot contact channel 3", "Contact value"),
]


def time_axis(horizon, metadata):
    """Use physical time only when the recorded control interval is available."""
    steps = np.arange(1, horizon + 1)
    dt = metadata.get("dt")
    if isinstance(dt, (int, float)) and np.isfinite(dt) and dt > 0:
        return steps * dt, "Time since rollout start (s)", f"1 step = {dt:g} s"
    return steps, "Prediction horizon (steps)", "Control interval not recorded"


def format_axis(axis):
    axis.grid(True, color="#dce2e8", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)
    axis.margins(x=0.02)


def plot_evaluation(plt, output_dir, history, metrics, predictions, config):
    """Explain model selection and error accumulation on unseen trajectories."""
    metadata = config.get("metadata", {})
    times, time_label, timing_note = time_axis(metrics["horizon"], metadata)
    fig, (loss_axis, error_axis) = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle("Surrogate training and prediction quality", fontsize=17, fontweight="bold")

    epochs = np.arange(1, len(history["train"]) + 1)
    loss_axis.plot(epochs, history["train"], label="Training trajectories", color="#2563a6")
    loss_axis.plot(
        epochs, history["val"], label="Validation trajectories", color="#d66a24"
    )
    best_epoch = config.get("training_summary", {}).get("best_epoch")
    if best_epoch is None and history["val"]:
        best_epoch = int(np.argmin(history["val"])) + 1
    if best_epoch is not None and 1 <= best_epoch <= len(history["val"]):
        loss_axis.axvline(
            best_epoch,
            color="#687583",
            linestyle=":",
            label=f"Best validation: epoch {best_epoch}",
        )
        loss_axis.scatter(
            best_epoch, history["val"][best_epoch - 1], color="#d66a24", zorder=3
        )
    loss_axis.set(
        xlabel="Training epoch",
        ylabel="Mean squared error (standardized state features)",
        title="Learning curves · lower is better",
    )
    loss_axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    loss_axis.legend(frameon=False, fontsize=9)

    count = len(predictions["actual"])
    error_axis.set(
        xlabel=time_label,
        ylabel="RMSE / training feature range",
        title=f"Rollout error on {count} unseen test trajectories",
    )
    if metrics["active_features"]:
        error_axis.plot(
            times, metrics["rollout_nrmse_by_step"],
            label="Surrogate predictions", color="#d66a24", linewidth=2,
        )
        error_axis.plot(
            times, metrics["persistence_rollout_nrmse_by_step"],
            label="Baseline: keep the initial state unchanged",
            color="#687583", linestyle="--",
        )
        error_axis.legend(frameon=False, fontsize=9)
    else:
        error_axis.text(
            0.5, 0.5, "Rollout score unavailable\nAll training features are constant",
            ha="center", va="center", transform=error_axis.transAxes,
        )
    for axis in (loss_axis, error_axis):
        format_axis(axis)
    fig.text(
        0.06, 0.065,
        "Left: validation trajectories select the saved model. "
        "Right: predictions feed back into the model; no state corrections from the simulator.",
        fontsize=9, color="#475569",
    )
    fig.text(
        0.06, 0.03,
        "A rollout score of 0.10 means 10% of training feature ranges in RMS terms; "
        f"constant features are excluded. {timing_note}.",
        fontsize=9, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93), pad=2)
    fig.savefig(output_dir / "evaluation.png", dpi=160)
    plt.close(fig)


def plot_trajectory(plt, output_dir, metrics, predictions, config):
    """Label the physical meaning and units of every displayed state feature."""
    dimensions = list(metrics["active_features"])
    if not dimensions:
        return

    metadata = config.get("metadata", {})
    num_features = predictions["actual"].shape[-1]
    is_go2 = (
        metadata.get("sim") == "genesis"
        and metadata.get("dog") == "go2"
        and num_features == len(GO2_FEATURES)
    )
    if is_go2:
        # Pair orientation and velocity panels so the six physical states form 3 rows.
        dimensions = [index for index in (0, 3, 1, 4, 2, 5, 6, 7, 8, 9) if index in dimensions]
    feature_labels = (
        GO2_FEATURES if is_go2
        else [(f"State feature {index}", "Value (recorded units)") for index in range(num_features)]
    )
    times, time_label, timing_note = time_axis(metrics["horizon"], metadata)
    columns = min(2, len(dimensions))
    rows = (len(dimensions) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(13, 2.7 * rows + 2), squeeze=False, sharex=True
    )
    simulator = "Genesis" if metadata.get("sim") == "genesis" else "Source simulation"
    fig.suptitle(
        f"{'Go2 motion' if is_go2 else 'State predictions'}: simulation vs surrogate",
        fontsize=17, fontweight="bold", y=0.98,
    )
    test_indices = predictions.get("test_indices", config.get("test_indices", []))
    identifier = f" · dataset index {test_indices[0]}" if len(test_indices) else ""
    fig.text(
        0.5, 0.94,
        f"Test trajectory 1 of {len(predictions['actual'])}{identifier} · "
        "Same initial state and recorded actions; model then uses its own predictions",
        ha="center", fontsize=10, color="#475569",
    )
    for axis, dimension in zip(axes.flat, dimensions):
        title, unit = feature_labels[dimension]
        axis.plot(
            times, predictions["actual"][0, :, dimension],
            label=f"{simulator}: recorded motion", color="#2563a6", linewidth=2,
        )
        axis.plot(
            times, predictions["rollout"][0, :, dimension],
            label="Surrogate: predicted motion", color="#d66a24",
            linestyle="--", linewidth=2,
        )
        axis.set(title=title, ylabel=unit, xlabel=time_label)
        axis.tick_params(labelbottom=True)
        format_axis(axis)
    for axis in list(axes.flat)[len(dimensions):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=2, frameon=False)

    if is_go2:
        note = "Angles are relative to the initial base orientation. Velocities use the robot's local body frame."
    else:
        note = "Feature definitions are unavailable for this dataset; values use the original recorded units."
    omitted = sorted(set(range(num_features)) - set(dimensions))
    if is_go2 and omitted == [6, 7, 8, 9]:
        omitted_note = "Four constant foot-contact channels are omitted."
    elif omitted:
        omitted_note = "Constant training features omitted: " + ", ".join(map(str, omitted)) + "."
    else:
        omitted_note = "All state features are shown."
    fig.text(0.06, 0.055, note, fontsize=9, color="#475569")
    fig.text(0.06, 0.028, f"{omitted_note} {timing_note}.", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.08, 1, 0.865), pad=1.8, h_pad=1.5)
    fig.savefig(output_dir / "trajectory.png", dpi=160)
    plt.close(fig)


def save_plots(output_dir, history, metrics, predictions, config=None) -> None:
    """Render labeled figures using saved run metadata, without loading the model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titlesize": 11}):
        plot_evaluation(plt, output_dir, history, metrics, predictions, config or {})
        plot_trajectory(plt, output_dir, metrics, predictions, config or {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Training output directory containing history, metrics, and predictions",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Plot output directory; defaults to the training output directory",
    )
    args = parser.parse_args()

    try:
        config_path = args.run_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        history = json.loads((args.run_dir / "history.json").read_text())
        metrics = json.loads((args.run_dir / "metrics.json").read_text())
        with np.load(args.run_dir / "predictions.npz", allow_pickle=False) as saved:
            predictions = {
                "actual": saved["actual"],
                "rollout": saved["rollout"],
                "test_indices": saved["test_indices"] if "test_indices" in saved else [],
            }
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Cannot load saved results from {args.run_dir}: {error}")

    output_dir = args.out if args.out is not None else args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_plots(output_dir, history, metrics, predictions, config)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
