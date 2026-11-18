"""Visualisation tools for streaming machine-learning results."""

import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
]


def plot_metric_over_time(
    metric_values,
    title="Metric over time",
    ylabel="Metric",
    save_path=None,
    show=True,
):
    """Plot a metric across streaming chunks."""
    values = np.asarray(metric_values, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("metric_values must be a non-empty 1D sequence")

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(values) + 1), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    save_path=None,
    show=True,
):
    """Compare two models using their metric values over time."""
    first = np.asarray(metric1, dtype=float)
    second = np.asarray(metric2, dtype=float)

    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Both metrics must be 1D sequences")

    if first.size == 0 or second.size == 0:
        raise ValueError("Metric sequences must not be empty")

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two names")

    fig, ax = plt.subplots()
    ax.plot(first, marker="o", label=labels[0])
    ax.plot(second, marker="s", label=labels[1])
    ax.set_title("Model comparison")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    save_path=None,
    show=True,
):
    """Plot predicted labels against actual labels."""
    actual = np.asarray(y_true).ravel()
    predicted = np.asarray(y_pred).ravel()

    if actual.size == 0 or predicted.size == 0:
        raise ValueError("Inputs must not be empty")

    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")

    fig, ax = plt.subplots()
    sample_numbers = np.arange(len(actual))

    ax.plot(sample_numbers, actual, marker="o", label="Ground truth")
    ax.plot(sample_numbers, predicted, marker="x", label="Predictions")
    ax.set_title("Predictions vs Ground Truth")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax