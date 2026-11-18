import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)


def test_plot_metric_over_time_returns_figure_and_axes():
    fig, ax = plot_metric_over_time(
        [0.6, 0.7, 0.8],
        show=False
    )

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1


def test_plot_metric_over_time_uses_custom_title():
    fig, ax = plot_metric_over_time(
        [0.5, 0.6],
        title="Accuracy",
        ylabel="Score",
        show=False
    )

    assert ax.get_title() == "Accuracy"
    assert ax.get_ylabel() == "Score"


def test_plot_metric_rejects_empty_values():
    with pytest.raises(ValueError):
        plot_metric_over_time([], show=False)


def test_plot_metric_rejects_two_dimensional_values():
    with pytest.raises(ValueError):
        plot_metric_over_time([[0.5, 0.6]], show=False)


def test_compare_models_returns_two_lines():
    fig, ax = compare_models(
        [0.6, 0.7],
        [0.7, 0.8],
        show=False
    )

    assert len(ax.lines) == 2


def test_compare_models_uses_custom_labels():
    fig, ax = compare_models(
        [0.6, 0.7],
        [0.7, 0.8],
        labels=("Tree", "Model"),
        show=False
    )

    assert ax.get_legend() is not None


def test_compare_models_rejects_empty_values():
    with pytest.raises(ValueError):
        compare_models([], [0.5], show=False)


def test_compare_models_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        compare_models([[0.5, 0.6]], [0.7], show=False)


def test_compare_models_rejects_invalid_labels():
    with pytest.raises(ValueError):
        compare_models(
            [0.5],
            [0.6],
            labels=("Only one",),
            show=False
        )


def test_predictions_plot_returns_two_lines():
    fig, ax = plot_predictions_vs_ground_truth(
        [0, 1, 1],
        [0, 1, 0],
        show=False
    )

    assert len(ax.lines) == 2


def test_predictions_plot_rejects_empty_values():
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([], [], show=False)


def test_predictions_plot_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth(
            [0, 1, 1],
            [0, 1],
            show=False
        )


def test_visualisation_can_save_image(tmp_path):
    output = tmp_path / "accuracy.png"

    fig, ax = plot_metric_over_time(
        [0.6, 0.7, 0.8],
        save_path=output,
        show=False
    )

    assert output.exists()
    assert output.stat().st_size > 0