"""Compare streaming-tree buffer sizes under a consistent chunk workload.

Each run uses the same generated dataset and 16-row incoming chunks. All tested
buffer sizes are multiples of the chunk size, so no call sends an oversized
chunk that could trigger an early rebuild. Training time is reported as the
median of repeated runs to reduce single-run timing noise.
"""

from __future__ import annotations

import pathlib
import statistics
import time

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUFFER_SIZES = (16, 32, 64, 128, 256)
CHUNK_SIZE = 16
N_RUNS = 3
RANDOM_SEED = 42


def generate_data(n: int = 3000, seed: int = RANDOM_SEED):
    """Generate a reproducible binary-classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    noise = rng.normal(0, 0.5, n)
    y = (X[:, 0] + X[:, 1] * 0.8 - X[:, 2] * 0.5 + noise > 0).astype(int)
    return X, y


def benchmark_buffer(
    buffer_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
    n_runs: int = N_RUNS,
):
    """Benchmark one buffer size using identical incoming chunk sizes."""
    if buffer_size < chunk_size or buffer_size % chunk_size != 0:
        raise ValueError(
            "Each buffer size must be a positive multiple of chunk_size."
        )
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")

    training_times = []
    accuracies = []
    retrain_counts = []

    for _ in range(n_runs):
        model = DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size)
        start = time.perf_counter()

        for start_idx in range(0, len(X_train), chunk_size):
            stop_idx = start_idx + chunk_size
            model.partial_fit(X_train[start_idx:stop_idx], y_train[start_idx:stop_idx])

        training_times.append(time.perf_counter() - start)
        predictions = model.predict(X_test)
        accuracies.append(float(np.mean(predictions == y_test)))
        retrain_counts.append(int(model.n_retrains_))

    if len(set(retrain_counts)) != 1:
        raise RuntimeError("Retraining count changed between identical benchmark runs.")

    return {
        "buffer_size": buffer_size,
        "chunk_size": chunk_size,
        "accuracy": statistics.mean(accuracies),
        "median_training_time": statistics.median(training_times),
        "retrain_count": retrain_counts[0],
        "runs": n_runs,
    }


def main():
    X, y = generate_data()
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results = []
    for buffer_size in BUFFER_SIZES:
        print(f"Testing buffer size: {buffer_size}")
        results.append(
            benchmark_buffer(buffer_size, X_train, y_train, X_test, y_test)
        )

    lines = [
        "# Streaming Decision Tree Benchmark",
        "",
        f"- Training samples: {len(X_train)}",
        f"- Test samples: {len(X_test)}",
        f"- Fixed incoming chunk size: {CHUNK_SIZE}",
        f"- Repeated runs per buffer size: {N_RUNS}",
        "- Training time: median across repeated runs",
        "",
        "| Buffer Size | Chunk Size | Accuracy | Median Training Time (seconds) | Retrain Count | Runs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        lines.append(
            f"| {result['buffer_size']} "
            f"| {result['chunk_size']} "
            f"| {result['accuracy']:.4f} "
            f"| {result['median_training_time']:.6f} "
            f"| {result['retrain_count']} "
            f"| {result['runs']} |"
        )

    output = "\n".join(lines) + "\n"
    print("\n" + output)

    path = ROOT / "benchmark" / "streaming_results.md"
    path.write_text(output, encoding="utf-8")
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
