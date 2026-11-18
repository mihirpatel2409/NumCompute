"""Benchmark streaming decision tree with different buffer sizes."""

import time
import pathlib
import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


ROOT = pathlib.Path(__file__).resolve().parents[1]


def generate_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n, 6))
    noise = rng.normal(0, 0.5, n)

    y = (
        X[:, 0] + X[:, 1] * 0.8 - X[:, 2] * 0.5 + noise > 0
    ).astype(int)

    return X, y


def benchmark_buffer(buffer_size, X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(
        max_depth=5,
        buffer_size=buffer_size
    )

    chunk_size = 64
    start = time.perf_counter()

    for i in range(0, len(X_train), chunk_size):
        model.partial_fit(
            X_train[i:i + chunk_size],
            y_train[i:i + chunk_size]
        )

    training_time = time.perf_counter() - start

    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    return {
        "buffer_size": buffer_size,
        "accuracy": accuracy,
        "training_time": training_time,
        "retrain_count": model.n_retrains_,
    }


def main():
    X, y = generate_data()

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results = []

    for buffer_size in [16, 32, 64, 128, 256]:
        print(f"Testing buffer size: {buffer_size}")

        result = benchmark_buffer(
            buffer_size,
            X_train,
            y_train,
            X_test,
            y_test
        )

        results.append(result)

    lines = [
        "# Streaming Decision Tree Benchmark",
        "",
        "| Buffer Size | Accuracy | Training Time (seconds) | Retrain Count |",
        "|---:|---:|---:|---:|",
    ]

    for result in results:
        lines.append(
            f"| {result['buffer_size']} "
            f"| {result['accuracy']:.4f} "
            f"| {result['training_time']:.6f} "
            f"| {result['retrain_count']} |"
        )

    output = "\n".join(lines)

    print("\n" + output)

    path = ROOT / "benchmark" / "streaming_results.md"
    path.write_text(output, encoding="utf-8")

    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()