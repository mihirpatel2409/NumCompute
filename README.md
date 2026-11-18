# NumCompute — Streaming Machine Learning

A Python and NumPy scientific-computing toolkit extended with a streaming machine-learning workflow. Assignment 2.2 focuses on processing data in sequential chunks, buffered decision-tree retraining, prequential evaluation, performance monitoring, and visualisation.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-orange.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/tests-354%20collected-brightgreen.svg)]()
[![Assignment](https://img.shields.io/badge/Assignment-2.2-informational.svg)]()

---

## 1. Project overview

Traditional batch machine learning generally trains on a complete dataset at once. A streaming workflow instead receives data over time and processes it in smaller chunks. This project extends NumCompute with components that accept successive chunks, evaluate the current model, update the model's training buffer, and record performance as the stream progresses.

The Assignment 2.2 implementation brings together:

- Chunk-based CSV ingestion using NumCompute's `iter_csv_chunks`.
- A decision-tree classifier with buffered periodic retraining.
- Test-then-train evaluation through `StreamTrainer`.
- An incremental pipeline for preprocessing and model training.
- Streaming metrics, timing information, retraining records, and model-memory estimates.
- Visualisation of metrics across processed chunks.

The original `numcompute` package remains part of the repository and provides the numerical, preprocessing, metrics, pipeline, and data-ingestion foundations.

## 2. Streaming architecture

The workflow follows this sequence:

```text
CSV dataset
    |
    v
Chunk reader (iter_csv_chunks)
    |
    v
StreamTrainer
    |
    +---- 1. Predict on the incoming chunk
    |
    +---- 2. Record evaluation metrics
    |
    +---- 3. Pass the chunk to partial_fit
    |           |
    |           v
    |     Buffered Decision Tree
    |           |
    |           +---- Accumulate samples
    |           +---- Retrain when buffer threshold is reached
    |           +---- Clear the buffer after retraining
    |
    v
Chunk history and cumulative summary
    |
    v
Metric visualisation
```

### Main components

| Component | Purpose |
|---|---|
| `numcompute_stream.tree.DecisionTreeClassifier` | Classification tree supporting Gini or entropy impurity, numerical threshold splits, prediction, and buffered `partial_fit` training. |
| `numcompute_stream.stream.StreamTrainer` | Coordinates chunk evaluation and training, and records metrics, timing, retraining events, and history. |
| `numcompute_stream.pipeline.StreamingPipeline` | Applies intermediate transformers and passes transformed chunks to an incremental final estimator. |
| `numcompute_stream.visualise` | Plots metrics over time and provides model/prediction comparison plots. |
| `numcompute.metrics` | Supplies streaming metrics, rolling metrics, and streaming AUC used by the trainer. |

### Important implementation distinction

The decision tree uses **buffered periodic retraining**. Incoming samples are accumulated until the configured buffer threshold is reached, at which point the tree is rebuilt from the buffered data. This is not a Hoeffding Tree and does not incrementally update individual split nodes. The distinction matters when interpreting the algorithm and its benchmark results.

## 3. Evaluation methodology

`StreamTrainer.process_chunk(X, y)` uses **prequential (test-then-train) evaluation**:

1. The current model predicts labels for the incoming chunk, if it has already been fitted.
2. The predictions are compared with the true labels and evaluation metrics are updated.
3. The same chunk is passed to the model's `partial_fit` method.
4. The trainer records processing time, training time, retraining information, buffer size, and memory information.

The first chunk is a warm-up when the model has not yet been fitted. It is used for training but is not scored, because the model cannot make meaningful predictions before its first fit.

The trainer's history includes chunk number, sample counts, whether predictions were available, chunk accuracy, error rate, cumulative accuracy, optional rolling accuracy and AUC, prediction/training/retraining times, retraining count, buffered samples, and estimated model memory.

## 4. Installation and setup

### Requirements

- Python 3.9 or later
- NumPy 1.24 or later
- pytest and Jupyter for development and notebook use
- Matplotlib for the demonstration plots

### Windows — PowerShell

Clone this repository and switch to the Assignment 2.2 branch:

```powershell
git clone https://github.com/mihirpatel2409/NumCompute.git
cd NumCompute
git checkout mihir-assignment-2.2
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the package and development dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install matplotlib
```

The editable installation makes the local packages available to Python while working in the repository.

## 5. Run the streaming demonstration

Open `demo/stream_demo.ipynb` in Visual Studio Code or Jupyter. Select the Python environment in which the project dependencies were installed, then run the notebook cells from top to bottom.

Alternatively, launch the notebook from the repository root:

```powershell
jupyter notebook demo/stream_demo.ipynb
```

The notebook uses `demo/sample_data.csv`, selects age and income as numerical input features, and uses churn as the target. It reads the CSV in chunks of 16 rows and configures a Gini decision tree with a maximum depth of 5, buffer size of 32, and random seed of 42.

### Core usage example

Run this from the repository root:

```python
from pathlib import Path

from numcompute.io import iter_csv_chunks
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer

data_path = Path("demo/sample_data.csv")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    buffer_size=32,
    random_state=42,
)
trainer = StreamTrainer(model)

chunks = iter_csv_chunks(
    data_path,
    chunksize=16,
    skip_header=1,
    usecols=[0, 1, 4],
    missing="nan",
)

for chunk in chunks:
    X = chunk[:, :2]
    y = chunk[:, 2].astype(int)
    trainer.process_chunk(X, y)

print(trainer.summary())
```

The notebook also extracts cumulative accuracy from `trainer.history` and visualises it with `plot_metric_over_time`.

## 6. Benchmark results

The repository's recorded buffer-size experiment is documented in [`benchmark/streaming_results.md`](benchmark/streaming_results.md).

| Buffer size | Accuracy | Training time (seconds) | Retrain count |
|---:|---:|---:|---:|
| 16 | 0.7183 | 0.154687 | 38 |
| 32 | 0.7183 | 0.151658 | 38 |
| 64 | 0.7150 | 0.152099 | 37 |
| 128 | 0.7450 | 0.112824 | 18 |
| 256 | 0.7617 | 0.088693 | 9 |

In this recorded experiment, a buffer size of 256 produced the highest listed accuracy (0.7617), the lowest listed training time (0.088693 seconds), and 9 retraining events. The results illustrate the trade-off between buffer size, retraining frequency, and observed performance for this particular run. They are not universal performance guarantees; results depend on the dataset, environment, and experimental configuration.

## 7. Run the tests

Install the development dependencies, then run the complete test suite from the repository root:

```powershell
python -m pytest -q
```

Run the streaming-focused tests:

```powershell
python -m pytest tests/test_stream_trainer.py tests/test_streaming_tree.py tests/test_streaming_pipeline.py tests/test_streaming_visualise.py -v
```

The latest recorded test collection contained **354 tests**, of which **73 are streaming-specific**. A previous full-suite execution completed with all 354 tests passing. Run the commands above on the current checkout to independently verify the results.

## 8. Project structure

```text
NumCompute/
├── numcompute/                  # Core numerical and scientific-computing package
├── numcompute_stream/           # Assignment 2.2 streaming extension
│   ├── metrics.py               # Streaming metric interfaces
│   ├── pipeline.py              # StreamingPipeline
│   ├── preprocessing.py         # Streaming preprocessing interfaces
│   ├── stats.py                 # Streaming statistics interfaces
│   ├── stream.py                 # StreamTrainer
│   ├── tree.py                   # Buffered decision-tree classifier
│   └── visualise.py              # Streaming visualisation
├── demo/
│   ├── stream_demo.ipynb        # Assignment 2.2 end-to-end demonstration
│   └── sample_data.csv          # Demonstration dataset
├── benchmark/
│   ├── run_benchmarks.py
│   ├── streaming_benchmark.py
│   ├── results.md
│   └── streaming_results.md
├── tests/                       # Automated test suite
├── pyproject.toml               # Package metadata and dependencies
├── BUGLOG.md                    # Defect and resolution log
├── CONVENTIONS.md               # Project conventions
└── README.md
```

## 9. Design considerations and limitations

- **Buffered retraining:** the classifier rebuilds the tree from its current buffer rather than updating existing split nodes continuously.
- **Warm-up evaluation:** the first training chunk is not scored when the model has not yet been fitted.
- **Stable preprocessing:** `StreamingPipeline` fits intermediate transformers on the first training chunk and keeps them fixed for subsequent chunks. This preserves a consistent feature representation.
- **Memory reporting:** model memory is an estimate based on arrays reported by model components; it is not a measurement of total Python-process RAM.
- **Benchmark scope:** the published table represents the recorded experiment and should be interpreted in that context.

## 10. Repository information

- **Repository:** [mihirpatel2409/NumCompute](https://github.com/mihirpatel2409/NumCompute)
- **Assignment branch:** `mihir-assignment-2.2`
- **Demonstration:** `demo/stream_demo.ipynb`
- **Streaming benchmark:** `benchmark/streaming_results.md`

---
