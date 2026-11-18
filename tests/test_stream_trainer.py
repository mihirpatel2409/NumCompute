import csv

import numpy as np
import pytest

from numcompute.utils import NotFittedError
from numcompute_stream.stream import StreamTrainer


class DummyStreamModel:
    """Small deterministic model for testing StreamTrainer."""

    def __init__(self):
        self.fitted = False
        self.n_retrains_ = 0
        self.n_buffered_ = 0
        self.last_fit_seconds_ = 0.01
        self.classes_ = np.array([0, 1])

    def partial_fit(self, X, y):
        self.fitted = True
        self.n_retrains_ += 1
        self.n_buffered_ = 0
        return self

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError("Model has not been fitted")

        X = np.asarray(X)
        return (X[:, 0] >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X)
        positive = np.clip(X[:, 0], 0, 1)

        return np.column_stack((1 - positive, positive))

    def memory_bytes(self):
        return 128


class ModelWithoutPredict:
    def partial_fit(self, X, y):
        pass


class ModelWithoutPartialFit:
    def predict(self, X):
        pass


class ModelWithoutProbability:
    def partial_fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def make_data():
    X = np.array([
        [0.1],
        [0.2],
        [0.8],
        [0.9],
    ])

    y = np.array([0, 0, 1, 1])

    return X, y


def test_constructor_rejects_model_without_partial_fit():
    with pytest.raises(TypeError):
        StreamTrainer(ModelWithoutPartialFit())


def test_constructor_rejects_model_without_predict():
    with pytest.raises(TypeError):
        StreamTrainer(ModelWithoutPredict())


def test_constructor_requires_predict_proba_for_auc():
    with pytest.raises(TypeError):
        StreamTrainer(ModelWithoutProbability(), track_auc=True)


def test_initial_trainer_state():
    trainer = StreamTrainer(DummyStreamModel())

    assert trainer.n_chunks_ == 0
    assert trainer.n_samples_seen_ == 0
    assert trainer.history == []
    assert trainer.rolling is None
    assert trainer.auc is None


def test_score_chunk_before_training_handles_warmup():
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel())

    result = trainer.score_chunk(X, y)

    assert result["predicted"] is False
    assert result["accuracy"] is None
    assert result["n_samples"] == 4
    assert trainer.metrics.n_samples_ == 0


def test_score_chunk_updates_metrics_after_training():
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel())

    trainer.model.partial_fit(X, y)
    result = trainer.score_chunk(X, y)

    assert result["predicted"] is True
    assert result["accuracy"] == 1.0
    assert result["error_rate"] == 0.0
    assert result["cumulative_accuracy"] == 1.0


def test_rolling_metrics_are_updated():
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel(), window=3)

    trainer.model.partial_fit(X, y)
    result = trainer.score_chunk(X, y)

    assert result["rolling_accuracy"] == 1.0


def test_auc_tracking():
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel(), track_auc=True)

    trainer.model.partial_fit(X, y)
    result = trainer.score_chunk(X, y)

    assert result["auc"] == pytest.approx(1.0)


def test_fit_chunk_records_training_information():
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel())

    result = trainer.fit_chunk(X, y)

    assert result["retrained"] is True
    assert result["n_retrains"] == 1
    assert result["n_samples_seen"] == 4
    assert result["memory_bytes"] == 128
    assert trainer.n_chunks_ == 1
    assert len(trainer.history) == 1


def test_failed_training_does_not_append_history():
    class FailingModel(DummyStreamModel):
        def partial_fit(self, X, y):
            raise RuntimeError("Training failed")

    X, y = make_data()
    trainer = StreamTrainer(FailingModel())

    with pytest.raises(RuntimeError):
        trainer.fit_chunk(X, y)

    assert trainer.history == []
    assert trainer.n_chunks_ == 0


def test_process_chunk_scores_then_trains():
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel())

    result = trainer.process_chunk(X, y)

    assert result["predicted"] is False
    assert trainer.n_chunks_ == 1
    assert trainer.n_samples_seen_ == 4


def test_run_processes_multiple_chunks():
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel())

    chunks = [
        (X[:2], y[:2]),
        (X[2:], y[2:]),
    ]

    results = trainer.run(chunks)

    assert len(results) == 2
    assert trainer.n_chunks_ == 2
    assert trainer.n_samples_seen_ == 4


def test_get_returns_history_column():
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel())

    trainer.process_chunk(X, y)

    assert trainer.get("chunk") == [1]
    assert trainer.get("n_samples") == [4]


def test_get_rejects_unknown_field():
    trainer = StreamTrainer(DummyStreamModel())

    with pytest.raises(KeyError):
        trainer.get("unknown_field")


def test_summary_returns_run_statistics():
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel())

    trainer.process_chunk(X, y)
    result = trainer.summary()

    assert result["n_chunks"] == 1
    assert result["n_samples_seen"] == 4
    assert result["n_retrains"] == 1
    assert result["final_memory_bytes"] == 128


def test_csv_export_rejects_empty_history(tmp_path):
    trainer = StreamTrainer(DummyStreamModel())

    with pytest.raises(ValueError):
        trainer.to_csv(tmp_path / "empty.csv")


def test_csv_export_writes_history(tmp_path):
    X, y = make_data()
    trainer = StreamTrainer(DummyStreamModel())

    trainer.process_chunk(X, y)

    path = tmp_path / "stream_results.csv"
    result = trainer.to_csv(path)

    assert result == str(path)
    assert path.exists()

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["chunk"] == "1"


def test_repr_contains_trainer_information():
    trainer = StreamTrainer(DummyStreamModel())

    result = repr(trainer)

    assert "StreamTrainer" in result
    assert "chunks=0" in result
    assert "samples=0" in result