import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute_stream.pipeline import StreamingPipeline
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer


def make_data():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [8.0, 9.0],
        [9.0, 8.0],
        [1.5, 2.5],
        [8.5, 9.5],
    ])
    y = np.array([0, 0, 1, 1, 0, 1])
    return X, y


def make_pipeline(buffer_size=2):
    return StreamingPipeline([
        ("tree", DecisionTreeClassifier(
            max_depth=3,
            buffer_size=buffer_size
        ))
    ])


def test_streaming_pipeline_inherits_pipeline():
    pipe = make_pipeline()
    assert isinstance(pipe, Pipeline)


def test_partial_fit_returns_pipeline():
    X, y = make_data()
    pipe = make_pipeline()

    result = pipe.partial_fit(X[:2], y[:2])

    assert result is pipe


def test_multiple_chunks_train_successfully():
    X, y = make_data()
    pipe = make_pipeline(buffer_size=2)

    pipe.partial_fit(X[:2], y[:2])
    pipe.partial_fit(X[2:4], y[2:4])
    pipe.partial_fit(X[4:], y[4:])

    assert pipe.steps[-1][1].n_retrains_ == 3


def test_predict_returns_correct_number_of_labels():
    X, y = make_data()
    pipe = make_pipeline()

    pipe.partial_fit(X[:4], y[:4])
    predictions = pipe.predict(X[:3])

    assert predictions.shape == (3,)


def test_predict_proba_returns_probabilities():
    X, y = make_data()
    pipe = make_pipeline()

    pipe.partial_fit(X[:4], y[:4])
    probabilities = pipe.predict_proba(X[:3])

    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_predict_before_fit_raises_error():
    pipe = make_pipeline()

    with pytest.raises(Exception):
        pipe.predict([[1.0, 2.0]])


def test_predict_proba_before_fit_raises_error():
    pipe = make_pipeline()

    with pytest.raises(Exception):
        pipe.predict_proba([[1.0, 2.0]])


def test_invalid_input_dimensions_raise_error():
    pipe = make_pipeline()

    with pytest.raises(ValueError):
        pipe.partial_fit([1.0, 2.0], [0])


def test_mismatched_sample_counts_raise_error():
    pipe = make_pipeline()

    with pytest.raises(ValueError):
        pipe.partial_fit(
            np.array([[1.0, 2.0], [2.0, 3.0]]),
            np.array([0])
        )


def test_empty_chunk_raises_error():
    pipe = make_pipeline()

    with pytest.raises(ValueError):
        pipe.partial_fit(
            np.empty((0, 2)),
            np.array([])
        )


def test_stream_trainer_accepts_streaming_pipeline():
    X, y = make_data()
    pipe = make_pipeline(buffer_size=2)
    trainer = StreamTrainer(pipe)

    trainer.fit_chunk(X[:2], y[:2])
    trainer.fit_chunk(X[2:4], y[2:4])

    assert trainer.n_chunks_ == 2


def test_memory_bytes_returns_integer():
    X, y = make_data()
    pipe = make_pipeline()
