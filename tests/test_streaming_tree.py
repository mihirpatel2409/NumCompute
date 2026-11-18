import numpy as np
import pytest

from numcompute_stream.tree import (
    DecisionTreeClassifier,
    gini_impurity,
    entropy_impurity,
    information_gain,
)

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


def test_tree_fit_returns_classifier():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=3)

    result = tree.fit(X, y)

    assert result is tree


def test_tree_is_fitted_after_training():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=3)

    tree.fit(X, y)

    assert tree.is_fitted_ is True


def test_tree_predict_returns_correct_shape():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    predictions = tree.predict(X)

    assert predictions.shape == (len(X),)


def test_tree_predict_returns_known_classes():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    predictions = tree.predict(X)

    assert set(predictions).issubset(set(y))


def test_tree_predict_proba_returns_correct_shape():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    probabilities = tree.predict_proba(X)

    assert probabilities.shape == (len(X), 2)


def test_tree_probabilities_sum_to_one():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    probabilities = tree.predict_proba(X)

    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_tree_score_returns_valid_accuracy():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    score = tree.score(X, y)

    assert 0.0 <= score <= 1.0


def test_tree_depth_respects_max_depth():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    assert tree.depth_ <= 2


def test_tree_has_at_least_one_node():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    assert tree.n_nodes_ >= 1


def test_tree_has_at_least_one_leaf():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    assert tree.n_leaves_ >= 1 

# =========================================
# Impurity and information gain tests
# =========================================

def test_gini_impurity_is_zero_for_pure_node():
    assert gini_impurity([5, 0]) == pytest.approx(0.0)


def test_gini_impurity_for_balanced_classes():
    assert gini_impurity([5, 5]) == pytest.approx(0.5)


def test_gini_impurity_supports_multiple_nodes():
    result = gini_impurity([[5, 0], [5, 5]])
    assert np.allclose(result, [0.0, 0.5])


def test_entropy_is_zero_for_pure_node():
    assert entropy_impurity([5, 0]) == pytest.approx(0.0)


def test_entropy_for_balanced_classes():
    assert entropy_impurity([5, 5]) == pytest.approx(1.0)


def test_entropy_supports_multiple_nodes():
    result = entropy_impurity([[5, 0], [5, 5]])
    assert np.allclose(result, [0.0, 1.0])


def test_information_gain_for_perfect_split():
    gain = information_gain([5, 5], [5, 0], [0, 5])
    assert gain == pytest.approx(0.5)


def test_information_gain_for_useless_split():
    gain = information_gain([5, 5], [2, 2], [3, 3])
    assert gain == pytest.approx(0.0)


def test_information_gain_supports_entropy():
    gain = information_gain(
        [5, 5], [5, 0], [0, 5],
        criterion="entropy"
    )
    assert gain == pytest.approx(1.0)


def test_information_gain_rejects_unknown_criterion():
    with pytest.raises(ValueError):
        information_gain(
            [5, 5], [5, 0], [0, 5],
            criterion="invalid"
        )

        # =========================================
# Decision tree splitting and validation
# =========================================

def test_tree_learns_numerical_feature_split():
    X = np.array([[1], [2], [3], [8], [9], [10]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    assert np.array_equal(tree.predict(X), y)


def test_tree_supports_entropy_criterion():
    X, y = make_data()

    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3
    )
    tree.fit(X, y)

    assert tree.is_fitted_ is True
    assert tree.predict(X).shape == y.shape


def test_tree_handles_single_class_dataset():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([1, 1, 1, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    assert np.all(tree.predict(X) == 1)
    assert tree.n_leaves_ == 1


def test_tree_handles_constant_features():
    X = np.ones((5, 2))
    y = np.array([0, 1, 0, 1, 0])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    assert tree.is_fitted_ is True
    assert tree.n_leaves_ == 1


def test_tree_handles_nan_values():
    X = np.array([
        [1.0],
        [2.0],
        [np.nan],
        [8.0],
        [9.0],
        [np.nan]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    predictions = tree.predict(X)

    assert predictions.shape == y.shape


def test_tree_rejects_infinite_training_values():
    X = np.array([[1.0], [np.inf], [3.0]])
    y = np.array([0, 1, 0])

    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.fit(X, y)


def test_tree_rejects_mismatched_training_samples():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1])

    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.fit(X, y)


def test_tree_rejects_prediction_with_wrong_feature_count():
    X, y = make_data()

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    with pytest.raises(ValueError):
        tree.predict(np.array([[1.0, 2.0, 3.0]]))


def test_tree_predict_before_fit_raises_error():
    tree = DecisionTreeClassifier()

    with pytest.raises(Exception):
        tree.predict(np.array([[1.0, 2.0]]))


def test_tree_predict_proba_before_fit_raises_error():
    tree = DecisionTreeClassifier()

    with pytest.raises(Exception):
        tree.predict_proba(np.array([[1.0, 2.0]]))