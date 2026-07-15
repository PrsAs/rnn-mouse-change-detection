import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_trial_neuron_matrix(
    spike_counts: pd.DataFrame,
    count_column: str = "response_spike_count",
):
    """
    Convert unit-trial spike counts to a trial-by-neuron matrix.

    Rows are change trials.
    Columns are neuron IDs.
    Labels:
        1 = hit
        0 = miss
    """
    required_columns = {
        "outcome",
        "outcome_trial_number",
        "unit_id",
        count_column,
    }

    missing = required_columns - set(spike_counts.columns)

    if missing:
        raise ValueError(
            f"Missing spike-count columns: {sorted(missing)}"
        )

    data = spike_counts.loc[
        spike_counts["outcome"].isin(["hit", "miss"])
    ].copy()

    matrix = (
        data
        .pivot(
            index=["outcome", "outcome_trial_number"],
            columns="unit_id",
            values=count_column,
        )
        .reset_index()
    )

    labels = (
        matrix["outcome"]
        .eq("hit")
        .astype(int)
        .to_numpy()
    )

    trial_ids = matrix[
        ["outcome", "outcome_trial_number"]
    ].copy()

    features = matrix.drop(
        columns=["outcome", "outcome_trial_number"]
    )

    unit_ids = features.columns.to_numpy()
    X = features.to_numpy(dtype=float)

    if np.isnan(X).any():
        raise ValueError(
            "Feature matrix contains missing values."
        )

    return X, labels, unit_ids, trial_ids


def make_decoder(
    regularization_c: float = 1.0,
    random_seed: int = 42,
):
    """
    Build a standardized L2-regularized logistic-regression decoder.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=regularization_c,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def cross_validated_decoder_scores(
    X: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 20,
    random_seed: int = 42,
    regularization_c: float = 1.0,
):
    """
    Evaluate hit/miss decoding with repeated stratified CV.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_seed,
    )

    decoder = make_decoder(
        regularization_c=regularization_c,
        random_seed=random_seed,
    )

    rows = []

    for split_number, (train_index, test_index) in enumerate(
        cv.split(X, labels),
        start=1,
    ):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = labels[train_index]
        y_test = labels[test_index]

        decoder.fit(X_train, y_train)

        predicted_labels = decoder.predict(X_test)
        predicted_probabilities = decoder.predict_proba(
            X_test
        )[:, 1]

        rows.append(
            {
                "split": split_number,
                "accuracy": accuracy_score(
                    y_test,
                    predicted_labels,
                ),
                "balanced_accuracy": (
                    balanced_accuracy_score(
                        y_test,
                        predicted_labels,
                    )
                ),
                "roc_auc": roc_auc_score(
                    y_test,
                    predicted_probabilities,
                ),
                "n_test_trials": len(test_index),
                "n_test_hits": int(y_test.sum()),
                "n_test_misses": int(
                    (y_test == 0).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def decode_hit_miss_population(
    spike_counts: pd.DataFrame,
    count_column: str = "response_spike_count",
    n_splits: int = 5,
    n_repeats: int = 20,
    random_seed: int = 42,
    regularization_c: float = 1.0,
):
    """
    Build the trial-neuron matrix and run repeated-CV decoding.
    """
    X, labels, unit_ids, trial_ids = build_trial_neuron_matrix(
        spike_counts=spike_counts,
        count_column=count_column,
    )

    cv_results = cross_validated_decoder_scores(
        X=X,
        labels=labels,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_seed=random_seed,
        regularization_c=regularization_c,
    )

    summary = {
        "n_trials": len(labels),
        "n_hits": int(labels.sum()),
        "n_misses": int((labels == 0).sum()),
        "n_units": len(unit_ids),
        "mean_accuracy": cv_results["accuracy"].mean(),
        "mean_balanced_accuracy": (
            cv_results["balanced_accuracy"].mean()
        ),
        "mean_roc_auc": cv_results["roc_auc"].mean(),
    }

    return {
        "X": X,
        "labels": labels,
        "unit_ids": unit_ids,
        "trial_ids": trial_ids,
        "cv_results": cv_results,
        "summary": pd.DataFrame([summary]),
    }


def permutation_test_decoder(
    X: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 200,
    n_splits: int = 5,
    n_repeats: int = 20,
    random_seed: int = 42,
    regularization_c: float = 1.0,
):
    """
    Permutation test for mean cross-validated balanced accuracy.

    The same decoding procedure is repeated after random label shuffling.
    """
    rng = np.random.default_rng(random_seed)

    observed_results = cross_validated_decoder_scores(
        X=X,
        labels=labels,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_seed=random_seed,
        regularization_c=regularization_c,
    )

    observed_score = (
        observed_results["balanced_accuracy"].mean()
    )

    permutation_scores = []

    for permutation_number in range(n_permutations):
        shuffled_labels = rng.permutation(labels)

        shuffled_results = cross_validated_decoder_scores(
            X=X,
            labels=shuffled_labels,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_seed=(
                random_seed + permutation_number + 1
            ),
            regularization_c=regularization_c,
        )

        permutation_scores.append(
            shuffled_results["balanced_accuracy"].mean()
        )

    permutation_scores = np.asarray(permutation_scores)

    p_value = (
        1
        + np.sum(permutation_scores >= observed_score)
    ) / (n_permutations + 1)

    permutation_table = pd.DataFrame(
        {
            "permutation_number": np.arange(
                1,
                n_permutations + 1,
            ),
            "balanced_accuracy": permutation_scores,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "observed_balanced_accuracy": observed_score,
                "permutation_mean": permutation_scores.mean(),
                "permutation_std": permutation_scores.std(
                    ddof=1
                ),
                "n_permutations": n_permutations,
                "permutation_p_value": p_value,
            }
        ]
    )

    return {
        "observed_cv_results": observed_results,
        "permutation_scores": permutation_table,
        "summary": summary,
    }
