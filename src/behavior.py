from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm


def decode_text(values):
    """
    Convert HDF5 byte strings to ordinary Python strings.
    """
    return np.array(
        [
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
            for value in values
        ]
    )


def extract_mouse_trials(nwb_path: str | Path):
    """
    Extract the small behavioral trials table from an Allen NWB file.

    This reads only behavioral trial fields, not neural spike-time arrays.
    """
    nwb_path = Path(nwb_path)

    with h5py.File(nwb_path, "r") as h5_file:
        trials = h5_file["intervals"]["trials"]

        mouse_trials = pd.DataFrame(
            {
                "trial_id": trials["id"][:],
                "start_time": trials["start_time"][:],
                "stop_time": trials["stop_time"][:],
                "trial_length": trials["trial_length"][:],
                "initial_image": decode_text(
                    trials["initial_image_name"][:]
                ),
                "change_image": decode_text(
                    trials["change_image_name"][:]
                ),
                "is_change": trials["is_change"][:].astype(int),
                "is_go": trials["go"][:].astype(int),
                "is_catch": trials["catch"][:].astype(int),
                "change_time": trials[
                    "change_time_no_display_delay"
                ][:],
                "response_time": trials["response_time"][:],
                "reward_time": trials["reward_time"][:],
                "reward_volume": trials["reward_volume"][:],
                "hit": trials["hit"][:].astype(int),
                "miss": trials["miss"][:].astype(int),
                "false_alarm": trials[
                    "false_alarm"
                ][:].astype(int),
                "correct_rejection": trials[
                    "correct_reject"
                ][:].astype(int),
                "aborted": trials["aborted"][:].astype(int),
                "auto_rewarded": trials[
                    "auto_rewarded"
                ][:].astype(int),
            }
        )

    mouse_trials["response_lick"] = (
        mouse_trials["hit"]
        + mouse_trials["false_alarm"]
    ).clip(upper=1)

    mouse_trials["correct"] = (
        mouse_trials["hit"]
        + mouse_trials["correct_rejection"]
    ).clip(upper=1)

    mouse_trials["outcome"] = np.select(
        [
            mouse_trials["hit"].eq(1),
            mouse_trials["miss"].eq(1),
            mouse_trials["false_alarm"].eq(1),
            mouse_trials["correct_rejection"].eq(1),
            mouse_trials["aborted"].eq(1),
        ],
        [
            "hit",
            "miss",
            "false_alarm",
            "correct_rejection",
            "aborted",
        ],
        default="other",
    )

    mouse_trials["reaction_time"] = (
        mouse_trials["response_time"]
        - mouse_trials["change_time"]
    )

    return mouse_trials


def prepare_valid_trials(mouse_trials: pd.DataFrame):
    """
    Exclude aborted and auto-rewarded trials.

    Returns
    -------
    valid_trials : pd.DataFrame
    """
    required_columns = {
        "aborted",
        "auto_rewarded",
        "outcome",
        "change_time",
    }

    missing = required_columns - set(mouse_trials.columns)

    if missing:
        raise ValueError(
            f"Missing required trial columns: {sorted(missing)}"
        )

    valid_trials = mouse_trials.loc[
        (mouse_trials["aborted"] == 0)
        & (mouse_trials["auto_rewarded"] == 0)
    ].copy()

    return valid_trials


def compute_signal_detection_metrics(
    n_hits: int,
    n_go_trials: int,
    n_false_alarms: int,
    n_catch_trials: int,
):
    """
    Compute raw and log-linear-corrected SDT measures.

    The correction prevents infinite z-scores when a rate is 0 or 1.
    """
    if n_go_trials <= 0 or n_catch_trials <= 0:
        raise ValueError(
            "Go and catch trial counts must both be positive."
        )

    hit_rate_raw = n_hits / n_go_trials
    false_alarm_rate_raw = (
        n_false_alarms / n_catch_trials
    )

    hit_rate_corrected = (
        (n_hits + 0.5) / (n_go_trials + 1)
    )

    false_alarm_rate_corrected = (
        (n_false_alarms + 0.5)
        / (n_catch_trials + 1)
    )

    z_hit = norm.ppf(hit_rate_corrected)
    z_false_alarm = norm.ppf(false_alarm_rate_corrected)

    d_prime = z_hit - z_false_alarm
    criterion = -0.5 * (z_hit + z_false_alarm)

    return {
        "n_go_trials": n_go_trials,
        "n_catch_trials": n_catch_trials,
        "n_hits": n_hits,
        "n_false_alarms": n_false_alarms,
        "hit_rate_raw": hit_rate_raw,
        "false_alarm_rate_raw": false_alarm_rate_raw,
        "hit_rate_corrected": hit_rate_corrected,
        "false_alarm_rate_corrected": (
            false_alarm_rate_corrected
        ),
        "d_prime": d_prime,
        "criterion": criterion,
    }


def summarize_mouse_behavior(
    valid_trials: pd.DataFrame,
    session_id: int | None = None,
):
    """
    Create a one-row mouse behavioral summary table.
    """
    hit_trials = valid_trials[
        valid_trials["outcome"] == "hit"
    ]

    miss_trials = valid_trials[
        valid_trials["outcome"] == "miss"
    ]

    false_alarm_trials = valid_trials[
        valid_trials["outcome"] == "false_alarm"
    ]

    correct_rejection_trials = valid_trials[
        valid_trials["outcome"] == "correct_rejection"
    ]

    metrics = compute_signal_detection_metrics(
        n_hits=len(hit_trials),
        n_go_trials=len(hit_trials) + len(miss_trials),
        n_false_alarms=len(false_alarm_trials),
        n_catch_trials=(
            len(false_alarm_trials)
            + len(correct_rejection_trials)
        ),
    )

    response_times = valid_trials.loc[
        valid_trials["response_lick"] == 1,
        "reaction_time",
    ].dropna()

    metrics["session_id"] = session_id
    metrics["n_valid_trials"] = len(valid_trials)
    metrics["median_reaction_time_s"] = (
        response_times.median()
    )

    return pd.DataFrame([metrics])


def summarize_rnn_behavior(
    rnn_trials: pd.DataFrame,
    model_name: str = "moderate_noise_gru",
):
    """
    Compute RNN behavioral SDT measures from a trial-level results table.
    """
    required_columns = {
        "is_change",
        "rnn_response_change",
    }

    missing = required_columns - set(rnn_trials.columns)

    if missing:
        raise ValueError(
            f"Missing RNN columns: {sorted(missing)}"
        )

    change_trials = rnn_trials[
        rnn_trials["is_change"] == 1
    ]

    same_trials = rnn_trials[
        rnn_trials["is_change"] == 0
    ]

    n_hits = int(
        change_trials["rnn_response_change"].sum()
    )

    n_false_alarms = int(
        same_trials["rnn_response_change"].sum()
    )

    metrics = compute_signal_detection_metrics(
        n_hits=n_hits,
        n_go_trials=len(change_trials),
        n_false_alarms=n_false_alarms,
        n_catch_trials=len(same_trials),
    )

    metrics["model"] = model_name

    return pd.DataFrame([metrics])


def find_threshold_for_false_alarm_rate(
    rnn_trials: pd.DataFrame,
    target_false_alarm_rate: float,
):
    """
    Find an RNN probability threshold matching a desired false-alarm rate.
    """
    if not 0 <= target_false_alarm_rate <= 1:
        raise ValueError(
            "target_false_alarm_rate must be between 0 and 1."
        )

    same_probabilities = rnn_trials.loc[
        rnn_trials["is_change"] == 0,
        "rnn_change_probability",
    ].to_numpy()

    return np.quantile(
        same_probabilities,
        1 - target_false_alarm_rate,
    )
