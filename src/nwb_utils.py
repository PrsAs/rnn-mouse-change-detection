from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def decode_text(values):
    """
    Convert HDF5 byte strings to plain strings.
    """
    return np.array(
        [
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
            for value in values
        ]
    )


def build_unit_metadata(nwb_path: str | Path):
    """
    Build a lightweight unit metadata table.

    Reads unit and electrode metadata only. It never loads the full
    spike_times array, which can contain hundreds of millions of values.
    """
    nwb_path = Path(nwb_path)

    electrode_path = (
        "general/extracellular_ephys/electrodes"
    )

    with h5py.File(nwb_path, "r") as h5_file:
        units = h5_file["units"]
        electrodes = h5_file[electrode_path]

        electrode_table = pd.DataFrame(
            {
                "peak_channel_id": electrodes["id"][:],
                "structure": decode_text(
                    electrodes["location"][:]
                ),
                "probe_id": electrodes["probe_id"][:],
                "probe_channel_number": electrodes[
                    "probe_channel_number"
                ][:],
                "valid_channel": electrodes[
                    "valid_data"
                ][:].astype(int),
                "x": electrodes["x"][:],
                "y": electrodes["y"][:],
                "z": electrodes["z"][:],
            }
        )

        unit_metadata = pd.DataFrame(
            {
                "unit_id": units["id"][:],
                "cluster_id": units["cluster_id"][:],
                "peak_channel_id": units[
                    "peak_channel_id"
                ][:],
                "quality": decode_text(units["quality"][:]),
                "firing_rate_hz": units["firing_rate"][:],
                "presence_ratio": units[
                    "presence_ratio"
                ][:],
                "isi_violations": units[
                    "isi_violations"
                ][:],
                "amplitude_cutoff": units[
                    "amplitude_cutoff"
                ][:],
                "snr": units["snr"][:],
                "amplitude": units["amplitude"][:],
                "isolation_distance": units[
                    "isolation_distance"
                ][:],
                "d_prime_quality": units["d_prime"][:],
            }
        )

    unit_metadata = unit_metadata.merge(
        electrode_table,
        on="peak_channel_id",
        how="left",
        validate="many_to_one",
    )

    return unit_metadata


def select_visp_units(
    unit_metadata: pd.DataFrame,
    structure: str = "VISp",
    require_good_label: bool = True,
    presence_ratio_minimum: float = 0.90,
    isi_violations_maximum: float = 0.50,
    amplitude_cutoff_maximum: float = 0.10,
):
    """
    Select quality-controlled units from one anatomical structure.
    """
    required_columns = {
        "structure",
        "quality",
        "presence_ratio",
        "isi_violations",
        "amplitude_cutoff",
    }

    missing = required_columns - set(unit_metadata.columns)

    if missing:
        raise ValueError(
            f"Missing unit metadata columns: {sorted(missing)}"
        )

    passes_numeric_qc = (
        (
            unit_metadata["presence_ratio"]
            > presence_ratio_minimum
        )
        & (
            unit_metadata["isi_violations"]
            < isi_violations_maximum
        )
        & (
            unit_metadata["amplitude_cutoff"]
            < amplitude_cutoff_maximum
        )
    )

    selected = unit_metadata.loc[
        (unit_metadata["structure"] == structure)
        & passes_numeric_qc
    ].copy()

    selected["passes_numeric_qc"] = True

    if require_good_label:
        selected = selected.loc[
            selected["quality"] == "good"
        ].copy()

    return selected.reset_index(drop=True)


def count_spikes_in_windows(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    window: tuple[float, float],
):
    """
    Count spikes in one relative-time window for each event.

    Assumes spike_times are sorted.
    """
    window_start, window_end = window

    if window_end <= window_start:
        raise ValueError(
            "window end must be larger than window start."
        )

    left_edges = event_times + window_start
    right_edges = event_times + window_end

    left_indices = np.searchsorted(
        spike_times,
        left_edges,
        side="left",
    )

    right_indices = np.searchsorted(
        spike_times,
        right_edges,
        side="left",
    )

    return right_indices - left_indices


def get_unit_spike_times(
    spike_times_dataset,
    spike_times_index: np.ndarray,
    unit_row: int,
):
    """
    Retrieve spike times for one NWB unit-table row.

    The NWB file stores all unit spike times in one flattened array.
    spike_times_index gives the exclusive endpoint for each unit.
    """
    start_index = (
        0
        if unit_row == 0
        else int(spike_times_index[unit_row - 1])
    )

    end_index = int(spike_times_index[unit_row])

    return spike_times_dataset[start_index:end_index]


def extract_unit_trial_spike_counts(
    nwb_path: str | Path,
    units: pd.DataFrame,
    hit_trials: pd.DataFrame,
    miss_trials: pd.DataFrame,
    baseline_window: tuple[float, float] = (-0.50, 0.00),
    response_window: tuple[float, float] = (0.00, 0.50),
):
    """
    Extract baseline and response spike counts for hit and miss trials.

    Returns one row per unit per trial.
    """
    required_unit_columns = {"unit_id", "probe_id"}
    required_trial_columns = {"change_time"}

    missing_units = required_unit_columns - set(units.columns)
    missing_trials = required_trial_columns - set(
        hit_trials.columns
    )

    if missing_units:
        raise ValueError(
            f"Missing unit columns: {sorted(missing_units)}"
        )

    if missing_trials:
        raise ValueError(
            f"Missing trial columns: {sorted(missing_trials)}"
        )

    nwb_path = Path(nwb_path)

    hit_times = hit_trials["change_time"].to_numpy()
    miss_times = miss_trials["change_time"].to_numpy()

    rows = []

    with h5py.File(nwb_path, "r") as h5_file:
        units_group = h5_file["units"]

        all_unit_ids = units_group["id"][:]
        unit_id_to_row = {
            int(unit_id): row_number
            for row_number, unit_id in enumerate(all_unit_ids)
        }

        spike_times_dataset = units_group["spike_times"]
        spike_times_index = units_group[
            "spike_times_index"
        ][:]

        for _, unit in units.iterrows():
            unit_id = int(unit["unit_id"])
            unit_row = unit_id_to_row[unit_id]

            spike_times = get_unit_spike_times(
                spike_times_dataset=spike_times_dataset,
                spike_times_index=spike_times_index,
                unit_row=unit_row,
            )

            for outcome, event_times in [
                ("hit", hit_times),
                ("miss", miss_times),
            ]:
                baseline_counts = count_spikes_in_windows(
                    spike_times=spike_times,
                    event_times=event_times,
                    window=baseline_window,
                )

                response_counts = count_spikes_in_windows(
                    spike_times=spike_times,
                    event_times=event_times,
                    window=response_window,
                )

                for trial_number, (
                    baseline_count,
                    response_count,
                ) in enumerate(
                    zip(baseline_counts, response_counts)
                ):
                    rows.append(
                        {
                            "unit_id": unit_id,
                            "probe_id": int(unit["probe_id"]),
                            "structure": unit.get(
                                "structure",
                                "unknown",
                            ),
                            "outcome": outcome,
                            "outcome_trial_number": trial_number,
                            "baseline_spike_count": int(
                                baseline_count
                            ),
                            "response_spike_count": int(
                                response_count
                            ),
                            "baseline_rate_hz": (
                                baseline_count
                                / (
                                    baseline_window[1]
                                    - baseline_window[0]
                                )
                            ),
                            "response_rate_hz": (
                                response_count
                                / (
                                    response_window[1]
                                    - response_window[0]
                                )
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def summarize_unit_evoked_rates(
    spike_counts: pd.DataFrame,
):
    """
    Create per-unit hit and miss evoked-response summaries.
    """
    rates = (
        spike_counts
        .groupby(
            ["unit_id", "outcome"],
            as_index=False,
        )
        .agg(
            mean_baseline_rate_hz=(
                "baseline_rate_hz",
                "mean",
            ),
            mean_response_rate_hz=(
                "response_rate_hz",
                "mean",
            ),
            n_trials=(
                "outcome_trial_number",
                "nunique",
            ),
        )
    )

    rates["mean_evoked_rate_hz"] = (
        rates["mean_response_rate_hz"]
        - rates["mean_baseline_rate_hz"]
    )

    wide = (
        rates
        .pivot(
            index="unit_id",
            columns="outcome",
            values="mean_evoked_rate_hz",
        )
        .reset_index()
        .rename(
            columns={
                "hit": "hit_evoked_rate_hz",
                "miss": "miss_evoked_rate_hz",
            }
        )
    )

    wide["hit_minus_miss_hz"] = (
        wide["hit_evoked_rate_hz"]
        - wide["miss_evoked_rate_hz"]
    )

    return rates, wide
