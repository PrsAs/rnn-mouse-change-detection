"""
Utilities for the RNN–mouse visual change-detection project.
"""

from .behavior import (
    compute_signal_detection_metrics,
    extract_mouse_trials,
    prepare_valid_trials,
)
from .decoding import (
    decode_hit_miss_population,
    permutation_test_decoder,
)
from .nwb_utils import (
    build_unit_metadata,
    extract_unit_trial_spike_counts,
    select_visp_units,
)
from .rnn_task import (
    ChangeDetectionGRU,
    generate_change_detection_trials,
)

__all__ = [
    "ChangeDetectionGRU",
    "build_unit_metadata",
    "compute_signal_detection_metrics",
    "decode_hit_miss_population",
    "extract_mouse_trials",
    "extract_unit_trial_spike_counts",
    "generate_change_detection_trials",
    "permutation_test_decoder",
    "prepare_valid_trials",
    "select_visp_units",
]
