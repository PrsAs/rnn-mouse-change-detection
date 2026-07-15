from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ChangeDetectionTaskConfig:
    sequence_length: int = 20
    reference_time: int = 2
    probe_time: int = 15
    stimulus_noise_std: float = 0.10
    change_probability: float = 0.50
    minimum_change: float = 0.20
    maximum_change: float = 1.00
    stimulus_minimum: float = -1.00
    stimulus_maximum: float = 1.00


class ChangeDetectionGRU(nn.Module):
    """
    GRU for delayed reference–probe change detection.

    Inputs have shape:
        [batch_size, sequence_length, 1]

    Outputs are logits for:
        0 = same
        1 = change
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        effective_dropout = (
            dropout if num_layers > 1 else 0.0
        )

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x, return_hidden_states=False):
        hidden_states, final_hidden = self.gru(x)
        final_state = hidden_states[:, -1, :]
        logits = self.classifier(final_state)

        if return_hidden_states:
            return logits, hidden_states, final_hidden

        return logits


def generate_change_detection_trials(
    n_trials: int,
    config: ChangeDetectionTaskConfig | None = None,
    random_seed: int | None = None,
):
    """
    Generate delayed reference–probe trials.

    The reference appears at `reference_time`; the probe appears at
    `probe_time`. A change trial has a nonzero probe-reference difference.

    Returns
    -------
    inputs : np.ndarray
        Shape [n_trials, sequence_length, 1].
    labels : np.ndarray
        0 for same trials and 1 for change trials.
    metadata : pandas.DataFrame
        Trial-level reference, probe, signed change, and label.
    """
    import pandas as pd

    if config is None:
        config = ChangeDetectionTaskConfig()

    rng = np.random.default_rng(random_seed)

    references = rng.uniform(
        config.stimulus_minimum,
        config.stimulus_maximum,
        size=n_trials,
    )

    is_change = (
        rng.random(n_trials) < config.change_probability
    )

    magnitudes = rng.uniform(
        config.minimum_change,
        config.maximum_change,
        size=n_trials,
    )

    directions = rng.choice(
        [-1.0, 1.0],
        size=n_trials,
    )

    signed_changes = (
        magnitudes * directions * is_change.astype(float)
    )

    probes = references + signed_changes

    probes = np.clip(
        probes,
        config.stimulus_minimum,
        config.stimulus_maximum,
    )

    actual_changes = probes - references

    inputs = np.zeros(
        (
            n_trials,
            config.sequence_length,
            1,
        ),
        dtype=np.float32,
    )

    reference_noise = rng.normal(
        0,
        config.stimulus_noise_std,
        size=n_trials,
    )

    probe_noise = rng.normal(
        0,
        config.stimulus_noise_std,
        size=n_trials,
    )

    inputs[
        :,
        config.reference_time,
        0,
    ] = references + reference_noise

    inputs[
        :,
        config.probe_time,
        0,
    ] = probes + probe_noise

    labels = is_change.astype(np.int64)

    metadata = pd.DataFrame(
        {
            "trial_id": np.arange(n_trials),
            "reference": references,
            "probe": probes,
            "signed_change": actual_changes,
            "absolute_change": np.abs(actual_changes),
            "is_change": labels,
        }
    )

    return inputs, labels, metadata


def make_torch_dataset(
    inputs: np.ndarray,
    labels: np.ndarray,
    device: str | torch.device = "cpu",
):
    """
    Convert NumPy task arrays into PyTorch tensors.
    """
    x_tensor = torch.tensor(
        inputs,
        dtype=torch.float32,
        device=device,
    )

    y_tensor = torch.tensor(
        labels,
        dtype=torch.long,
        device=device,
    )

    return x_tensor, y_tensor


@torch.no_grad()
def evaluate_rnn(
    model: nn.Module,
    inputs: np.ndarray,
    labels: np.ndarray,
    metadata,
    device: str | torch.device = "cpu",
):
    """
    Generate a trial-level RNN results table.
    """
    import pandas as pd

    model.eval()
    x_tensor, y_tensor = make_torch_dataset(
        inputs=inputs,
        labels=labels,
        device=device,
    )

    logits = model(x_tensor)
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    predictions = (probabilities >= 0.50).long()

    results = metadata.copy()

    results["rnn_change_probability"] = (
        probabilities.detach().cpu().numpy()
    )

    results["rnn_response_change"] = (
        predictions.detach().cpu().numpy()
    )

    results["rnn_correct"] = (
        predictions.eq(y_tensor)
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    results["rnn_outcome"] = np.select(
        [
            (results["is_change"] == 1)
            & (results["rnn_response_change"] == 1),
            (results["is_change"] == 1)
            & (results["rnn_response_change"] == 0),
            (results["is_change"] == 0)
            & (results["rnn_response_change"] == 1),
            (results["is_change"] == 0)
            & (results["rnn_response_change"] == 0),
        ],
        [
            "hit",
            "miss",
            "false_alarm",
            "correct_rejection",
        ],
        default="unknown",
    )

    return results
