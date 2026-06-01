"""Project-wide constants."""

from __future__ import annotations

CHANNEL_NAMES: list[str] = [
    "Fz", "F3", "F4", "F7", "F8",
    "Cz", "C3", "C4",
    "T3", "T4",
    "Pz", "P3", "P4",
    "T5", "T6",
]

MONTAGE_RENAME_MAP: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

CLASS_NAMES: dict[int, str] = {
    0: "left_thigh_mi",
    1: "right_thigh_mi",
}

RAW_SFREQ: int = 1000
TARGET_SFREQ: int = 125
EPOCH_SAMPLES: int = 501
WINDOW_SIZE: int = 251
STEP_SIZE: int = 50
SEGMENTS_PER_TRIAL: int = 6
TRIAL_DURATION_SEC: float = 10.0
N_TRIALS_PER_SUBJECT: int = 150
TRAIN_RATIO: float = 0.70

PAPER_REFERENCE_METRICS: dict[str, dict[str, float]] = {
    "csp_svm": {"accuracy": 0.538, "kappa": 0.095, "sensitivity": 0.549},
    "eegnet": {"accuracy": 0.637, "kappa": 0.271, "sensitivity": 0.614},
    "convnet": {"accuracy": 0.650, "kappa": 0.305, "sensitivity": 0.623},
    "knn": {"accuracy": 0.596, "kappa": 0.206, "sensitivity": 0.569},
    "cnn_lstm": {"accuracy": 0.680, "kappa": 0.348, "sensitivity": 0.675},
    "proposed": {"accuracy": 0.721, "kappa": 0.436, "sensitivity": 0.699},
}
