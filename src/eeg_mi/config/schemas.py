"""Configuration schemas and data validation dataclasses."""

from dataclasses import dataclass, field


@dataclass
class PreprocessingConfig:
    """Configuration options for EEG signal preprocessing."""

    sampling_rate: int = 160
    l_freq: float = 7.0
    h_freq: float = 30.0
    notch_freq: float = 60.0
    window_duration: float = 3.0
    window_overlap: float = 1.5


@dataclass
class SplitConfig:
    """Configuration options for subject-independent train/val/test splits."""

    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    exclude_subjects: list[int] = field(default_factory=list)
