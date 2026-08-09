"""Unit tests for Preprocessing Pipeline configuration and hashing (Phase 5)."""

from eeg_mi.preprocessing.pipeline import PreprocessingPipeline


def test_pipeline_config_hash() -> None:
    """Test deterministic hashing of preprocessing configuration."""
    pipe1 = PreprocessingPipeline(l_freq=7.0, h_freq=30.0, notch_freq=60.0)
    pipe2 = PreprocessingPipeline(l_freq=7.0, h_freq=30.0, notch_freq=60.0)
    pipe3 = PreprocessingPipeline(l_freq=8.0, h_freq=30.0, notch_freq=60.0)

    assert pipe1.config_hash == pipe2.config_hash
    assert pipe1.config_hash != pipe3.config_hash
    assert len(pipe1.config_hash) == 12
