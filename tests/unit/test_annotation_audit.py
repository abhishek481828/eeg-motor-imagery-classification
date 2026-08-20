from eeg_mi.data_quality.annotation_audit import (
    PHYSIONET_RUN_PROTOCOL,
    RunAuditRecord,
    RunStatus,
    classify_run_status,
)


def test_run_status_enum_values() -> None:
    """Verify that all four required RunStatus values exist."""
    assert RunStatus.VALID == "VALID"
    assert RunStatus.VALID_WITH_WARNINGS == "VALID_WITH_WARNINGS"
    assert RunStatus.INVALID_FOR_BINARY_MI == "INVALID_FOR_BINARY_MI"
    assert RunStatus.CORRUPT_OR_UNREADABLE == "CORRUPT_OR_UNREADABLE"


def test_physionet_run_protocol_correct_mapping() -> None:
    """Verify protocol mapping for binary MI runs (R04, R08, R12)."""
    for run_id in [4, 8, 12]:
        info = PHYSIONET_RUN_PROTOCOL[run_id]
        assert info["use_for_binary_mi"] is True
        assert info["t1_label"] == "left_fist_imagery"
        assert info["t2_label"] == "right_fist_imagery"


def test_different_run_types_have_different_meanings() -> None:
    """Verify that execution (R03/07/11) and both fists/feet (R06/10/14) runs are marked false for binary MI."""
    for run_id in [3, 5, 6, 7, 9, 10, 11, 13, 14]:
        info = PHYSIONET_RUN_PROTOCOL[run_id]
        assert info["use_for_binary_mi"] is False


def test_audit_record_fields_complete() -> None:
    """Verify RunAuditRecord initialization and serialization."""
    rec = RunAuditRecord(
        subject_id="S001",
        run_id=4,
        run_label="R04",
        file_path="/tmp/S001R04.edf",
        split="train",
    )
    d = rec.to_dict()
    assert d["subject_id"] == "S001"
    assert d["run_id"] == 4
    assert d["run_label"] == "R04"
    assert "status" in d
    assert "usable_trial_count" in d


def test_classify_run_valid() -> None:
    """Mock record with standard 160Hz, 15 events -> VALID."""
    rec = RunAuditRecord(
        subject_id="S001",
        run_id=4,
        run_label="R04",
        file_path="/tmp/S001R04.edf",
        split="train",
        file_exists=True,
        readable=True,
        is_binary_mi_run=True,
        has_annotation_channel=True,
        n_t1_events=8,
        n_t2_events=7,
        sfreq=160.0,
        duration_seconds=125.0,
        n_eeg_channels=64,
    )
    classified = classify_run_status(rec)
    assert classified.status == RunStatus.VALID.value
    assert len(classified.warnings) == 0


def test_classify_run_invalid_zero_t1() -> None:
    """Mock record with T1=0 -> INVALID_FOR_BINARY_MI."""
    rec = RunAuditRecord(
        subject_id="S001",
        run_id=4,
        run_label="R04",
        file_path="/tmp/S001R04.edf",
        split="train",
        file_exists=True,
        readable=True,
        is_binary_mi_run=True,
        has_annotation_channel=True,
        n_t1_events=0,
        n_t2_events=7,
        sfreq=160.0,
        duration_seconds=125.0,
    )
    classified = classify_run_status(rec)
    assert classified.status == RunStatus.INVALID_FOR_BINARY_MI.value
    assert "Zero T1 events" in classified.classification_reason


def test_classify_run_invalid_zero_t2() -> None:
    """Mock record with T2=0 -> INVALID_FOR_BINARY_MI."""
    rec = RunAuditRecord(
        subject_id="S001",
        run_id=4,
        run_label="R04",
        file_path="/tmp/S001R04.edf",
        split="train",
        file_exists=True,
        readable=True,
        is_binary_mi_run=True,
        has_annotation_channel=True,
        n_t1_events=8,
        n_t2_events=0,
        sfreq=160.0,
        duration_seconds=125.0,
    )
    classified = classify_run_status(rec)
    assert classified.status == RunStatus.INVALID_FOR_BINARY_MI.value
    assert "Zero T2 events" in classified.classification_reason


def test_classify_run_warning_sfreq() -> None:
    """Mock record with 128Hz sfreq -> VALID_WITH_WARNINGS."""
    rec = RunAuditRecord(
        subject_id="S088",
        run_id=4,
        run_label="R04",
        file_path="/tmp/S088R04.edf",
        split="validation",
        file_exists=True,
        readable=True,
        is_binary_mi_run=True,
        has_annotation_channel=True,
        n_t1_events=10,
        n_t2_events=9,
        sfreq=128.0,
        duration_seconds=124.0,
        n_eeg_channels=64,
    )
    classified = classify_run_status(rec)
    assert classified.status == RunStatus.VALID_WITH_WARNINGS.value
    assert any("128" in w for w in classified.warnings)


def test_classify_run_warning_truncated() -> None:
    """Mock record with duration < 120s -> VALID_WITH_WARNINGS."""
    rec = RunAuditRecord(
        subject_id="S104",
        run_id=8,
        run_label="R08",
        file_path="/tmp/S104R08.edf",
        split="test",
        file_exists=True,
        readable=True,
        is_binary_mi_run=True,
        has_annotation_channel=True,
        n_t1_events=7,
        n_t2_events=6,
        sfreq=160.0,
        duration_seconds=106.0,
        n_eeg_channels=64,
    )
    classified = classify_run_status(rec)
    assert classified.status == RunStatus.VALID_WITH_WARNINGS.value
    assert any("truncated" in w.lower() or "duration" in w.lower() for w in classified.warnings)
