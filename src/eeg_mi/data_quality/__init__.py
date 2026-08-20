"""Data quality subpackage for PhysioNet EEG Motor Imagery dataset auditing.

Public API
----------
RunStatus          : Enum of per-run quality classifications.
RunAuditRecord     : Dataclass holding every per-run audit metric.
AnnotationAudit    : Callable helper – wraps audit_single_run for convenience.
audit_single_run   : Low-level function that reads one EDF and returns a record.
audit_subject      : Iterates multiple runs for one subject directory.
classify_run_status: Applies predeclared rules to a RunAuditRecord → RunStatus.
PHYSIONET_RUN_PROTOCOL: Canonical task/run mapping table.
"""

from eeg_mi.data_quality.annotation_audit import (
    PHYSIONET_RUN_PROTOCOL,
    AnnotationAudit,
    RunAuditRecord,
    RunStatus,
    audit_single_run,
    audit_subject,
    classify_run_status,
)

__all__ = [
    "PHYSIONET_RUN_PROTOCOL",
    "AnnotationAudit",
    "RunAuditRecord",
    "RunStatus",
    "audit_single_run",
    "audit_subject",
    "classify_run_status",
]
