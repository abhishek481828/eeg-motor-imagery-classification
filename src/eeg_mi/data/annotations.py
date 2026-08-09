"""Annotation parser and run-specific event remapping for PhysioNet Motor Imagery."""

import mne


def parse_run_event_mapping(run_id: int) -> dict[str, int]:
    """Return explicit zero-indexed event mapping for specific PhysioNet run.

    For Runs 4, 8, 12 (Motor Imagery: Left Fist vs Right Fist):
        - T1: Left Fist -> 0
        - T2: Right Fist -> 1
        - T0: Rest -> ignored
    """
    if run_id in [4, 8, 12]:
        return {"T1": 0, "T2": 1}
    elif run_id in [3, 7, 11]:
        # Execution Left vs Right Fist
        return {"T1": 0, "T2": 1}
    elif run_id in [6, 10, 14]:
        # Motor Imagery Both Fists vs Both Feet
        return {"T1": 0, "T2": 1}
    else:
        # Default mapping
        return {"T1": 0, "T2": 1}


def extract_events_from_annotations(
    raw: mne.io.Raw, event_id_map: dict[str, int] | None = None
) -> tuple[mne.Annotations, dict[str, int]]:
    """Extract annotations and apply explicit zero-indexed event mapping."""
    annotations = raw.annotations
    if event_id_map is not None:
        event_id = {
            desc: code for desc, code in event_id_map.items() if desc in annotations.description
        }
    else:
        unique_descriptions = sorted(set(annotations.description))
        event_id = {desc: idx for idx, desc in enumerate(unique_descriptions)}
    return annotations, event_id
