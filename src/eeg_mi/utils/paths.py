"""Project directory and path utilities."""

from pathlib import Path


def get_project_root() -> Path:
    """Return absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return Path object."""
    path.mkdir(parents=True, exist_ok=True)
    return path
