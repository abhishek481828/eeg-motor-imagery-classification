#!/usr/bin/env python3
"""Environment Verification Script for EEG Motor Imagery Classification.

Verifies system specifications, Python version, required ML/EEG libraries,
CUDA acceleration availability, and project directory structure.
"""

import importlib.util
import os
import platform
import sys
from pathlib import Path


def check_python_version() -> tuple[bool, str]:
    """Check if Python version is 3.11 or newer."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    passed = (major, minor) >= (3, 11)
    if passed:
        return True, f"Python {version_str} (>= 3.11)"
    return False, f"Python {version_str} (Required: >= 3.11)"


def check_package(package_name: str, min_version: str | None = None) -> tuple[bool, str]:
    """Check if a Python package is installed and optionally verify version."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False, "Not installed"

    try:
        mod = importlib.import_module(package_name)
        version = getattr(mod, "__version__", "Installed (unknown version)")
        return True, f"{package_name} {version}"
    except Exception as e:
        return False, f"Installed but failed to import: {e}"


def check_torch_and_cuda() -> tuple[bool, str, str]:
    """Check PyTorch installation and CUDA support."""
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            cuda_info = f"CUDA Available ({device_count} x {device_name})"
        else:
            cuda_info = "CUDA Not Available (CPU mode will be used)"
        return True, f"torch {torch_version}", cuda_info
    except ImportError:
        return False, "Not installed", "N/A"


def check_directories(base_dir: Path) -> list[tuple[str, bool]]:
    """Check required project directories."""
    required_dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/splits",
        "reports/figures",
        "reports/tables",
        "reports/experiments",
        "logs",
        "configs",
        "configs/data",
        "configs/experiments",
        "configs/splits",
        "src/eeg_mi",
        "tests",
    ]
    results = []
    for rel_path in required_dirs:
        dir_path = base_dir / rel_path
        results.append((rel_path, dir_path.exists() and dir_path.is_dir()))
    return results


def main() -> int:
    """Run all environment and dependency checks."""
    print("=" * 70)
    print("      EEG Motor Imagery Classification - Environment Check")
    print("=" * 70)

    # OS Information
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    print(f"\n[OS Platform]: {os_info}")
    print(
        f"[VS Code Detected]: {'TERM_PROGRAM' in os.environ and 'vscode' in os.environ['TERM_PROGRAM'].lower()}"
    )

    # Python Version
    py_pass, py_msg = check_python_version()
    status_symbol = "✓" if py_pass else "✗"
    print(f"[{status_symbol}] Python Version: {py_msg}")

    # PyTorch & CUDA
    torch_pass, torch_msg, cuda_msg = check_torch_and_cuda()
    t_symbol = "✓" if torch_pass else "✗"
    print(f"[{t_symbol}] Deep Learning: {torch_msg}")
    print(f"  └─ Hardware Device: {cuda_msg}")

    # Essential packages
    packages = [
        ("mne", "MNE-Python"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-Learn"),
        ("pywt", "PyWavelets"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("yaml", "PyYAML"),
        ("hydra", "Hydra-Core"),
        ("mlflow", "MLflow"),
    ]

    print("\n--- Package Verification ---")
    missing_packages = []
    for pkg_import, display_name in packages:
        passed, msg = check_package(pkg_import)
        symbol = "✓" if passed else "✗"
        print(f"[{symbol}] {display_name}: {msg}")
        if not passed:
            missing_packages.append(display_name)

    # Directory Structure
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dir_results = check_directories(project_root)

    print("\n--- Directory Structure ---")
    missing_dirs = []
    for rel_path, exists in dir_results:
        symbol = "✓" if exists else "✗"
        print(f"[{symbol}] {rel_path}")
        if not exists:
            missing_dirs.append(rel_path)

    # Hardware Requirements Summary
    print("\n--- Hardware Requirements & Recommendations ---")
    print(
        "  • CPU Mode: Fully supported for data processing, baseline ML, and small training runs."
    )
    print(
        "  • GPU (CUDA) Mode: Recommended for full CNN-LSTM model training and GAN data augmentation."
    )
    print("  • Minimum RAM: 8 GB recommended for streaming EDF file processing.")

    # Overall Summary
    print("\n" + "=" * 70)
    if py_pass and not missing_packages and not missing_dirs:
        print(" SUCCESS: All environment checks passed!")
        print("=" * 70)
        return 0
    else:
        print(" WARNING: Some requirements are missing or not configured yet.")
        if missing_packages:
            print(f" Missing packages: {', '.join(missing_packages)}")
            print(" Run `make install` or `uv pip install -e .` after project setup.")
        if missing_dirs:
            print(f" Missing directories: {', '.join(missing_dirs)}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
