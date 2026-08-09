"""Command Line Interface (CLI) for EEG Motor Imagery framework."""

import argparse
import sys


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Enhanced EEG Motor Imagery Classification CLI")
    parser.add_argument("--check-env", action="store_true", help="Run environment check")
    args = parser.parse_args()

    if args.check_env:
        from scripts.check_environment import main as check_main

        sys.exit(check_main())

    print("EEG Motor Imagery Framework CLI v0.1.0. Use --help to list options.")


if __name__ == "__main__":
    main()
