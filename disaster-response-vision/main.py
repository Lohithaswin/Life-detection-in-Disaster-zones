"""Backward-compatible entry point — delegates to the Typer CLI."""

from disaster_vision.cli import main

if __name__ == "__main__":
    main()
