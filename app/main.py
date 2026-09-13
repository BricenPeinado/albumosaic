"""Command-line entry point for Albumosaic."""

from app.ui.app import build_interface


def main() -> None:
    """Launch the local Albumosaic web interface."""
    interface = build_interface()
    interface.launch(show_error=True)


if __name__ == "__main__":
    main()
