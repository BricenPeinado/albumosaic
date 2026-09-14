"""Command-line entry point for Albumosaic."""

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence

from app.ui.app import CSS, THEME, build_interface


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    """Parse local Gradio server options."""
    parser = ArgumentParser(description="Launch the Albumosaic Gradio app.")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Request a temporary public Gradio share link",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the local Albumosaic web interface."""
    args = parse_args(argv)
    interface = build_interface()
    interface.launch(
        show_error=True,
        theme=THEME,
        css=CSS,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
