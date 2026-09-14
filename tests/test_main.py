"""Tests for the application command-line entry point."""

from app.main import parse_args


def test_cli_defaults_to_local_server() -> None:
    args = parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.share is False
    assert args.debug is False


def test_cli_accepts_server_options() -> None:
    args = parse_args(["--host", "0.0.0.0", "--port", "9000", "--share", "--debug"])

    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.share is True
    assert args.debug is True
