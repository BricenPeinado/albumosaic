"""Smoke tests for the initial Gradio interface."""

import gradio as gr

from app.ui.app import build_interface, generation_placeholder


def test_build_interface_returns_blocks() -> None:
    assert isinstance(build_interface(), gr.Blocks)


def test_generate_reports_placeholder_status() -> None:
    status, video = generation_placeholder("", None, 2)

    assert status == "Generation is not implemented yet."
    assert video is None
