"""Gradio interface for Albumosaic."""

import gradio as gr


def generation_placeholder(
    playlist_url: str,
    video_path: str | None,
    tile_count: int,
) -> tuple[str, None]:
    """Explain the current scaffold state when Generate is selected."""
    del playlist_url, video_path, tile_count
    return "Generation is not implemented yet.", None


def build_interface() -> gr.Blocks:
    """Build the minimal Albumosaic interface without launching it."""
    with gr.Blocks(title="Albumosaic") as interface:
        gr.Markdown(
            "# Albumosaic\n"
            "Turn a video into a photomosaic made from Spotify album covers."
        )

        playlist_url = gr.Textbox(
            label="Spotify playlist URL",
            placeholder="https://open.spotify.com/playlist/...",
        )
        source_video = gr.Video(label="Source video")
        tile_count = gr.Slider(
            minimum=2,
            maximum=3,
            value=2,
            step=1,
            label="Album-cover tiles per frame",
            info="Enabled after playlist ingestion determines the album count.",
            interactive=False,
        )
        generate = gr.Button("Generate", variant="primary")
        progress = gr.Markdown("Ready.")
        result_video = gr.Video(label="Generated MP4")

        generate.click(
            fn=generation_placeholder,
            inputs=[playlist_url, source_video, tile_count],
            outputs=[progress, result_video],
        )

    return interface
