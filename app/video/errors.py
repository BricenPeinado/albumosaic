"""Domain errors raised by Albumosaic video processing."""


class VideoProcessingError(RuntimeError):
    """Base class for video decoding and encoding failures."""


class CorruptVideoError(VideoProcessingError):
    """Raised when an input video cannot be decoded reliably."""


class UnsupportedCodecError(VideoProcessingError):
    """Raised when OpenCV cannot encode the requested output format."""


class VideoWriteError(VideoProcessingError):
    """Raised when an output frame is invalid or no frames are written."""


class FFmpegError(VideoProcessingError):
    """Raised when FFprobe or FFmpeg cannot complete an operation."""
