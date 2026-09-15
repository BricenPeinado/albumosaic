"""Shared verified TLS configuration for standard-library HTTP requests."""

from __future__ import annotations

import ssl
from functools import lru_cache
from http.client import HTTPResponse
from typing import cast
from urllib.request import Request, urlopen

import certifi


@lru_cache(maxsize=1)
def verified_ssl_context() -> ssl.SSLContext:
    """Return a reusable CA-backed context with normal TLS verification enabled."""
    return ssl.create_default_context(cafile=certifi.where())


def open_url(request: Request, *, timeout: float) -> HTTPResponse:
    """Open a request using the shared verified TLS context."""
    return cast(
        HTTPResponse,
        urlopen(request, timeout=timeout, context=verified_ssl_context()),
    )
