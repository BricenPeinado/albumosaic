"""Tests for shared certificate-verified standard-library networking."""

from __future__ import annotations

import ssl
from json import dumps
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

from app import network
from app.playlist.spotify_auth import SpotifyOAuthConfig, SpotifyOAuthManager


class JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = dumps(payload).encode()

    def __enter__(self) -> JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def getcode(self) -> int:
        return 200

    def read(self) -> bytes:
        return self.payload


@pytest.fixture(autouse=True)
def clear_context_cache() -> None:
    network.verified_ssl_context.cache_clear()
    yield
    network.verified_ssl_context.cache_clear()


def test_verified_context_uses_certifi_and_not_unverified_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ssl.create_default_context()
    captured_cafile: list[str] = []
    unverified_calls = 0

    def fake_create_default_context(*, cafile: str) -> ssl.SSLContext:
        captured_cafile.append(cafile)
        return context

    def forbidden_unverified_context() -> ssl.SSLContext:
        nonlocal unverified_calls
        unverified_calls += 1
        raise AssertionError("Unverified TLS context must not be used")

    monkeypatch.setattr(network.certifi, "where", lambda: "/trusted/cacert.pem")
    monkeypatch.setattr(
        network.ssl, "create_default_context", fake_create_default_context
    )
    monkeypatch.setattr(
        network.ssl,
        "_create_unverified_context",
        forbidden_unverified_context,
    )

    assert network.verified_ssl_context() is context
    assert network.verified_ssl_context() is context
    assert captured_cafile == ["/trusted/cacert.pem"]
    assert unverified_calls == 0


def test_verified_context_keeps_certificate_and_hostname_checks_enabled() -> None:
    context = network.verified_ssl_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_open_url_passes_shared_context_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = network.verified_ssl_context()
    response = JsonResponse({})
    captured: dict[str, object] = {}

    def fake_urlopen(
        request: Request,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> JsonResponse:
        captured.update(request=request, timeout=timeout, context=context)
        return response

    monkeypatch.setattr(network, "urlopen", fake_urlopen)
    request = Request("https://example.com/")

    assert network.open_url(request, timeout=3.5) is response
    assert captured == {"request": request, "timeout": 3.5, "context": context}


def test_spotify_oauth_uses_shared_verified_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_context = network.verified_ssl_context()
    received_contexts: list[ssl.SSLContext] = []

    def fake_urlopen(
        request: Request,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> JsonResponse:
        del request, timeout
        received_contexts.append(context)
        return JsonResponse({"access_token": "access", "expires_in": 3600})

    monkeypatch.setattr(network, "urlopen", fake_urlopen)
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    query = parse_qs(urlparse(manager.begin_authorization()).query)

    manager.complete_callback(f"/spotify/callback?code=abc&state={query['state'][0]}")

    assert received_contexts == [shared_context]
