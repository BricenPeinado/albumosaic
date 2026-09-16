"""Tests for independent artwork resolution and provider priority."""

from __future__ import annotations

import logging
from json import dumps, loads
from pathlib import Path
from threading import Lock
from time import sleep
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest
from PIL import Image

from app.playlist import artwork_sources
from app.playlist.artwork import ArtworkCache, ArtworkDownloadError
from app.playlist.artwork_sources import (
    ArtworkReference,
    IndependentArtworkProvider,
    LocalManifestArtworkResolver,
    MusicBrainzArtworkResolver,
)
from app.playlist.models import Album
from app.playlist.progress import PreparationProgress, PreparationStage


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, object] | None = None,
        *,
        final_url: str = "https://archive.org/download/cover.jpg",
    ) -> None:
        self.payload = dumps(payload or {}).encode()
        self.final_url = final_url

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self.payload

    def geturl(self) -> str:
        return self.final_url


def album(name: str = "The Album", artist: str = "The Artist") -> Album:
    return Album(f"{artist}:{name}", name, (artist,), None, None)


def candidate(
    *,
    name: str = "The Album",
    artist: str = "The Artist",
    mbid: str = "release-group-id",
    score: int = 100,
) -> dict[str, object]:
    return {
        "id": mbid,
        "title": name,
        "artist-credit": [{"name": artist}],
        "primary-type": "Album",
        "score": score,
    }


def cover_payload(
    url: str = "https://coverartarchive.org/cover-500.jpg",
) -> dict[str, object]:
    return {
        "images": [
            {
                "front": True,
                "image": "https://coverartarchive.org/cover.jpg",
                "thumbnails": {"500": url},
            }
        ]
    }


def test_musicbrainz_exact_and_normalized_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_methods: list[str] = []
    cover_accept_headers: list[str | None] = []
    searches = iter(
        [
            {"release-groups": [candidate()]},
            {
                "release-groups": [
                    candidate(name="the—album", artist="THE ARTIST", mbid="normalized")
                ]
            },
        ]
    )

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        del timeout
        request_methods.append(request.get_method())  # type: ignore[attr-defined]
        if urlparse(request.full_url).hostname == "coverartarchive.org":  # type: ignore[attr-defined]
            cover_accept_headers.append(request.get_header("Accept"))  # type: ignore[attr-defined]
            return FakeResponse(cover_payload())
        return FakeResponse(next(searches))

    monkeypatch.setattr(artwork_sources, "open_url", fake_urlopen)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
        minimum_interval=0,
    )

    exact = resolver.resolve(album())
    normalized = resolver.resolve(album("The Album", "the artist"))

    assert exact is not None and exact.identifier == "release-group-id"
    assert normalized is not None and normalized.identifier == "normalized"
    assert request_methods == ["GET", "GET", "GET", "GET"]
    assert cover_accept_headers == ["application/json", "application/json"]


def test_ambiguous_musicbrainz_match_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "release-groups": [
            candidate(mbid="one"),
            candidate(mbid="two"),
        ]
    }
    monkeypatch.setattr(
        artwork_sources,
        "open_url",
        lambda request, timeout: FakeResponse(payload),
    )
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
        minimum_interval=0,
    )

    assert resolver.resolve(album()) is None


def test_missing_cover_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        del timeout
        if urlparse(request.full_url).hostname == "coverartarchive.org":  # type: ignore[attr-defined]
            raise HTTPError("url", 404, "missing", {}, None)
        return FakeResponse({"release-groups": [candidate()]})

    monkeypatch.setattr(artwork_sources, "open_url", fake_urlopen)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
        minimum_interval=0,
    )

    assert resolver.resolve(album()) is None


def test_interactive_remote_timeouts_are_separate(tmp_path: Path) -> None:
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
    )

    assert resolver.search_timeout == 7.0
    assert resolver.cover_art_timeout == 5.0
    assert ArtworkCache(tmp_path / "artwork-cache").timeout == 8.0


def test_musicbrainz_rate_limit_is_between_request_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    starts: list[float] = []

    def fake_open(request: object, timeout: float) -> FakeResponse:
        del request, timeout
        starts.append(clock[0])
        clock[0] += 0.25
        return FakeResponse({"release-groups": []})

    monkeypatch.setattr(artwork_sources, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        artwork_sources,
        "sleep",
        lambda delay: clock.__setitem__(0, clock[0] + delay),
    )
    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
        minimum_interval=1.0,
    )

    resolver._get_json("https://musicbrainz.org/ws/2/release-group/?query=one")
    clock[0] += 0.15
    resolver._get_json("https://musicbrainz.org/ws/2/release-group/?query=two")

    assert starts == [100.0, 101.0]


def test_local_manifest_takes_priority_and_partial_results_continue(
    tmp_path: Path,
) -> None:
    cover = tmp_path / "cover.png"
    Image.new("RGB", (16, 16), "blue").save(cover)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        dumps([{"artist": "The Artist", "album": "The Album", "path": "cover.png"}]),
        encoding="utf-8",
    )
    local = LocalManifestArtworkResolver(manifest)

    class RemoteResolver:
        calls = 0

        def resolve(self, target: Album) -> ArtworkReference | None:
            del target
            self.calls += 1
            return None

    remote = RemoteResolver()
    provider = IndependentArtworkProvider(
        (local, remote),
        ArtworkCache(tmp_path / "cache", allowed_hosts={"coverartarchive.org"}),
    )
    results = provider.get_many((album(), album("Missing")))

    assert len(results) == 1
    assert results[0].album.album_name == "The Album"
    assert results[0].path.is_file()
    assert remote.calls == 1


def test_spotify_image_hosts_are_rejected_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_call(request: object, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return FakeResponse()

    monkeypatch.setattr("app.playlist.artwork.open_url", forbidden_call)
    cache = ArtworkCache(tmp_path, allowed_hosts={"coverartarchive.org", "archive.org"})
    reference = ArtworkReference(
        "forbidden",
        "id",
        url="https://i.scdn.co/image/spotify-cover",
    )

    with pytest.raises(ArtworkDownloadError, match="not permitted"):
        cache.get(album(), reference)
    assert calls == 0


class FakeCache:
    def cached_path(
        self,
        target: Album,
        reference: ArtworkReference,
    ) -> Path | None:
        del target, reference
        return None

    def get(self, target: Album, reference: ArtworkReference) -> Path:
        del reference
        return Path(f"/{target.album_name}.png")


def test_duplicate_albums_are_resolved_once() -> None:
    calls: list[tuple[str, ...]] = []

    class Resolver:
        def resolve(self, target: Album) -> ArtworkReference:
            calls.append(target.identity_key)
            return ArtworkReference(
                "test", target.album_name, url="https://example.com/a"
            )

    first = album("One")
    duplicate = album("One")
    second = album("Two")
    provider = IndependentArtworkProvider((Resolver(),), FakeCache())  # type: ignore[arg-type]

    results = provider.get_many((first, duplicate, second))

    assert len(results) == 2
    assert calls.count(first.identity_key) == 1
    assert calls.count(second.identity_key) == 1


def test_musicbrainz_timeout_skips_one_album_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_open(request: object, timeout: float) -> FakeResponse:
        del timeout
        parsed = urlparse(request.full_url)  # type: ignore[attr-defined]
        if parsed.hostname == "coverartarchive.org":
            return FakeResponse(cover_payload())
        if "Slow" in request.full_url:  # type: ignore[attr-defined]
            raise TimeoutError
        name = "Good One" if "Good+One" in request.full_url else "Good Two"  # type: ignore[attr-defined]
        return FakeResponse({"release-groups": [candidate(name=name, mbid=name)]})

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
        minimum_interval=0,
    )
    provider = IndependentArtworkProvider((resolver,), FakeCache())  # type: ignore[arg-type]

    results = provider.get_many((album("Good One"), album("Slow"), album("Good Two")))

    assert [item.album.album_name for item in results] == ["Good One", "Good Two"]


def test_cover_art_timeout_skips_one_album_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_open(request: object, timeout: float) -> FakeResponse:
        del timeout
        url = request.full_url  # type: ignore[attr-defined]
        if "coverartarchive.org" in url:
            if "Bad" in url:
                raise TimeoutError
            return FakeResponse(cover_payload())
        name = "Bad" if "Bad" in url else "Good"
        return FakeResponse({"release-groups": [candidate(name=name, mbid=name)]})

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
        minimum_interval=0,
    )
    provider = IndependentArtworkProvider((resolver,), FakeCache())  # type: ignore[arg-type]

    results = provider.get_many((album("Good"), album("Bad")))

    assert [item.album.album_name for item in results] == ["Good"]


def test_musicbrainz_mapping_cache_avoids_all_second_pass_lookups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_open(request: object, timeout: float) -> FakeResponse:
        nonlocal calls
        del timeout
        calls += 1
        if urlparse(request.full_url).hostname == "coverartarchive.org":  # type: ignore[attr-defined]
            return FakeResponse(cover_payload())
        return FakeResponse({"release-groups": [candidate()]})

    mapping_path = tmp_path / "mapping.json"
    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    first = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )
    assert first.resolve(album()) is not None
    assert calls == 2

    second = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )
    monkeypatch.setattr(
        artwork_sources,
        "open_url",
        lambda request, timeout: pytest.fail("cached mapping performed network I/O"),
    )

    reference = second.resolve(album())

    assert reference is not None
    assert reference.url is not None and reference.url.startswith("https://")
    saved = loads(mapping_path.read_text(encoding="utf-8"))
    assert saved["version"] == 2
    assert (
        saved["covers"]["release-group-id"]
        == "https://coverartarchive.org/cover-500.jpg"
    )


def test_same_album_set_warm_run_has_no_remote_requests_or_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    albums = tuple(
        album(name) for name in ("Good A", "Good B", "Missing A", "Missing B")
    )
    counts = {"mb": 0, "caa": 0, "downloads": 0}

    def fake_open(request: object, timeout: float) -> FakeResponse:
        del timeout
        parsed = urlparse(request.full_url)  # type: ignore[attr-defined]
        if parsed.hostname == "coverartarchive.org":
            counts["caa"] += 1
            return FakeResponse(cover_payload())
        counts["mb"] += 1
        query = parse_qs(parsed.query)["query"][0]
        matched_name = next(
            name
            for name in ("Good A", "Good B", "Missing A", "Missing B")
            if name in query
        )
        if matched_name.startswith("Missing"):
            return FakeResponse({"release-groups": []})
        return FakeResponse(
            {"release-groups": [candidate(name=matched_name, mbid=matched_name)]}
        )

    class CachedFiles:
        def __init__(self) -> None:
            self.paths: dict[str, Path] = {}

        def cached_path(
            self, target: Album, reference: ArtworkReference
        ) -> Path | None:
            del reference
            return self.paths.get(target.album_name)

        def get(self, target: Album, reference: ArtworkReference) -> Path:
            del reference
            counts["downloads"] += 1
            path = tmp_path / f"{target.album_name}.png"
            self.paths[target.album_name] = path
            return path

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    cache = CachedFiles()
    mapping_path = tmp_path / "mapping.json"
    first = IndependentArtworkProvider(
        (
            MusicBrainzArtworkResolver(
                "maintainer@example.com", mapping_path, minimum_interval=0
            ),
        ),
        cache,  # type: ignore[arg-type]
    )
    assert len(first.get_many(albums)) == 2
    assert counts == {"mb": 6, "caa": 2, "downloads": 2}

    second = IndependentArtworkProvider(
        (
            MusicBrainzArtworkResolver(
                "maintainer@example.com", mapping_path, minimum_interval=0
            ),
        ),
        cache,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        artwork_sources,
        "open_url",
        lambda request, timeout: pytest.fail("warm run performed network I/O"),
    )
    with caplog.at_level(logging.DEBUG, logger="app.playlist.artwork_sources"):
        assert len(second.get_many(albums)) == 2
    assert counts == {"mb": 6, "caa": 2, "downloads": 2}
    assert "MusicBrainz 0 searches" in caplog.text
    assert "CAA 0 metadata lookups" in caplog.text
    assert "artwork download/cache" in caplog.text


def test_negative_mapping_persists_and_fresh_resolver_uses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def no_results(request: object, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return FakeResponse({"release-groups": []})

    mapping_path = tmp_path / "mapping.json"
    monkeypatch.setattr(artwork_sources, "open_url", no_results)
    first = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )
    assert first.resolve(album()) is None
    assert calls == 2  # Exact and broader; no edition fallback is needed.
    saved = loads(mapping_path.read_text(encoding="utf-8"))
    entry = saved["entries"]["\x1f".join(album().identity_key)]
    assert entry["mbid"] is None
    assert entry["reason"] == "no_musicbrainz_results"
    assert isinstance(entry["match_context"], str)

    second = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )
    monkeypatch.setattr(
        artwork_sources,
        "open_url",
        lambda request, timeout: pytest.fail("cached negative performed network I/O"),
    )
    assert second.resolve(album()) is None


def test_unknown_cache_version_invalidates_negative_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        dumps(
            {
                "version": 1,
                "entries": {
                    "\x1f".join(album().identity_key): {
                        "mbid": None,
                        "reason": "low_confidence",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def fake_open(request: object, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return FakeResponse({"release-groups": []})

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )

    assert resolver.resolve(album()) is None
    assert calls == 2


def test_negative_cache_rechecks_when_matching_metadata_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping_path = tmp_path / "mapping.json"
    calls = 0

    def fake_open(request: object, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return FakeResponse({"release-groups": []})

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    first = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )
    assert first.resolve(Album("same-id", "The Album", ("The Artist",), None)) is None
    assert calls == 2

    richer = Album("same-id", "The Album", ("The Artist",), None, "2001")
    second = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )
    assert second.resolve(richer) is None
    assert calls == 4


def test_edition_fallback_uses_at_most_three_musicbrainz_searches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_open(request: object, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return FakeResponse({"release-groups": []})

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com", tmp_path / "mapping.json", minimum_interval=0
    )

    assert resolver.resolve(album("The Album (Remastered)")) is None
    assert calls == 3


def test_positive_mapping_without_cover_rechecks_only_caa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping_path = tmp_path / "mapping.json"

    def first_open(request: object, timeout: float) -> FakeResponse:
        del timeout
        if urlparse(request.full_url).hostname == "coverartarchive.org":  # type: ignore[attr-defined]
            raise HTTPError("url", 404, "missing", {}, None)
        return FakeResponse({"release-groups": [candidate()]})

    monkeypatch.setattr(artwork_sources, "open_url", first_open)
    first = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )
    assert first.resolve(album()) is None
    saved = loads(mapping_path.read_text(encoding="utf-8"))
    assert saved["entries"]["\x1f".join(album().identity_key)] == {
        "mbid": "release-group-id",
        "reason": "no_cover_art",
        "match_context": None,
    }

    hosts: list[str | None] = []

    def warm_open(request: object, timeout: float) -> FakeResponse:
        del timeout
        hosts.append(urlparse(request.full_url).hostname)  # type: ignore[attr-defined]
        return FakeResponse(cover_payload())

    monkeypatch.setattr(artwork_sources, "open_url", warm_open)
    second = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )
    assert second.resolve(album()) is not None
    assert hosts == ["coverartarchive.org"]


def test_deluxe_title_fallback_resolves_without_changing_album_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries: list[str] = []

    def fake_open(request: object, timeout: float) -> FakeResponse:
        del timeout
        parsed = urlparse(request.full_url)  # type: ignore[attr-defined]
        if parsed.hostname == "coverartarchive.org":
            return FakeResponse(cover_payload())
        query = parse_qs(parsed.query)["query"][0]
        queries.append(query)
        return FakeResponse(
            {"release-groups": [candidate(name="The Album")]}
            if len(queries) == 2
            else {"release-groups": []}
        )

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    target = album("The Album (Deluxe Edition)")
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com", tmp_path / "mapping.json", minimum_interval=0
    )
    reference = resolver.resolve(target)

    assert reference is not None
    assert target.album_name == "The Album (Deluxe Edition)"
    assert len(queries) == 2
    assert 'releasegroup:"The Album"' in queries[1]


@pytest.mark.parametrize(
    "suffix",
    [
        "(Deluxe)",
        "(Deluxe Edition)",
        "[Expanded Edition]",
        "(Remastered)",
        "- Remaster",
        "(25th Anniversary Edition)",
        "(Bonus Track Version)",
        "(Explicit)",
        "(Clean Version)",
    ],
)
def test_edition_suffix_is_only_removed_for_matching(suffix: str) -> None:
    title = f"The Album {suffix}"
    assert artwork_sources._edition_title(title) == "The Album"


def test_release_year_selects_correct_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older = candidate(mbid="older", score=99)
    newer = candidate(mbid="newer", score=100)
    older["first-release-date"] = "2001-04-03"
    newer["first-release-date"] = "2015-04-03"

    def fake_open(request: object, timeout: float) -> FakeResponse:
        del timeout
        if urlparse(request.full_url).hostname == "coverartarchive.org":  # type: ignore[attr-defined]
            return FakeResponse(cover_payload())
        return FakeResponse({"release-groups": [newer, older]})

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    target = Album("album-id", "The Album", ("The Artist",), None, "2001")
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com", tmp_path / "mapping.json", minimum_interval=0
    )
    reference = resolver.resolve(target)

    assert reference is not None and reference.identifier == "older"


def test_near_identical_editions_with_same_year_are_not_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = candidate(mbid="plain")
    deluxe = candidate(name="The Album (Deluxe Edition)", mbid="deluxe")
    plain["first-release-date"] = "2001"
    deluxe["first-release-date"] = "2001"

    def fake_open(request: object, timeout: float) -> FakeResponse:
        del timeout
        if urlparse(request.full_url).hostname == "coverartarchive.org":  # type: ignore[attr-defined]
            return FakeResponse(cover_payload())
        return FakeResponse({"release-groups": [plain, deluxe]})

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com", tmp_path / "mapping.json", minimum_interval=0
    )
    assert resolver.resolve(album()) is not None


def test_legacy_negative_mapping_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        dumps({"\x1f".join(album().identity_key): None}), encoding="utf-8"
    )
    calls = 0

    def fake_open(request: object, timeout: float) -> FakeResponse:
        nonlocal calls
        del timeout
        calls += 1
        if urlparse(request.full_url).hostname == "coverartarchive.org":  # type: ignore[attr-defined]
            return FakeResponse(cover_payload())
        return FakeResponse({"release-groups": [candidate()]})

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com", mapping_path, minimum_interval=0
    )

    assert resolver.resolve(album()) is not None
    assert calls == 2
    assert all(value is not None for value in loads(mapping_path.read_text()).values())


def test_cover_art_rejects_unexpected_redirect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_open(request: object, timeout: float) -> FakeResponse:
        del timeout
        if urlparse(request.full_url).hostname == "coverartarchive.org":  # type: ignore[attr-defined]
            return FakeResponse(cover_payload(), final_url="https://example.com/cover")
        return FakeResponse({"release-groups": [candidate()]})

    monkeypatch.setattr(artwork_sources, "open_url", fake_open)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com", tmp_path / "mapping.json", minimum_interval=0
    )

    with pytest.raises(artwork_sources.ArtworkResolutionError, match="redirect"):
        resolver.resolve(album())


def test_preparation_progress_is_incremental_and_downloads_are_concurrent() -> None:
    state_lock = Lock()
    active = 0
    maximum_active = 0

    class Resolver:
        def resolve(self, target: Album) -> ArtworkReference:
            return ArtworkReference(
                "test",
                target.album_name,
                url=f"https://example.com/{target.album_name}",
            )

    class ConcurrentCache(FakeCache):
        def get(self, target: Album, reference: ArtworkReference) -> Path:
            nonlocal active, maximum_active
            del reference
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            sleep(0.02)
            with state_lock:
                active -= 1
            return Path(f"/{target.album_name}.png")

    albums = tuple(album(f"Album {index}") for index in range(20))
    progress: list[PreparationProgress] = []
    provider = IndependentArtworkProvider((Resolver(),), ConcurrentCache())  # type: ignore[arg-type]

    results = provider.get_many(albums, progress.append)

    assert len(results) == 20
    assert 1 < maximum_active <= 4
    resolution_updates = [
        update
        for update in progress
        if update.stage is PreparationStage.RESOLVING_ARTWORK
    ]
    download_updates = [
        update
        for update in progress
        if update.stage is PreparationStage.DOWNLOADING_ARTWORK
    ]
    assert resolution_updates
    assert download_updates[-1].processed == download_updates[-1].total == 20


def test_second_pass_reports_artwork_cache_hit_without_redownload() -> None:
    class Resolver:
        def resolve(self, target: Album) -> ArtworkReference:
            return ArtworkReference(
                "test",
                target.album_name,
                url="https://example.com/cover.png",
            )

    class ReusableCache(FakeCache):
        cached: Path | None = None
        downloads = 0

        def cached_path(
            self,
            target: Album,
            reference: ArtworkReference,
        ) -> Path | None:
            del target, reference
            return self.cached

        def get(self, target: Album, reference: ArtworkReference) -> Path:
            del target, reference
            self.downloads += 1
            self.cached = Path("/cached.png")
            return self.cached

    cache = ReusableCache()
    provider = IndependentArtworkProvider((Resolver(),), cache)  # type: ignore[arg-type]

    provider.get_many((album(),))
    second_progress: list[PreparationProgress] = []
    provider.get_many((album(),), second_progress.append)

    assert cache.downloads == 1
    assert any("cache: HIT" in update.message for update in second_progress)
