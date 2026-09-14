"""Tests for playlist-domain models and unique album selection."""

from app.playlist.models import Album, Playlist, Track, deduplicate_albums


def album(
    album_name: str,
    artists: tuple[str, ...],
    album_id: str | None = None,
) -> Album:
    return Album(
        album_id=album_id,
        album_name=album_name,
        artists=artists,
        artwork_url=f"https://images.example/{album_name}.jpg",
        source_url=f"https://music.example/{album_id or album_name}",
    )


def track(
    track_name: str,
    parent_album: Album,
    track_id: str | None,
) -> Track:
    return Track(
        track_id=track_id,
        track_name=track_name,
        artists=parent_album.artists,
        album=parent_album,
        source_url=f"https://music.example/track/{track_id or track_name}",
    )


def playlist(*tracks: Track) -> Playlist:
    return Playlist(
        playlist_id="playlist-1",
        playlist_name="Example Playlist",
        source_url="https://music.example/playlist/playlist-1",
        tracks=tracks,
    )


def test_multiple_songs_from_same_album_produce_one_album() -> None:
    first_copy = album("Discovery", ("Daft Punk",), album_id="album-1")
    second_copy = album("Discovery", ("Daft Punk",), album_id="album-1")

    result = playlist(
        track("One More Time", first_copy, "track-1"),
        track("Digital Love", second_copy, "track-2"),
    )

    assert result.albums == (first_copy,)
    assert result.unique_album_count == 1


def test_same_album_title_from_different_artists_stays_distinct() -> None:
    first = album("Home", ("Artist One",))
    second = album("Home", ("Artist Two",))

    result = playlist(
        track("First Song", first, "track-1"),
        track("Second Song", second, "track-2"),
    )

    assert result.albums == (first, second)
    assert result.unique_album_count == 2


def test_duplicate_tracks_do_not_duplicate_album() -> None:
    parent_album = album("Kind of Blue", ("Miles Davis",), album_id="album-1")
    duplicate = track("So What", parent_album, "track-1")

    result = playlist(duplicate, duplicate)

    assert len(result.tracks) == 2
    assert result.albums == (parent_album,)


def test_capitalization_and_whitespace_differences_are_normalized() -> None:
    first = album("The Miseducation", ("Lauryn Hill",))
    second = album("  THE   MISEDUCATION  ", ("LAURYN   HILL",))

    result = deduplicate_albums((first, second))

    assert result == (first,)


def test_stable_album_ids_take_precedence_over_names() -> None:
    first = album("Original Name", ("Original Artist",), album_id="stable-id")
    renamed = album("Deluxe Rename", ("Different Credit",), album_id="stable-id")

    assert deduplicate_albums((first, renamed)) == (first,)


def test_distinct_stable_ids_are_not_merged_by_matching_names() -> None:
    first = album("Album", ("Artist",), album_id="edition-1")
    second = album("Album", ("Artist",), album_id="edition-2")

    assert deduplicate_albums((first, second)) == (first, second)
