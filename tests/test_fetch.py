"""
Tests for nhisml.fetch — URL registry, path helpers, and download_file
(non-network parts only; actual HTTP calls are not made in the test suite).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nhisml.fetch import (
    NHIS_URLS,
    _default_zip_path,
    download_file,
    fetch_year,
)


# ---------------------------------------------------------------------------
# URL registry
# ---------------------------------------------------------------------------

class TestNhisUrls:
    def test_known_years_present(self):
        assert 2023 in NHIS_URLS
        assert 2024 in NHIS_URLS

    def test_urls_are_strings(self):
        for year, url in NHIS_URLS.items():
            assert isinstance(url, str), f"URL for {year} is not a string"

    def test_urls_end_with_zip(self):
        for year, url in NHIS_URLS.items():
            assert url.lower().endswith(".zip"), f"URL for {year} doesn't end with .zip"

    def test_urls_reference_cdc_gov(self):
        for year, url in NHIS_URLS.items():
            assert "cdc.gov" in url, f"URL for {year} doesn't point to cdc.gov"


# ---------------------------------------------------------------------------
# _default_zip_path
# ---------------------------------------------------------------------------

class TestDefaultZipPath:
    def test_year_2023(self):
        p = _default_zip_path("data", 2023)
        assert p == Path("data") / "raw" / "2023" / "adult23csv.zip"

    def test_year_2024(self):
        p = _default_zip_path("data", 2024)
        assert p == Path("data") / "raw" / "2024" / "adult24csv.zip"

    def test_custom_data_dir(self):
        p = _default_zip_path("/custom/dir", 2023)
        assert str(p).startswith("/custom/dir")

    def test_returns_path_object(self):
        p = _default_zip_path("data", 2023)
        assert isinstance(p, Path)

    def test_filename_contains_year_suffix(self):
        p = _default_zip_path("data", 2023)
        assert "23" in p.name


# ---------------------------------------------------------------------------
# fetch_year — error cases (no network)
# ---------------------------------------------------------------------------

class TestFetchYear:
    def test_unknown_year_no_url_raises(self):
        with pytest.raises(ValueError, match="No URL configured"):
            fetch_year(1990)  # year with no configured URL and no --url override

    def test_known_year_calls_download(self, tmp_path):
        """fetch_year should call download_file with the right URL."""
        with patch("nhisml.fetch.download_file") as mock_dl:
            mock_dl.return_value = tmp_path / "adult23csv.zip"
            fetch_year(2023, data_dir=str(tmp_path))
            mock_dl.assert_called_once()
            call_args = mock_dl.call_args
            assert "2023" in call_args[0][0]  # URL contains year

    def test_url_override_used(self, tmp_path):
        """When url= is given explicitly, it should be passed to download_file."""
        custom_url = "https://example.com/custom.zip"
        with patch("nhisml.fetch.download_file") as mock_dl:
            mock_dl.return_value = tmp_path / "custom.zip"
            fetch_year(2023, data_dir=str(tmp_path), url=custom_url)
            call_args = mock_dl.call_args
            assert call_args[0][0] == custom_url


# ---------------------------------------------------------------------------
# download_file — caching logic (no actual network)
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_uses_cache_when_exists(self, tmp_path):
        out_path = tmp_path / "file.zip"
        out_path.write_bytes(b"cached")
        result = download_file("https://example.com/file.zip", out_path, force=False)
        assert result == out_path
        # File should still contain the cached bytes (no download happened)
        assert out_path.read_bytes() == b"cached"

    def test_force_redownloads(self, tmp_path):
        """With force=True, the file should be downloaded even if it exists."""
        out_path = tmp_path / "file.zip"
        out_path.write_bytes(b"old_content")

        fake_content = b"new_content"
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Length": str(len(fake_content))}
        mock_response.iter_content = MagicMock(return_value=[fake_content])

        with patch("nhisml.fetch.requests.get", return_value=mock_response):
            result = download_file("https://example.com/file.zip", out_path, force=True)

        assert result == out_path
        assert out_path.read_bytes() == fake_content

    def test_creates_parent_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "file.zip"
        fake_content = b"data"

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Length": "4"}
        mock_response.iter_content = MagicMock(return_value=[fake_content])

        with patch("nhisml.fetch.requests.get", return_value=mock_response):
            download_file("https://example.com/file.zip", deep_path)

        assert deep_path.exists()
        assert deep_path.parent.is_dir()
