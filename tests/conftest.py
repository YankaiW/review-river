"""Shared pytest fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's environment and ``.env`` out of every test.

    Runs each test from an empty temporary directory, so the relative
    ``.env`` path in ``Settings`` finds nothing, and removes any
    ``REVIEW_RIVER_*`` variables set in the shell.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Changes the working directory and environment for one test only.
    tmp_path : Path
        Empty directory unique to the test.
    """
    monkeypatch.chdir(tmp_path)
    for name in os.environ:
        if name.startswith("REVIEW_RIVER_"):
            monkeypatch.delenv(name)
