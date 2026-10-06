from __future__ import annotations

import pytest

from investing_engine.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        db_path=str(tmp_path / "history.db"),
        model_dir=str(tmp_path / "models"),
        evds_api_key=None,
        enable_yfinance=False,
    )
