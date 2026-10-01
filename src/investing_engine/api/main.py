"""ASGI entry point: ``uvicorn investing_engine.api.main:app``."""

from investing_engine.api.app import create_app
from investing_engine.config import get_settings
from investing_engine.observability.logging import configure_logging

_settings = get_settings()
configure_logging(_settings.log_level, json_output=_settings.json_logs)

app = create_app(_settings)
