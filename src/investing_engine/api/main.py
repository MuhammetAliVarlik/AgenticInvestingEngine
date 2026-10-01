"""ASGI entry point: ``uvicorn investing_engine.api.main:app``."""

from investing_engine.api.app import create_app

app = create_app()
