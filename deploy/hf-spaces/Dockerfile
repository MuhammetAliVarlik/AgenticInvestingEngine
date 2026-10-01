# Single-container build for Hugging Face Spaces (Docker SDK, free CPU tier).
# The API listens on loopback only; the Streamlit UI is the sole public port
# and signs users in with OIDC (Google) before the allowlist check.

FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    AUTH_MODE=trusted-proxy \
    AUTH_PROVIDER=oidc \
    LLM_PROVIDER=groq \
    ENABLE_YFINANCE=false \
    API_BASE_URL=http://127.0.0.1:8000 \
    DB_PATH=/tmp/data/predictions.db \
    MODEL_DIR=/tmp/data/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md LICENSE ./
COPY --chown=user src ./src
RUN pip install ".[observability]"

COPY --chown=user ui ./ui
RUN pip install -r ui/requirements.txt \
    && rm -f ui/.streamlit/secrets.toml

COPY --chown=user deploy/hf-spaces/start.sh ./start.sh
RUN chmod 0555 start.sh

USER user
EXPOSE 7860
CMD ["./start.sh"]
