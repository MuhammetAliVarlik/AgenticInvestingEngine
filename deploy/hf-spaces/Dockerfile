# Single-container build for Hugging Face Spaces (Docker SDK, free CPU tier).
# The API listens on loopback only; the web gateway is the sole public port.
# Sign-in: anonymous access codes by default (AUTH_PROVIDER=accesscode), or
# Google OIDC with an allowlist (AUTH_PROVIDER=oidc).

# ---- Build the React single-page app --------------------------------------------
FROM node:26-alpine AS web
WORKDIR /web
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/tsconfig.json ui/vite.config.ts ui/index.html ./
COPY ui/public ./public
COPY ui/src ./src
RUN npm run build

# ---- API + gateway ------------------------------------------------------------------
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    AUTH_MODE=trusted-proxy \
    AUTH_PROVIDER=accesscode \
    LLM_PROVIDER=groq \
    ENABLE_YFINANCE=false \
    API_BASE_URL=http://127.0.0.1:8000 \
    STATIC_DIR=/home/user/app/ui/dist \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
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

COPY --chown=user ui/requirements.txt ./ui/requirements.txt
RUN pip install -r ui/requirements.txt
COPY --chown=user ui/gateway ./ui/gateway
COPY --from=web --chown=user /web/dist ./ui/dist

COPY --chown=user deploy/hf-spaces/start.sh ./start.sh
RUN chmod 0555 start.sh

USER user
EXPOSE 7860
CMD ["./start.sh"]
