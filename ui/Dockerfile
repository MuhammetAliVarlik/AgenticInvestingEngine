# ---- Build the React single-page app --------------------------------------------
FROM node:26-alpine AS web
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY tsconfig.json vite.config.ts index.html ./
COPY public ./public
COPY src ./src
RUN npm run build

# ---- Gateway: serves the app, signs users in, relays calls to the API ------------
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STATIC_DIR=/app/dist \
    GATEWAY_DB_PATH=/var/lib/gateway/gateway.db

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /var/lib/gateway && chown app:app /var/lib/gateway

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY gateway ./gateway
COPY --from=web /web/dist ./dist

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/healthz', timeout=4)"
CMD ["uvicorn", "gateway.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8501", "--no-server-header", "--proxy-headers", "--no-access-log"]
