#!/usr/bin/env bash
# Start the API (loopback only) and the Streamlit UI (public port 7860).
set -euo pipefail

umask 077
mkdir -p /tmp/data

# The UI and API share a per-boot random token unless one is provided.
if [[ -z "${INTERNAL_API_TOKEN:-}" ]]; then
  INTERNAL_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
fi
export INTERNAL_API_TOKEN

# Streamlit reads OIDC settings from secrets.toml; build it from Space secrets.
: "${OIDC_CLIENT_ID:?Set the OIDC_CLIENT_ID Space secret}"
: "${OIDC_CLIENT_SECRET:?Set the OIDC_CLIENT_SECRET Space secret}"
: "${OIDC_COOKIE_SECRET:?Set the OIDC_COOKIE_SECRET Space secret}"
: "${OIDC_REDIRECT_URI:?Set OIDC_REDIRECT_URI, e.g. https://<space>.hf.space/oauth2callback}"
cat > ui/.streamlit/secrets.toml <<EOF
[auth]
redirect_uri = "${OIDC_REDIRECT_URI}"
cookie_secret = "${OIDC_COOKIE_SECRET}"

[auth.google]
client_id = "${OIDC_CLIENT_ID}"
client_secret = "${OIDC_CLIENT_SECRET}"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
EOF

uvicorn investing_engine.api.main:app --host 127.0.0.1 --port 8000 --no-server-header &
api_pid=$!

cd ui
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=7860 --server.headless=true &
ui_pid=$!

# If either process exits, stop the container so the platform restarts it.
wait -n "$api_pid" "$ui_pid"
exit 1
