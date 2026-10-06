#!/usr/bin/env bash
# Start the API (loopback only) and the web gateway (public port 7860).
set -euo pipefail

umask 077
mkdir -p /tmp/data

# The gateway and API share a per-boot random token unless one is provided.
if [[ -z "${INTERNAL_API_TOKEN:-}" ]]; then
  INTERNAL_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
fi
export INTERNAL_API_TOKEN

if [[ "${AUTH_PROVIDER}" == "accesscode" ]]; then
  : "${ACCESS_CODE_SECRET:?Set the ACCESS_CODE_SECRET Space secret}"
  # The gateway validates the codes; the API admits code identities only from it.
  export ALLOWED_USERS="code:*"
  # Anonymous deployments keep prompt and document text out of traces.
  export TRACE_CONTENT=false
else
  : "${OIDC_CLIENT_ID:?Set the OIDC_CLIENT_ID Space secret}"
  : "${OIDC_CLIENT_SECRET:?Set the OIDC_CLIENT_SECRET Space secret}"
  : "${OIDC_COOKIE_SECRET:?Set the OIDC_COOKIE_SECRET Space secret}"
  : "${OIDC_REDIRECT_URI:?Set OIDC_REDIRECT_URI, e.g. https://<space>.hf.space/oauth2callback}"
fi

uvicorn investing_engine.api.main:app --host 127.0.0.1 --port 8000 --no-server-header &
api_pid=$!

cd ui
uvicorn gateway.app:create_app --factory --host 0.0.0.0 --port 7860 \
  --no-server-header --proxy-headers --forwarded-allow-ips="*" --no-access-log &
ui_pid=$!

# If either process exits, stop the container so the platform restarts it.
wait -n "$api_pid" "$ui_pid"
exit 1
