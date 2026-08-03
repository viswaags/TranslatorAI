#!/usr/bin/env bash
set -euo pipefail

URL="${KIOSK_URL:-http://127.0.0.1:8080/}"
exec chromium \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --autoplay-policy=no-user-gesture-required \
  "${URL}"
