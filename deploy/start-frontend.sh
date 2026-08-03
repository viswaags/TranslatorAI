#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PROJECT_DIR}/venv/bin/python" -m http.server \
  8080 --bind 127.0.0.1 --directory "${PROJECT_DIR}/frontend"
