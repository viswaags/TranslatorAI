#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
ARCH="$(uname -m)"

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "Python 3.10 or newer is required." >&2
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    python3-venv python3-dev build-essential \
    libgl1 libglib2.0-0 libgomp1 libopenblas0 libsndfile1 \
    chromium
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"

if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
  if [[ -z "${PADDLE_WHEEL:-}" ]]; then
    echo "PADDLE_WHEEL must point to a compatible Linux ARM64 PaddlePaddle wheel." >&2
    echo "The pinned PaddlePaddle 2.6.2 wheel availability is Python- and architecture-specific." >&2
    exit 1
  fi
  "${VENV_DIR}/bin/python" -m pip install "${PADDLE_WHEEL}"
fi

install -d "${PROJECT_DIR}/models" /var/tmp/ai-translator/uploads

if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
  sed "s|@PROJECT_DIR@|${PROJECT_DIR}|g" \
    "${PROJECT_DIR}/.env.example" > "${PROJECT_DIR}/.env"
  echo "Created ${PROJECT_DIR}/.env; update all model paths before startup."
fi

echo "Python installation complete."
echo "Provision local model artifacts, then run:"
echo "  ${VENV_DIR}/bin/python ${PROJECT_DIR}/scripts/verify_deployment.py"
