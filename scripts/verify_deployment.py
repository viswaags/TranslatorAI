#!/usr/bin/env python3
"""Validate runtime packages, writable paths, and offline model artifacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.core.config import settings  # noqa: E402
from ai_translator_api.core.readiness import model_artifact_status  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    upload_dir = Path(settings.UPLOAD_DIR).expanduser()
    checks = {
        "python": {
            "ok": sys.version_info >= (3, 10),
            "value": platform.python_version(),
        },
        "architecture": {
            "ok": platform.machine().lower() in {"x86_64", "aarch64", "arm64"},
            "value": platform.machine(),
        },
        "upload_directory": {
            "ok": upload_dir.is_dir() and os.access(upload_dir, os.W_OK),
            "value": str(upload_dir),
        },
        "models": {
            name: status.to_dict()
            for name, status in model_artifact_status().items()
        },
    }
    ok = (
        checks["python"]["ok"]
        and checks["architecture"]["ok"]
        and checks["upload_directory"]["ok"]
        and all(item["available"] for item in checks["models"].values())
    )
    checks["ready"] = ok

    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        print(f"Python: {checks['python']['value']}")
        print(f"Architecture: {checks['architecture']['value']}")
        print(
            f"Upload directory: {checks['upload_directory']['value']} "
            f"({'ready' if checks['upload_directory']['ok'] else 'NOT WRITABLE'})"
        )
        for name, item in checks["models"].items():
            state = "ready" if item["available"] else "MISSING"
            print(f"{name}: {state} — {item['info']} ({item['location']})")
        print("Deployment readiness:", "READY" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
