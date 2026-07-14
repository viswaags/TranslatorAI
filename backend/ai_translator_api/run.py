"""
run.py — Launch script for AI Translator API
=============================================

Run this from ANYWHERE inside the project:

    # From project root (ai-translator/):
    python backend/api/run.py

    # From backend/:
    python api/run.py

    # From backend/api/:
    python run.py

This script adds all the right directories to sys.path so every
import inside the API resolves correctly regardless of where you run it.
"""

import os
import sys
from pathlib import Path

# ── Resolve paths ─────────────────────────────────────────────────────────────
# run.py lives at: <project>/backend/api/run.py  (or wherever you placed it)
THIS_FILE   = Path(__file__).resolve()
API_DIR     = THIS_FILE.parent                # .../backend/api/
BACKEND_DIR = API_DIR.parent                  # .../backend/
PROJECT_DIR = BACKEND_DIR.parent              # .../ai-translator/

# Add all relevant roots so every `from services.xxx import` works
for p in [str(API_DIR), str(BACKEND_DIR), str(PROJECT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Start uvicorn ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # The ASGI app is main:app — relative to API_DIR which is in sys.path
    os.chdir(str(API_DIR))

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        reload_dirs=[str(API_DIR)],
        log_level="info",
    )
