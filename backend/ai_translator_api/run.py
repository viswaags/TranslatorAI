"""
run.py — Launch script for AI Translator API
=============================================

Run from the backend directory:

    python -m ai_translator_api.run
"""

import os

# ── Start uvicorn ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ai_translator_api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info",
    )
