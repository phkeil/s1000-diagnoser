"""Local-only FastAPI app for the labeling / training / diagnose tool.
Completely separate from api/main.py - no import in either direction - so the
deployed inference service can never be affected by this tool's SQLite writes or
`python -m src.train` subprocess spawning, and StaticFiles(directory="web")
(which raises at import time if web/ is missing) can never take down
api/main.py's own import.

Not part of the Docker image; run locally via:

    uvicorn api.tool_app:app --port 8001 --reload
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from api.inference_tab import router as diagnose_router
from api.labeling import router as labeling_router

app = FastAPI(title="s1000-diagnoser labeling & crawler tool")
app.include_router(labeling_router)
# Read-only diagnostic view over a trained model - mounted here rather than on
# api/main.py so the deployed inference service keeps a single, separately
# loaded model of its own (see api/inference_tab.py's docstring).
app.include_router(diagnose_router)


@app.middleware("http")
async def disable_caching(request: Request, call_next):
    """`uvicorn --reload` only restarts the process on a .py change - web/
    (html/css/js) is re-read from disk on every request regardless, so an
    edit there is always "live". Without this, though, the *browser* still
    caches those static responses on a normal reload and keeps showing the
    old version - this is a single-user local dev server, so there's no
    upside to letting it."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/", StaticFiles(directory="web", html=True), name="web")
