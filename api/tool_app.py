"""Local-only FastAPI app for the segment-labeling tool. Completely separate
from api/main.py - no import in either direction - so the deployed inference
service can never be affected by this tool's SQLite writes or
`python -m src.train` subprocess spawning, and StaticFiles(directory="web")
(which raises at import time if web/ is missing) can never take down
api/main.py's own import.

Not part of the Docker image; run locally via:

    uvicorn api.tool_app:app --port 8001 --reload
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.labeling import router as labeling_router

app = FastAPI(title="s1000-diagnoser labeling & crawler tool")
app.include_router(labeling_router)
app.mount("/", StaticFiles(directory="web", html=True), name="web")
