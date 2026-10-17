"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Also allow ``python backend/app/main.py`` from the repository root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # type: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[reportMissingImports]

from app.api.routes import discoveries, documents, graph
from app.db.session import init_db
from app.services.startup_sources import load_context_sources

logger = logging.getLogger(__name__)


async def _load_context_sources_in_background() -> None:
    try:
        report = await asyncio.to_thread(load_context_sources)
        logger.info(
            "Startup context sources finished: loaded=%d already_loaded=%d failed=%d",
            report.loaded,
            report.already_loaded,
            len(report.failed),
        )
    except Exception:
        logger.exception("Could not load startup context sources")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    source_task = asyncio.create_task(_load_context_sources_in_background())
    try:
        yield
    finally:
        if not source_task.done():
            source_task.cancel()


app = FastAPI(title="HaURI26 Knowledge Graph", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(documents.router, prefix="/api")
app.include_router(graph.router, prefix="/api")
app.include_router(discoveries.router, prefix="/api")


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn  # type: ignore[reportMissingImports]

    uvicorn.run(app, host="127.0.0.1", port=8000)
