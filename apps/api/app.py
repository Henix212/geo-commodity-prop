"""FastAPI app entrypoint for the geo-commodity dashboard."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from apps.api.routes import graph, portfolio, tensions

app = FastAPI(
    title="Geo Commodity Prop API",
    version="0.1.0",
    description="Graph / tensions / portfolio for the dashboard",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph.router, prefix="/api")
app.include_router(tensions.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
