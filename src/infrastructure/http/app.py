from fastapi import FastAPI
from application.ports import ReportRepositoryPort, TraceRepositoryPort
from config import Settings
from infrastructure.http.routes_runs import RunsDeps, router as runs_router


def create_app(*, settings: Settings,
               report_repo: ReportRepositoryPort,
               trace_repo: TraceRepositoryPort) -> FastAPI:
    app = FastAPI(title="Receipt Processing Agent")
    deps = RunsDeps(settings=settings, report_repo=report_repo, trace_repo=trace_repo)
    app.state.runs_deps = deps
    app.include_router(runs_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "llm_mode": settings.llm_mode}

    return app
