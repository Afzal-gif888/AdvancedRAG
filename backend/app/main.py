import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from advancerag_common.service import metrics_endpoint

from app.api.router import api_router
from app.api.routes import health
from app.core.config import settings
from app.core.logging import configure_logging
from app.db import session as db_session
from app.integrations.http import close_clients
from app.middleware.body_limit import BodySizeLimitMiddleware
from app.middleware.error_handler import register_exception_handlers
from app.middleware import rate_limit
from app.middleware.request_id import RequestContextMiddleware

logger = logging.getLogger("advancerag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    worker = None
    if settings.WORKER_ENABLED:
        from app.workers.runner import Worker

        worker = Worker()
        await worker.start()
    logger.info("startup", extra={"environment": settings.ENVIRONMENT})
    if settings.ENVIRONMENT in ("staging", "production") and not settings.email_configured:
        logger.warning("password_reset_email_disabled", extra={"hint": "set EMAIL_PROVIDER=resend with EMAIL_API_KEY and EMAIL_FROM"})
    try:
        yield
    finally:
        if worker:
            await worker.stop()
        await close_clients()
        backend = rate_limit._backend
        if hasattr(backend, "aclose"):
            await backend.aclose()
        if db_session.engine is not None:
            await db_session.engine.dispose()


def create_app() -> FastAPI:
    docs_enabled = not settings.is_production
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json" if docs_enabled else None,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    # Outermost first: request context wraps everything so every log line has a request id.
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    app.add_middleware(RequestContextMiddleware)

    # Prometheus-format metrics; requires the internal service token when one is configured.
    metrics_endpoint(app, settings.internal_token)
    app.include_router(health.router, tags=["health"])
    app.include_router(api_router, prefix=settings.API_V1_STR)
    return app


app = create_app()
