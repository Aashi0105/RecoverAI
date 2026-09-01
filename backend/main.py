from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.config import settings
from backend.routes.recovery import router as recovery_router
from backend.routes.webhooks import router as webhook_router
from backend.routes.approvals import router as approvals_router
from backend.routes.demo import router as demo_router
from database.database import engine, Base
import database.models  # Ensures ORM models are registered


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure database tables exist on FastAPI startup
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[Warning] Could not initialize DB tables on startup: {e}")
    yield


app = FastAPI(
    title="RecoverAI — AI Revenue Recovery Agent",
    description="Automated ML-powered payment recovery system for Razorpay merchants.",
    version="0.1.0",
    lifespan=lifespan
)

# Register API routes
app.include_router(recovery_router, prefix="/api/v1/recovery", tags=["recovery"])
app.include_router(webhook_router, prefix="/api/v1/webhooks", tags=["webhooks"])
app.include_router(approvals_router, prefix="/api/v1/approvals", tags=["approvals"])
app.include_router(demo_router, prefix="/api/v1/demo", tags=["demo"])


@app.get("/health", tags=["system"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "service": "recover-ai",
        "version": "0.1.0",
        "env": settings.APP_ENV
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
