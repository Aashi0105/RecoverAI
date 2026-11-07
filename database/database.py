import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import settings

# Resolve database URL from settings or environment
db_url = settings.DATABASE_URL or os.getenv("DATABASE_URL", "sqlite:///./recover_ai.db")

# Normalize legacy postgres:// URI scheme (Render / Heroku) to postgresql:// for SQLAlchemy 2.0+
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Configure SQLite or PostgreSQL connection arguments
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}
    # Neon and managed cloud PostgreSQL require SSL. Enforce sslmode=require if not explicitly specified and not local
    if "sslmode=" not in db_url and ("neon.tech" in db_url or ("localhost" not in db_url and "127.0.0.1" not in db_url)):
        connect_args["sslmode"] = "require"

engine = create_engine(
    db_url,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for providing database session to endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
