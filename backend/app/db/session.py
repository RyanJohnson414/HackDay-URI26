"""Database engine/session setup for the local SQLite application."""

from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.models import Base


def _engine_options(database_url: str) -> dict:
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        if url.database and url.database != ":memory:":
            Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return {"connect_args": {"check_same_thread": False}}
    return {}


DATABASE_URL = get_settings().database_url
engine = create_engine(DATABASE_URL, future=True, **_engine_options(DATABASE_URL))
if make_url(DATABASE_URL).get_backend_name() == "sqlite":
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def init_db() -> None:
    """Create any missing tables during FastAPI startup."""
    Base.metadata.create_all(bind=engine)
    if make_url(DATABASE_URL).get_backend_name() == "sqlite":
        columns = {column["name"] for column in inspect(engine).get_columns("documents")}
        with engine.begin() as connection:
            if "title" not in columns:
                connection.exec_driver_sql("ALTER TABLE documents ADD COLUMN title VARCHAR(512) NOT NULL DEFAULT ''")
                connection.exec_driver_sql("UPDATE documents SET title = filename WHERE title = ''")
            if "domain" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE documents ADD COLUMN domain VARCHAR(256) NOT NULL DEFAULT 'Uncategorized'"
                )
            if "tags" not in columns:
                connection.exec_driver_sql("ALTER TABLE documents ADD COLUMN tags JSON NOT NULL DEFAULT '[]'")


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
