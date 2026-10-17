"""Database engine/session setup for the local SQLite application."""

from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import BACKEND_DIR, get_settings
from app.db.models import Base


def _resolve_database_url(database_url: str) -> str:
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database and url.database != ":memory:":
        database_path = Path(url.database).expanduser()
        if not database_path.is_absolute():
            database_path = BACKEND_DIR / database_path
        return url.set(database=str(database_path.resolve())).render_as_string(hide_password=False)
    return database_url


def _engine_options(database_url: str) -> dict:
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        if url.database and url.database != ":memory:":
            Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return {"connect_args": {"check_same_thread": False}}
    return {}


DATABASE_URL = _resolve_database_url(get_settings().database_url)
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
        connection_columns = {
            column["name"] for column in inspect(engine).get_columns("connections")
        }
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
            if "sections" not in columns:
                connection.exec_driver_sql("ALTER TABLE documents ADD COLUMN sections JSON NOT NULL DEFAULT '[]'")
            if "discipline_scope" not in connection_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE connections ADD COLUMN discipline_scope VARCHAR(32) NOT NULL DEFAULT 'unknown'"
                )
                connection.exec_driver_sql("""
                    UPDATE connections
                    SET discipline_scope = CASE
                        WHEN COALESCE((
                            SELECT lower(trim(domain))
                            FROM chunks JOIN documents ON documents.id = chunks.document_id
                            WHERE chunks.id = connections.source_chunk_id
                        ), 'uncategorized') = 'uncategorized'
                          OR COALESCE((
                            SELECT lower(trim(domain))
                            FROM chunks JOIN documents ON documents.id = chunks.document_id
                            WHERE chunks.id = connections.target_chunk_id
                        ), 'uncategorized') = 'uncategorized' THEN 'unknown'
                        WHEN (
                            SELECT lower(trim(domain))
                            FROM chunks JOIN documents ON documents.id = chunks.document_id
                            WHERE chunks.id = connections.source_chunk_id
                        ) = (
                            SELECT lower(trim(domain))
                            FROM chunks JOIN documents ON documents.id = chunks.document_id
                            WHERE chunks.id = connections.target_chunk_id
                        ) THEN 'within_discipline'
                        ELSE 'cross_discipline'
                    END
                """)
            if "semantic_distance" not in connection_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE connections ADD COLUMN semantic_distance VARCHAR(16) NOT NULL DEFAULT 'distant'"
                )
                connection.exec_driver_sql("""
                    UPDATE connections
                    SET semantic_distance = CASE
                        WHEN weight >= 0.80 THEN 'local'
                        WHEN weight >= 0.60 THEN 'related'
                        ELSE 'distant'
                    END
                """)


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
