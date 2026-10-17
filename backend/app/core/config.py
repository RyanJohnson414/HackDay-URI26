"""Application settings loaded from backend/.env and the process environment."""

from functools import lru_cache
from pathlib import Path

import os
from dataclasses import dataclass, fields

BACKEND_DIR = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    gemini_api_key: str = ""
    gemini_model: str = "gemma-3-27b-it"
    embedding_model: str = "gemini-embedding-001"
    database_url: str = "sqlite:///./knowledge_graph.db"
    embedding_dimensions: int = 768

    def __post_init__(self) -> None:
        values: dict[str, str] = {}
        env_file = BACKEND_DIR / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip().upper()] = value.strip().strip("\"'")

        values.update({key.upper(): value for key, value in os.environ.items()})
        for field in fields(self):
            value = values.get(field.name.upper())
            if value is not None:
                setattr(
                    self,
                    field.name,
                    int(value) if field.name == "embedding_dimensions" else value,
                )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
