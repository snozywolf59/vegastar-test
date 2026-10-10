"""Shared database and serialization helpers for maritime query tools."""

import json
import os
from datetime import datetime
import psycopg2
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor


class VesselIdInput(BaseModel):
    """Shared arguments requiring a database vessel identifier."""

    vessel_id: str = Field(description="Database vessel identifier")


def fetch_all(
    query: str,
    params: tuple[object, ...],
) -> list[dict[str, object]]:
    """Run a parameterized read-only query and return dictionary rows."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    with psycopg2.connect(database_url, connect_timeout=5) as connection:
        connection.set_session(readonly=True, autocommit=False)
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL statement_timeout = '10s'")
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


def to_json(data: object) -> str:
    """Serialize a tool result, converting database-specific values to text."""
    return json.dumps(data, ensure_ascii=False, default=str)


def clamp_limit(value: int, maximum: int = 100) -> int:
    """Clamp a requested result limit to a safe positive range."""
    return max(1, min(value, maximum))


def parse_time(value: str) -> datetime:
    """Parse an ISO-8601 timestamp that includes a timezone offset."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone, preferably UTC.")
    return parsed
