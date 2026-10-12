"""LangChain tools for AIS positions, signal gaps, and vessel journeys."""

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.agents.tools._common import (
    VesselIdInput,
    clamp_limit,
    fetch_all,
    parse_time,
    to_json,
)
from src.app.resources import load_sql_section


class VesselPositionAtTimeInput(BaseModel):
    """Arguments for a time-specific AIS position lookup."""

    vessel_id: str = Field(description="Database vessel identifier")
    event_time: str = Field(description="ISO-8601 timestamp with timezone")


class VesselRouteInput(BaseModel):
    """Arguments for route summary and GeoJSON lookup."""

    vessel_id: str = Field(description="Database vessel identifier")
    start_time: str = Field(description="Inclusive ISO-8601 start timestamp")
    end_time: str = Field(description="Exclusive ISO-8601 end timestamp")


class DarkGapsInput(BaseModel):
    """Arguments for searching AIS signal gap events."""

    vessel_id: str | None = Field(default=None, description="Optional vessel identifier")
    start_time: str | None = Field(default=None, description="Inclusive ISO-8601 start")
    end_time: str | None = Field(default=None, description="Exclusive ISO-8601 end")
    longest_first: bool = Field(default=False, description="Sort by duration descending")
    limit: int = Field(default=20, description="Maximum number of gap records")


class JourneysInput(BaseModel):
    """Arguments for bounded multi-vessel journey retrieval."""

    start_time: str = Field(description="Inclusive ISO-8601 start timestamp")
    end_time: str = Field(description="Exclusive ISO-8601 end timestamp")
    company_name: str | None = Field(default=None, description="Optional company filter")
    company_role: str | None = Field(default=None, description="Optional exact company role")
    ship_type: str | None = Field(default=None, description="Optional vessel type filter")
    limit: int = Field(default=20, description="Maximum number of vessels")
    max_points_per_vessel: int = Field(default=500, description="Maximum map points per vessel")


@tool(args_schema=VesselPositionAtTimeInput)
def get_vessel_position_near_time(vessel_id: str, event_time: str) -> str:
    """Get the nearest reported AIS position to an ISO-8601 time and its offset."""
    timestamp = parse_time(event_time)
    rows = fetch_all(
        load_sql_section("vessel_tools.sql", "vessel_position_near_time"),
        (timestamp, vessel_id.strip(), timestamp),
    )
    return to_json({"found": bool(rows), "requested_time": timestamp,
                  "position": rows[0] if rows else None})


@tool(args_schema=VesselIdInput)
def get_vessel_last_position(vessel_id: str) -> str:
    """Get the latest AIS position available for a vessel."""
    rows = fetch_all(
        load_sql_section("vessel_tools.sql", "vessel_last_position"),
        (vessel_id.strip(),),
    )
    return to_json({"found": bool(rows), "position": rows[0] if rows else None})


@tool(args_schema=VesselRouteInput)
def get_vessel_route(vessel_id: str, start_time: str, end_time: str) -> str:
    """Summarize a vessel route and return GeoJSON for an ISO-8601 time range."""
    start, end = parse_time(start_time), parse_time(end_time)
    if end <= start:
        return to_json({"error": "end_time must be after start_time"})
    rows = fetch_all(
        load_sql_section("vessel_tools.sql", "vessel_route"),
        (vessel_id.strip(), start, end, vessel_id.strip(), start, end),
    )
    return to_json({"vessel_id": vessel_id, **rows[0]})


@tool(args_schema=DarkGapsInput)
def get_vessel_dark_gaps(
    vessel_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    longest_first: bool = False,
    limit: int = 20,
) -> str:
    """List AIS signal gaps, optionally filtering by vessel and UTC time range."""
    filters: list[str] = []
    params: list[object] = []
    if vessel_id and vessel_id.strip():
        filters.append("vessel_id = %s")
        params.append(vessel_id.strip())
    if start_time:
        filters.append("gap_start_ts >= %s")
        params.append(parse_time(start_time))
    if end_time:
        filters.append("gap_start_ts < %s")
        params.append(parse_time(end_time))
    result_limit = clamp_limit(limit)
    params.append(result_limit)
    ordering = (
        "gap_duration_seconds DESC NULLS LAST"
        if longest_first else "gap_start_ts DESC NULLS LAST"
    )
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    rows = fetch_all(
        load_sql_section("vessel_tools.sql", "dark_gaps")
        .replace("/*WHERE_CLAUSE*/", where_clause)
        .replace("/*ORDERING*/", ordering),
        tuple(params),
    )
    return to_json({"count": len(rows), "limit": result_limit, "gaps": rows})


@tool(args_schema=JourneysInput)
def get_vessels_journeys(
    start_time: str,
    end_time: str,
    company_name: str | None = None,
    company_role: str | None = None,
    ship_type: str | None = None,
    limit: int = 20,
    max_points_per_vessel: int = 500,
) -> str:
    """Return bounded per-vessel GeoJSON journeys for map display."""
    start, end = parse_time(start_time), parse_time(end_time)
    if end <= start:
        return to_json({"error": "end_time must be after start_time"})
    vessel_limit = clamp_limit(limit)
    point_limit = clamp_limit(max_points_per_vessel, 2000)
    filters = ["p.event_ts >= %s", "p.event_ts < %s"]
    params: list[object] = [start, end]
    if company_name and company_name.strip():
        filters.append(
            "EXISTS (SELECT 1 FROM ownership o WHERE o.vessel_id = p.vessel_id "
            "AND (o.company_name ILIKE %s OR o.company_name_norm ILIKE %s) "
            "AND (%s IS NULL OR o.role = %s))"
        )
        pattern = f"%{company_name.strip()}%"
        role = company_role.strip() if company_role and company_role.strip() else None
        params.extend((pattern, pattern, role, role))
    if ship_type and ship_type.strip():
        filters.append(
            "EXISTS (SELECT 1 FROM vessels v WHERE v.vessel_id = p.vessel_id "
            "AND (v.ship_type_summary ILIKE %s OR v.ship_type_detail_name ILIKE %s))"
        )
        pattern = f"%{ship_type.strip()}%"
        params.extend((pattern, pattern))
    params.append(vessel_limit)
    rows = fetch_all(
        load_sql_section("vessel_tools.sql", "vessel_journeys").replace(
            "/*FILTERS*/", " AND ".join(filters)
        ),
        tuple(params + [start, end, point_limit, point_limit]),
    )
    return to_json({"count": len(rows), "journeys": rows,
                  "limits": {"vessels": vessel_limit, "points_per_vessel": point_limit}})


AIS_JOURNEY_TOOLS = [
    get_vessel_position_near_time,
    get_vessel_last_position,
    get_vessel_route,
    get_vessel_dark_gaps,
    get_vessels_journeys,
]

