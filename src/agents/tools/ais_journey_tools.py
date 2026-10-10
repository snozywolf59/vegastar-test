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
from src.agents.tools.vessel_company_tools import VESSEL_COMPANY_TOOLS


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
        """SELECT vessel_id, event_ts, lat, lon, speed_knots, course_deg,
                  heading_deg, nav_status,
                  abs(extract(epoch FROM (event_ts - %s::timestamptz))) AS offset_seconds
           FROM ais_positions WHERE vessel_id = %s
           ORDER BY abs(extract(epoch FROM (event_ts - %s::timestamptz))), event_ts DESC
           LIMIT 1""",
        (timestamp, vessel_id.strip(), timestamp),
    )
    return to_json({"found": bool(rows), "requested_time": timestamp,
                  "position": rows[0] if rows else None})


@tool(args_schema=VesselIdInput)
def get_vessel_last_position(vessel_id: str) -> str:
    """Get the latest AIS position available for a vessel."""
    rows = fetch_all(
        """SELECT vessel_id, event_ts, lat, lon, speed_knots, course_deg,
                  heading_deg, nav_status FROM ais_positions
           WHERE vessel_id = %s ORDER BY event_ts DESC LIMIT 1""",
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
        """WITH points AS (
             SELECT event_ts, lat, lon, speed_knots, geom,
                    lag(geom) OVER (ORDER BY event_ts, position_id) AS previous_geom
             FROM ais_positions WHERE vessel_id = %s
               AND event_ts >= %s AND event_ts < %s AND geom IS NOT NULL
           ), stats AS (
             SELECT count(*) AS point_count, min(event_ts) AS first_ts,
                    max(event_ts) AS last_ts,
                    (array_agg(lat ORDER BY event_ts, lat))[1] AS first_lat,
                    (array_agg(lon ORDER BY event_ts, lon))[1] AS first_lon,
                    (array_agg(lat ORDER BY event_ts DESC, lat DESC))[1] AS last_lat,
                    (array_agg(lon ORDER BY event_ts DESC, lon DESC))[1] AS last_lon,
                    avg(speed_knots) AS average_speed_knots,
                    sum(ST_Distance(geom::geography, previous_geom::geography)
                        / 1852.0) AS distance_nm
             FROM points
           ), route AS (
             SELECT ST_AsGeoJSON(ST_MakeLine(geom ORDER BY event_ts))::json AS geojson
             FROM ais_positions WHERE vessel_id = %s
               AND event_ts >= %s AND event_ts < %s AND geom IS NOT NULL
           ) SELECT stats.*, route.geojson FROM stats CROSS JOIN route""",
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
        f"""SELECT gap_id, vessel_id, mmsi, gap_start_ts, gap_end_ts,
                   gap_duration_seconds, distance_nm, implied_speed_knots,
                   start_lat, start_lon, end_lat, end_lon
            FROM dark_gaps {where_clause}
            ORDER BY {ordering}, gap_id LIMIT %s""",
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
        f"""WITH selected AS (
             SELECT p.vessel_id, min(p.event_ts) AS first_ts, max(p.event_ts) AS last_ts,
                    count(*) AS total_points
             FROM ais_positions p WHERE {' AND '.join(filters)}
             GROUP BY p.vessel_id ORDER BY p.vessel_id LIMIT %s
           ), sampled AS (
             SELECT s.vessel_id, s.first_ts, s.last_ts, s.total_points,
                    p.event_ts, p.geom,
                    row_number() OVER (PARTITION BY s.vessel_id ORDER BY p.event_ts) AS rn
             FROM selected s JOIN ais_positions p ON p.vessel_id = s.vessel_id
               AND p.event_ts >= %s AND p.event_ts < %s AND p.geom IS NOT NULL
           ) SELECT vessel_id, min(first_ts) AS first_ts, max(last_ts) AS last_ts,
                    max(total_points) AS total_points,
                    count(*) FILTER (WHERE rn <= %s) AS returned_points,
                    ST_AsGeoJSON(ST_MakeLine(geom ORDER BY event_ts)
                       FILTER (WHERE rn <= %s))::json AS geojson
             FROM sampled GROUP BY vessel_id ORDER BY vessel_id""",
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
VESSEL_TOOLS = VESSEL_COMPANY_TOOLS + AIS_JOURNEY_TOOLS
