
"""Tools for searching vessels and companies in the maritime database."""

import json
import os
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_core.tools import tool


def _fetch_all(
    query: str,
    params: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Execute a parameterized read-only query and return dictionary rows."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    with psycopg2.connect(database_url, connect_timeout=5) as conn:
        # Defense in depth: reject accidental writes from these tools.
        conn.set_session(readonly=True, autocommit=False)

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL statement_timeout = '5s'")
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


def _json(data: Any) -> str:
    """Serialize tool results into JSON."""
    return json.dumps(data, ensure_ascii=False, default=str)


def _validate_limit(limit: int, maximum: int = 100) -> int:
    """Clamp result limits to a safe range."""
    return max(1, min(limit, maximum))


@tool
def search_vessels(
    ship_name: str | None = None,
    mmsi: str | None = None,
    imo: str | None = None,
    flag_code: str | None = None,
    ship_type: str | None = None,
    limit: int = 10,
) -> str:
    """Search vessels by ship name, MMSI, IMO, flag code, or ship type.

    Use this tool when the user identifies a vessel by name or other
    identifying attributes. Results include vessel_id for subsequent
    detail, ownership, and AIS queries.

    At least one search criterion is required.
    """
    filters: list[str] = []
    params: list[Any] = []

    if ship_name and ship_name.strip():
        filters.append("shipname ILIKE %s")
        params.append(f"%{ship_name.strip()}%")

    if mmsi and mmsi.strip():
        filters.append("mmsi = %s")
        params.append(mmsi.strip())

    if imo and imo.strip():
        filters.append("imo = %s")
        params.append(imo.strip())

    if flag_code and flag_code.strip():
        filters.append("flag_code ILIKE %s")
        params.append(flag_code.strip())

    if ship_type and ship_type.strip():
        filters.append(
            "(ship_type_summary ILIKE %s "
            "OR ship_type_detail_name ILIKE %s)"
        )
        value = f"%{ship_type.strip()}%"
        params.extend([value, value])

    if not filters:
        return _json({
            "error": "missing_search_criteria",
            "message": "Provide at least one search criterion.",
        })

    limit = _validate_limit(limit)
    params.append(limit)

    query = f"""
        SELECT
            vessel_id,
            mmsi,
            imo,
            shipname,
            callsign,
            flag_code,
            flag,
            ship_type_summary,
            ship_type_detail_name
        FROM vessels
        WHERE {" AND ".join(filters)}
        ORDER BY shipname NULLS LAST, vessel_id
        LIMIT %s
    """

    rows = _fetch_all(query, tuple(params))
    return _json({
        "count": len(rows),
        "limit": limit,
        "results": rows,
    })


@tool
def get_vessel_details(vessel_id: str) -> str:
    """Get the registered specifications of one vessel by vessel_id.

    Call search_vessels first if the vessel_id is not known.
    This tool does not retrieve the vessel's latest AIS position.
    """
    if not vessel_id.strip():
        return _json({"error": "vessel_id is required"})

    query = """
        SELECT
            vessel_id,
            mmsi,
            imo,
            callsign,
            shipname,
            flag_code,
            flag,
            ship_type_summary,
            ship_type_detail_name,
            length_m,
            width_m,
            dwt,
            grt,
            year_built
        FROM vessels
        WHERE vessel_id = %s
        LIMIT 1
    """

    rows = _fetch_all(query, (vessel_id.strip(),))
    if not rows:
        return _json({
            "found": False,
            "vessel_id": vessel_id,
            "message": "No vessel found for this vessel_id.",
        })

    return _json({"found": True, "vessel": rows[0]})


@tool
def get_vessel_ownership(
    vessel_id: str,
    role: str | None = None,
) -> str:
    """Get recorded ownership and management company associations.

    Optionally filter by an exact role value from the ownership table.
    Check the database's actual role values before using this filter.
    """
    if not vessel_id.strip():
        return _json({"error": "vessel_id is required"})

    query = """
        SELECT
            ownership_id,
            vessel_id,
            role,
            company_name,
            company_country,
            start_date
        FROM ownership
        WHERE vessel_id = %s
          AND (%s IS NULL OR role = %s)
        ORDER BY start_date DESC NULLS LAST, ownership_id DESC
        LIMIT 100
    """

    normalized_role = role.strip() if role and role.strip() else None
    rows = _fetch_all(
        query,
        (vessel_id.strip(), normalized_role, normalized_role),
    )

    return _json({
        "vessel_id": vessel_id,
        "count": len(rows),
        "ownership_records": rows,
    })


@tool
def search_vessels_by_company(
    company_name: str,
    role: str | None = None,
    limit: int = 20,
) -> str:
    """Find vessels associated with a company in the ownership table.

    Match company names partially. Optionally filter by an exact role
    value, such as a registered owner or operator, using the role values
    actually present in the database.
    """
    if not company_name.strip():
        return _json({
            "error": "company_name is required",
        })

    limit = _validate_limit(limit)
    company_pattern = f"%{company_name.strip()}%"
    normalized_role = role.strip() if role and role.strip() else None

    query = """
        SELECT DISTINCT
            v.vessel_id,
            v.mmsi,
            v.imo,
            v.shipname,
            v.flag,
            v.flag_code,
            v.ship_type_summary,
            v.ship_type_detail_name
        FROM vessels AS v
        JOIN ownership AS o
          ON o.vessel_id = v.vessel_id
        WHERE (
            o.company_name ILIKE %s
            OR o.company_name_norm ILIKE %s
        )
          AND (%s IS NULL OR o.role = %s)
        ORDER BY v.shipname NULLS LAST, v.vessel_id
        LIMIT %s
    """

    rows = _fetch_all(
        query,
        (
            company_pattern,
            company_pattern,
            normalized_role,
            normalized_role,
            limit,
        ),
    )

    return _json({
        "company_query": company_name,
        "role_filter": normalized_role,
        "count": len(rows),
        "limit": limit,
        "vessels": rows,
    })


VESSEL_COMPANY_TOOLS = [
    search_vessels,
    get_vessel_details,
    get_vessel_ownership,
    search_vessels_by_company,
]
