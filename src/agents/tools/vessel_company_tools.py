
"""Tools for searching vessels and companies in the maritime database."""

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.agents.tools._common import (
    VesselIdInput,
    clamp_limit,
    fetch_all,
    to_json,
)
from src.app.resources import load_sql_section


class SearchVesselsInput(BaseModel):
    """Arguments for searching vessels."""

    ship_name: str | None = Field(default=None, description="Partial vessel name")
    mmsi: str | None = Field(default=None, description="Exact MMSI")
    imo: str | None = Field(default=None, description="Exact IMO")
    flag_code: str | None = Field(default=None, description="Flag country code")
    ship_type: str | None = Field(default=None, description="Vessel type")
    limit: int = Field(default=10, description="Maximum number of matches")


class VesselOwnershipInput(VesselIdInput):
    """Arguments for vessel ownership lookup."""

    role: str | None = Field(default=None, description="Exact ownership role")


class CompanyVesselsInput(BaseModel):
    """Arguments for finding vessels associated with a company."""

    company_name: str = Field(description="Full or partial company name")
    role: str | None = Field(default=None, description="Exact ownership role")
    limit: int = Field(default=20, description="Maximum number of vessels")


@tool(args_schema=SearchVesselsInput)
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
    params: list[object] = []

    if ship_name and ship_name.strip():
        filters.append("shipname %% %s")
        params.append((ship_name.strip().upper()))

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
        return to_json({
            "error": "missing_search_criteria",
            "message": "Provide at least one search criterion.",
        })
    limit = clamp_limit(limit)

    query = load_sql_section("vessel_tools.sql", "search_vessels").replace(
        "/*FILTERS*/", " AND ".join(filters)
    )
    if ship_name and ship_name.strip():
        params.append(ship_name.strip().upper())
        params.insert(0, ship_name.strip().upper())
    else:
        params.append(None)
        params.insert(0, None)
    params.append(limit)
    rows = fetch_all(query, tuple(params))
    return to_json({
        "count": len(rows),
        "limit": limit,
        "results": rows,
    })


@tool(args_schema=VesselIdInput)
def get_vessel_details(vessel_id: str) -> str:
    """Get the registered specifications of one vessel by vessel_id.

    Call search_vessels first if the vessel_id is not known.
    This tool does not retrieve the vessel's latest AIS position.
    """
    if not vessel_id.strip():
        return to_json({"error": "vessel_id is required"})

    query = load_sql_section("vessel_tools.sql", "vessel_details")

    rows = fetch_all(query, (vessel_id.strip(),))
    if not rows:
        return to_json({
            "found": False,
            "vessel_id": vessel_id,
            "message": "No vessel found for this vessel_id.",
        })

    return to_json({"found": True, "vessel": rows[0]})


@tool(args_schema=VesselOwnershipInput)
def get_vessel_ownership(
    vessel_id: str,
    role: str | None = None,
) -> str:
    """Get recorded ownership and management company associations.

    Optionally filter by an exact role value from the ownership table.
    Role input must be exactly one of those:
        "beneficial_owner", "commercial_manager", "ism_manager", "operator", "registered_owner", "technical_manager"
    """
    if not vessel_id.strip():
        return to_json({"error": "vessel_id is required"})

    query = load_sql_section("vessel_tools.sql", "vessel_ownership")

    normalized_role = role.strip() if role and role.strip() else None
    rows = fetch_all(
        query,
        (vessel_id.strip(), normalized_role, normalized_role),
    )

    return to_json({
        "vessel_id": vessel_id,
        "count": len(rows),
        "ownership_records": rows,
    })


@tool(args_schema=CompanyVesselsInput)
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
        return to_json({
            "error": "company_name is required",
        })

    limit = clamp_limit(limit)
    company_pattern = f"%{company_name.strip()}%"
    normalized_role = role.strip() if role and role.strip() else None

    query = load_sql_section("vessel_tools.sql", "vessels_by_company")

    rows = fetch_all(
        query,
        (
            company_pattern,
            company_pattern,
            normalized_role,
            normalized_role,
            limit,
        ),
    )

    return to_json({
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
