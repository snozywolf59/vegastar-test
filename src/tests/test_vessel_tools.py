from dotenv import load_dotenv

from src.agents.tools.vessel_company_tools import search_vessels, get_vessel_details, get_vessel_ownership

load_dotenv()

result = search_vessels.invoke({
    # "ship_name": "EVER GIVEN",
    "mmsi": "563246800",
    "limit": 5,
})

print(result)


vessel_detail = get_vessel_details.invoke({
    "vessel_id": "016b4973-9118-70c5-a5ce-3696e0654b65"
})

print(vessel_detail)

vessel_ownship = get_vessel_ownership.invoke({
    "vessel_id": "016b4973-9118-70c5-a5ce-3696e0654b65",
    "role": "beneficial_owner"
})

print(vessel_ownship)