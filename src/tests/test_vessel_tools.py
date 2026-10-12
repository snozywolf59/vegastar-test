import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.agents.tools._common import clamp_limit, parse_time
from src.agents.tools.ais_journey_tools import (
    get_vessel_dark_gaps,
    get_vessel_route,
    get_vessels_journeys,
)
from src.agents.tools.vessel_company_tools import search_vessels


class CommonQueryHelperTests(unittest.TestCase):
    def test_clamp_limit_keeps_value_in_safe_range(self) -> None:
        self.assertEqual(clamp_limit(-5), 1)
        self.assertEqual(clamp_limit(25), 25)
        self.assertEqual(clamp_limit(500), 100)

    def test_parse_time_requires_timezone(self) -> None:
        parsed = parse_time("2026-10-13T10:30:00Z")

        self.assertEqual(parsed, datetime(2026, 10, 13, 10, 30, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, "timezone"):
            parse_time("2026-10-13T10:30:00")


class VesselQueryToolTests(unittest.TestCase):
    @patch("src.agents.tools.vessel_company_tools.fetch_all")
    def test_search_vessels_builds_fuzzy_name_query(self, fetch_all_mock) -> None:
        fetch_all_mock.return_value = [
            {"vessel_id": "v-1", "shipname": "KOTA GAYA", "name_similarity": 0.91}
        ]

        result = json.loads(search_vessels.invoke({"ship_name": " kota gaya ", "limit": 500}))

        query, params = fetch_all_mock.call_args.args
        self.assertIn("shipname %% %s", query)
        self.assertNotIn("/*FILTERS*/", query)
        self.assertEqual(params, ("KOTA GAYA", "KOTA GAYA", "KOTA GAYA", 100))
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["limit"], 100)

    def test_search_vessels_rejects_empty_criteria(self) -> None:
        result = json.loads(search_vessels.invoke({}))

        self.assertEqual(result["error"], "missing_search_criteria")

    @patch("src.agents.tools.ais_journey_tools.fetch_all")
    def test_route_uses_same_time_range_for_statistics_and_geojson(self, fetch_all_mock) -> None:
        fetch_all_mock.return_value = [{
            "point_count": 2,
            "distance_nm": 12.5,
            "geojson": {"type": "LineString", "coordinates": [[106.0, 10.0], [107.0, 11.0]]},
        }]

        result = json.loads(get_vessel_route.invoke({
            "vessel_id": " vessel-1 ",
            "start_time": "2026-10-01T00:00:00Z",
            "end_time": "2026-10-02T00:00:00Z",
        }))

        _, params = fetch_all_mock.call_args.args
        self.assertEqual(params[0], "vessel-1")
        self.assertEqual(params[3], "vessel-1")
        self.assertEqual(params[1], params[4])
        self.assertEqual(params[2], params[5])
        self.assertEqual(result["point_count"], 2)
        self.assertEqual(result["geojson"]["type"], "LineString")

    @patch("src.agents.tools.ais_journey_tools.fetch_all")
    def test_dark_gap_query_applies_filters_and_longest_order(self, fetch_all_mock) -> None:
        fetch_all_mock.return_value = []

        result = json.loads(get_vessel_dark_gaps.invoke({
            "vessel_id": "v-1",
            "start_time": "2026-10-01T00:00:00Z",
            "end_time": "2026-10-05T00:00:00Z",
            "longest_first": True,
            "limit": 200,
        }))

        query, params = fetch_all_mock.call_args.args
        self.assertIn("WHERE vessel_id = %s AND gap_start_ts >= %s AND gap_start_ts < %s", query)
        self.assertIn("gap_duration_seconds DESC NULLS LAST", query)
        self.assertEqual(params[0], "v-1")
        self.assertEqual(params[-1], 100)
        self.assertEqual(result, {"count": 0, "limit": 100, "gaps": []})

    @patch("src.agents.tools.ais_journey_tools.fetch_all")
    def test_multi_vessel_journey_clamps_vessels_and_points(self, fetch_all_mock) -> None:
        fetch_all_mock.return_value = []

        result = json.loads(get_vessels_journeys.invoke({
            "start_time": "2026-10-01T00:00:00Z",
            "end_time": "2026-10-02T00:00:00Z",
            "company_name": "PIL",
            "company_role": "operator",
            "ship_type": "container",
            "limit": 500,
            "max_points_per_vessel": 5000,
        }))

        query, params = fetch_all_mock.call_args.args
        self.assertNotIn("/*FILTERS*/", query)
        self.assertIn("EXISTS (SELECT 1 FROM ownership", query)
        self.assertIn("EXISTS (SELECT 1 FROM vessels", query)
        self.assertEqual(params[-5], 100)
        self.assertEqual(params[-2:], (2000, 2000))
        self.assertEqual(result["limits"], {"vessels": 100, "points_per_vessel": 2000})


if __name__ == "__main__":
    unittest.main()
