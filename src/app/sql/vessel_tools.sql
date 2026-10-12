-- name: vessel_details
SELECT vessel_id, mmsi, imo, callsign, shipname, flag_code, flag,
       ship_type_summary, ship_type_detail_name, length_m, width_m,
       dwt, grt, year_built
FROM vessels WHERE vessel_id = %s LIMIT 1

-- name: vessel_ownership
SELECT ownership_id, vessel_id, role, company_name, company_country, start_date
FROM ownership
WHERE vessel_id = %s AND (%s IS NULL OR role = %s)
ORDER BY start_date DESC NULLS LAST, ownership_id DESC LIMIT 100

-- name: vessels_by_company
SELECT DISTINCT v.vessel_id, v.mmsi, v.imo, v.shipname, v.flag,
       v.flag_code, v.ship_type_summary, v.ship_type_detail_name
FROM vessels AS v JOIN ownership AS o ON o.vessel_id = v.vessel_id
WHERE (o.company_name ILIKE %s OR o.company_name_norm ILIKE %s)
  AND (%s IS NULL OR o.role = %s)
ORDER BY v.shipname NULLS LAST, v.vessel_id LIMIT %s

-- name: vessel_position_near_time
SELECT vessel_id, event_ts, lat, lon, speed_knots, course_deg,
       heading_deg, nav_status,
       abs(extract(epoch FROM (event_ts - %s::timestamptz))) AS offset_seconds
FROM ais_positions WHERE vessel_id = %s
ORDER BY abs(extract(epoch FROM (event_ts - %s::timestamptz))), event_ts DESC
LIMIT 1

-- name: vessel_last_position
SELECT vessel_id, event_ts, lat, lon, speed_knots, course_deg,
       heading_deg, nav_status FROM ais_positions
WHERE vessel_id = %s ORDER BY event_ts DESC LIMIT 1

-- name: vessel_route
WITH points AS (
    SELECT event_ts, lat, lon, speed_knots, geom,
           lag(geom) OVER (ORDER BY event_ts, position_id) AS previous_geom
    FROM ais_positions WHERE vessel_id = %s
      AND event_ts >= %s AND event_ts < %s AND geom IS NOT NULL
), stats AS (
    SELECT count(*) AS point_count, min(event_ts) AS first_ts, max(event_ts) AS last_ts,
           (array_agg(lat ORDER BY event_ts, lat))[1] AS first_lat,
           (array_agg(lon ORDER BY event_ts, lon))[1] AS first_lon,
           (array_agg(lat ORDER BY event_ts DESC, lat DESC))[1] AS last_lat,
           (array_agg(lon ORDER BY event_ts DESC, lon DESC))[1] AS last_lon,
           avg(speed_knots) AS average_speed_knots,
           sum(ST_Distance(geom::geography, previous_geom::geography) / 1852.0) AS distance_nm
    FROM points
), route AS (
    SELECT ST_AsGeoJSON(ST_MakeLine(geom ORDER BY event_ts))::json AS geojson
    FROM ais_positions WHERE vessel_id = %s
      AND event_ts >= %s AND event_ts < %s AND geom IS NOT NULL
)
SELECT stats.*, route.geojson FROM stats CROSS JOIN route

-- name: search_vessels
SELECT vessel_id, mmsi, imo, shipname, callsign, flag_code, flag,
       ship_type_summary, ship_type_detail_name,
       similarity(shipname, %s) AS name_similarity
FROM vessels
WHERE /*FILTERS*/
ORDER BY similarity(shipname, %s) DESC NULLS LAST, shipname NULLS LAST, vessel_id
LIMIT %s

-- name: dark_gaps
SELECT gap_id, vessel_id, mmsi, gap_start_ts, gap_end_ts,
       gap_duration_seconds, distance_nm, implied_speed_knots,
       start_lat, start_lon, end_lat, end_lon
FROM dark_gaps /*WHERE_CLAUSE*/
ORDER BY /*ORDERING*/, gap_id LIMIT %s

-- name: vessel_journeys
WITH selected AS (
    SELECT p.vessel_id, min(p.event_ts) AS first_ts, max(p.event_ts) AS last_ts,
           count(*) AS total_points
    FROM ais_positions p WHERE /*FILTERS*/
    GROUP BY p.vessel_id ORDER BY p.vessel_id LIMIT %s
), sampled AS (
    SELECT s.vessel_id, s.first_ts, s.last_ts, s.total_points,
           p.event_ts, p.geom,
           row_number() OVER (PARTITION BY s.vessel_id ORDER BY p.event_ts) AS rn
    FROM selected s JOIN ais_positions p ON p.vessel_id = s.vessel_id
      AND p.event_ts >= %s AND p.event_ts < %s AND p.geom IS NOT NULL
)
SELECT vessel_id, min(first_ts) AS first_ts, max(last_ts) AS last_ts,
       max(total_points) AS total_points,
       count(*) FILTER (WHERE rn <= %s) AS returned_points,
       ST_AsGeoJSON(ST_MakeLine(geom ORDER BY event_ts)
           FILTER (WHERE rn <= %s))::json AS geojson
FROM sampled GROUP BY vessel_id ORDER BY vessel_id
