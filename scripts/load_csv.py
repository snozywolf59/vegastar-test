import argparse
import csv
import os
import sys
import time
from dotenv import load_dotenv
from pathlib import Path
import psycopg2

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

SCHEMA_FILE = BASE_DIR / "schemas.sql"

TABLES = [
    ("vessels", "vessels.csv", 1000,
     {"vessel_id", "mmsi", "imo", "callsign", "shipname", "flag_code", "flag",
      "ship_type_summary", "ship_type_detail_name", "length_m", "width_m",
      "dwt", "grt", "year_built"}),
    ("ais_positions", "ais_positions.csv", 171073,
     {"vessel_id", "mmsi", "event_ts", "lat", "lon", "speed_knots",
      "course_deg", "heading_deg", "nav_status","reported_dest", "draught_m"}),
    ("dark_gaps", "dark_gaps.csv", 880,
     {"gap_id", "vessel_id", "mmsi", "gap_start_ts", "gap_end_ts",
      "gap_duration_seconds", "distance_nm", "implied_speed_knots",
      "start_lat", "start_lon", "end_lat", "end_lon"}),
    ("ownership", "ownership.csv", 4127,
     {"vessel_id", "role", "company_name", "company_country", "start_date"}),
]


def read_header(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [h.strip() for h in next(csv.reader(f))]


def load_table(cur, table, path, required):
    header = read_header(path)
    missing = required - set(header)
    if missing:
        sys.exit(f"[LỖI] {path} thiếu cột: {sorted(missing)}")
    extra = set(header) - required
    if extra:
        sys.exit(f"[LỖI] {path} có cột lạ chưa có trong schema: {sorted(extra)}")

    cols = ", ".join(header)
    sql = (f"COPY {table} ({cols}) FROM STDIN "
           f"WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        cur.copy_expert(sql, f)


def main():
    ap = argparse.ArgumentParser(description="Nạp 4 file CSV vào PostgreSQL/PostGIS")
    ap.add_argument("--data-dir", default="data", help="thư mục chứa CSV")
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL"),
                    help="chuỗi kết nối; mặc định lấy từ biến môi trường DATABASE_URL")
    args = ap.parse_args()

    if not args.dsn:
        sys.exit("Thiếu kết nối: đặt DATABASE_URL hoặc dùng --dsn")

    for _, fname, _, _ in TABLES:
        p = os.path.join(args.data_dir, fname)
        if not os.path.isfile(p):
            sys.exit(f"Không thấy file: {p}")

    conn = psycopg2.connect(args.dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                t0 = time.time()
                print("Tạo schema / chỉ mục (nếu chưa có)...")
                with SCHEMA_FILE.open("r", encoding="utf-8") as f:
                    cur.execute(f.read())

                print("Làm rỗng bảng để nạp lại...")
                cur.execute("TRUNCATE ais_positions, dark_gaps, ownership, vessels "
                            "RESTART IDENTITY")

                for table, fname, expected, required in TABLES:
                    t = time.time()
                    load_table(cur, table, os.path.join(args.data_dir, fname), required)
                    cur.execute(f"SELECT count(*) FROM {table}")
                    n = cur.fetchone()[0]
                    flag = "" if n == expected else f"  (!) kỳ vọng {expected:,}"
                    print(f"  {table:<14} {n:>9,} dòng  [{time.time() - t:.1f}s]{flag}")

                print(f"Nạp xong trong {time.time() - t0:.1f}s")

        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("ANALYZE vessels; ANALYZE ais_positions; "
                        "ANALYZE dark_gaps; ANALYZE ownership;")

            cur.execute("""
                SELECT
                  (SELECT count(*) FROM ais_positions p
                     WHERE NOT EXISTS (SELECT 1 FROM vessels v WHERE v.vessel_id = p.vessel_id)),
                  (SELECT count(*) FROM vessels WHERE shipname IS NULL OR btrim(shipname) = ''),
                  (SELECT count(*) FROM vessels v
                     WHERE NOT EXISTS (SELECT 1 FROM ownership o WHERE o.vessel_id = v.vessel_id))
            """)
            orphan, noname, noown = cur.fetchone()
            print("Kiểm tra nhanh:")
            print(f"  điểm AIS không có tàu trong vessels : {orphan:,}")
            print(f"  tàu tên rỗng                        : {noname:,}")
            print(f"  tàu không có thông tin chủ sở hữu   : {noown:,}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()