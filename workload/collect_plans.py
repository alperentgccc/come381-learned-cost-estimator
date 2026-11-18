import os
import json
import time
import psycopg2
from workload.query_generator import build_full_workload

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://come381:come381pass@localhost:5432/ai4db_bench"
)
OUTPUT_PATH = os.getenv("PLANS_OUTPUT_PATH", "results/collected_plans.json")


def collect_single_plan(cur, sql: str, execute_analyze: bool = True) -> dict:
    clean_sql = sql.strip().rstrip(";")

    t0 = time.perf_counter()
    cur.execute(f"EXPLAIN (FORMAT JSON) {clean_sql};")
    explain_only = cur.fetchone()[0][0]
    pg_planning_ms = (time.perf_counter() - t0) * 1000.0

    if execute_analyze:
        cur.execute(f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {clean_sql};")
        analyzed = cur.fetchone()[0][0]
        return {
            "plan_json": analyzed["Plan"],
            "planning_time_ms": float(analyzed.get("Planning Time", pg_planning_ms)),
            "execution_time_ms": float(analyzed.get("Execution Time", 0.01)),
            "pure_explain_wall_ms": float(pg_planning_ms),
        }
    return {
        "plan_json": explain_only["Plan"],
        "planning_time_ms": float(pg_planning_ms),
        "execution_time_ms": None,
        "pure_explain_wall_ms": float(pg_planning_ms),
    }


def collect_all_plans(n_train_per_tier: int = 80, n_unseen_per_tier: int = 30) -> list[dict]:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    workload = build_full_workload(
        seed=42, n_train_per_tier=n_train_per_tier, n_unseen_per_tier=n_unseen_per_tier
    )
    print(f"[Workload] Collecting EXPLAIN ANALYZE plans for {len(workload)} queries...")

    conn = psycopg2.connect(DATABASE_URL)
    records = []
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '15000';")
            for item in workload:
                try:
                    plan_data = collect_single_plan(cur, item["sql"], execute_analyze=True)
                    records.append({**item, **plan_data})
                except Exception as exc:
                    conn.rollback()
                    print(f"[Warn] Skipped query {item['query_id']}: {exc}")
    finally:
        conn.close()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[Workload] Saved {len(records)} executed query plans to {OUTPUT_PATH}.")
    return records


if __name__ == "__main__":
    collect_all_plans()