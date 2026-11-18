import os
import json
import time
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.models import AI4DBBenchmarkSuite
from workload.collect_plans import collect_single_plan

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://come381:come381pass@localhost:5432/ai4db_bench"
)
MODEL_PATH = os.getenv("MODEL_PATH", "models/ai4db_suite.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "results/metrics_summary.json")

app = FastAPI(
    title="COME 381 AI4DB — Cardinality-Aware Query Cost Estimator",
    version="1.0.0",
)


class AnalyzeRequest(BaseModel):
    sql: str
    execute_actual: bool = True


def get_model() -> AI4DBBenchmarkSuite:
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `docker exec -it come381_ai4db_app python run_all.py` first.",
        )
    return AI4DBBenchmarkSuite.load(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health_check():
    db_ok = False
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "ok",
        "database_connected": db_ok,
        "model_loaded": os.path.exists(MODEL_PATH),
        "benchmark_ready": os.path.exists(METRICS_PATH),
    }


@app.get("/api/sample-queries")
def sample_queries():
    return {
        "samples": [
            {
                "title": "Skewed 4-Join Correlated Query (PostgreSQL Under-Estimation Case)",
                "sql": (
                    "SELECT u.tier, p.is_premium, COUNT(*), SUM(o.order_amount) "
                    "FROM skewed_db.users u "
                    "JOIN skewed_db.orders o ON u.user_id = o.user_id "
                    "JOIN skewed_db.order_items oi ON o.order_id = oi.order_id "
                    "JOIN skewed_db.products p ON oi.product_id = p.product_id "
                    "WHERE u.region_id = 1 AND u.tier = 'VIP' AND p.is_premium = 1 "
                    "GROUP BY u.tier, p.is_premium;"
                ),
            },
            {
                "title": "Uniform 6-Join Analytical Query (TPC-H Style)",
                "sql": (
                    "SELECT r.r_name, n.n_name, COUNT(*), SUM(l.l_extendedprice) "
                    "FROM uniform_db.region r "
                    "JOIN uniform_db.nation n ON r.r_regionkey = n.n_regionkey "
                    "JOIN uniform_db.customer c ON n.n_nationkey = c.c_nationkey "
                    "JOIN uniform_db.orders o ON c.c_custkey = o.o_custkey "
                    "JOIN uniform_db.lineitem l ON o.o_orderkey = l.l_orderkey "
                    "JOIN uniform_db.part p ON l.l_partkey = p.p_partkey "
                    "WHERE r.r_regionkey = 2 AND c.c_mktsegment = 'AUTOMOBILE' AND l.l_quantity <= 20 "
                    "GROUP BY r.r_name, n.n_name;"
                ),
            },
        ]
    }


@app.post("/api/analyze")
def analyze_query(req: AnalyzeRequest):
    sql_clean = req.sql.strip()
    if not sql_clean.upper().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only read-only SELECT queries are permitted.")

    suite = get_model()
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '15000';")
            plan_data = collect_single_plan(cur, sql_clean, execute_analyze=req.execute_actual)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"PostgreSQL EXPLAIN error: {exc}")
    finally:
        conn.close()

    t0 = time.perf_counter()
    predictions = suite.predict_single_query(plan_data["plan_json"], sql_clean)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "planning_time_ms": round(plan_data["planning_time_ms"], 3),
        "actual_execution_time_ms": (
            round(plan_data["execution_time_ms"], 3) if plan_data["execution_time_ms"] else None
        ),
        "model_inference_ms": round(inference_ms, 3),
        "predictions": predictions,
    }


@app.get("/api/benchmark-results")
def benchmark_results():
    if not os.path.exists(METRICS_PATH):
        return {"available": False}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "available": True,
        "summary_table": data["summary_table"],
        "ablation_summary": data["ablation_summary"],
        "system_overhead": data["system_overhead"],
    }