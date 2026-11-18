import time
import psycopg2
from db.populate_data import main as populate_db, inject_data_drift, DATABASE_URL
from workload.collect_plans import collect_all_plans, collect_single_plan
from workload.query_generator import generate_skewed_queries
from src.evaluate import run_full_evaluation, compute_q_error
from src.models import AI4DBBenchmarkSuite
import numpy as np


def evaluate_data_drift_sensitivity():
    """Tests sensitivity/brittleness under +20% skewed data drift without retraining."""
    print("[Sensitivity] Injecting +20% skewed data drift and testing model stability...")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        inject_data_drift(conn)
        suite = AI4DBBenchmarkSuite.load("models/ai4db_suite.pkl")
        rng = np.random.default_rng(77)
        drift_queries = generate_skewed_queries(rng, n_per_tier=15, unseen=True)
        q_b1, q_prop = [], []
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '15000';")
            for q in drift_queries:
                pdata = collect_single_plan(cur, q["sql"], execute_analyze=True)
                pred = suite.predict_single_query(pdata["plan_json"], q["sql"])
                actual = pdata["execution_time_ms"]
                q_b1.append(compute_q_error([actual], [pred["b1_native_pg"]["pred_time_ms"]])[0])
                q_prop.append(compute_q_error([actual], [pred["proposed_two_stage"]["pred_time_ms"]])[0])
        print(
            f"[Sensitivity] Post-Drift P90 Q-Error -> Native PG: {np.percentile(q_b1, 90):.2f} | "
            f"Proposed (No Retrain): {np.percentile(q_prop, 90):.2f}"
        )
    finally:
        conn.close()


def main():
    t_start = time.perf_counter()
    print("=== STEP 1: Populating Uniform & Skewed Datasets in PostgreSQL ===")
    populate_db()

    print("\n=== STEP 2: Generating Multi-Join Workloads & Collecting EXPLAIN ANALYZE Plans ===")
    collect_all_plans(n_train_per_tier=60, n_unseen_per_tier=25)

    print("\n=== STEP 3: Training Baselines + Two-Stage Model, Ablations & Statistical Tests ===")
    run_full_evaluation(seeds=[42, 101, 202, 303, 404])

    print("\n=== STEP 4: Running Data-Drift Sensitivity Analysis ===")
    evaluate_data_drift_sensitivity()

    elapsed = time.perf_counter() - t_start
    print(f"\n=== ALL STEPS COMPLETED SUCCESSFULLY IN {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()