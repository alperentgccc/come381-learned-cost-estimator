# COME 381 Research Project: Cardinality-Aware Learned Query Cost Estimation

## 1. Quick Start (Reproducible Execution)
1. Start PostgreSQL 16 and the FastAPI + Dashboard service:
   ```bash
   docker-compose up -d --build
   ```
2. Run the complete end-to-end experiment pipeline (populates `uniform_db` and `skewed_db`, collects `EXPLAIN ANALYZE` plans, trains Baselines B1–B3 & Proposed Two-Stage Model across 5 seeds, runs ablation & Wilcoxon statistical tests, and exports IEEE LaTeX tables & figures):
   ```bash
   docker exec -it come381_ai4db_app python run_all.py
   ```
3. Open the live web dashboard at:
   `http://localhost:8000`
4. Generated paper assets are saved automatically in `./results/`:
   - `results/table_baselines.tex`
   - `results/table_ablation.tex`
   - `results/fig_qerror_by_joins.png`
   - `results/fig_latency_tradeoff.png`
   - `results/metrics_summary.json`