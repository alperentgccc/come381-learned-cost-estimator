import os
import json
import time
import psutil
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import AI4DBBenchmarkSuite


def compute_q_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.maximum(np.asarray(y_true, dtype=np.float64), 1e-2)
    yp = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-2)
    return np.maximum(yt / yp, yp / yt)


def summarize_q_errors(q_errs: np.ndarray) -> dict:
    return {
        "p50": float(np.percentile(q_errs, 50)),
        "p90": float(np.percentile(q_errs, 90)),
        "p95": float(np.percentile(q_errs, 95)),
        "p99": float(np.percentile(q_errs, 99)),
        "mean": float(np.mean(q_errs)),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(n_boot)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    diff = x - y
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    stat, pval = wilcoxon(x, y, alternative="two-sided")
    return float(stat), float(pval)


def run_full_evaluation(
    plans_path: str = "results/collected_plans.json",
    results_dir: str = "results",
    seeds: list[int] = [42, 101, 202, 303, 404],
) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    with open(plans_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    train_records = [r for r in all_records if r["split"] == "train_seen"]
    test_records = [r for r in all_records if r["split"] == "test_unseen"]
    if not test_records:
        split_idx = int(0.8 * len(all_records))
        train_records, test_records = all_records[:split_idx], all_records[split_idx:]

    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    seed_runs = {m: [] for m in ["B1_NativePG", "B2_FlatRF", "B3_SingleStage", "Proposed_TwoStage"]}
    ablation_runs = {m: [] for m in ["Full", "w/o_Stage1_Card", "w/o_Hierarchy", "w/o_Predicates"]}

    best_suite = None
    last_eval_details = []

    for s in seeds:
        suite = AI4DBBenchmarkSuite(seed=s)
        suite.fit(train_records)
        if best_suite is None:
            best_suite = suite
            suite.save("models/ai4db_suite.pkl")

        y_true_ms = []
        preds = {m: [] for m in seed_runs}
        ab_preds = {m: [] for m in ablation_runs}
        inf_latencies_ms = []
        eval_details = []

        for r in test_records:
            t0 = time.perf_counter()
            out = suite.predict_single_query(r["plan_json"], r["sql"])
            inf_ms = (time.perf_counter() - t0) * 1000.0
            inf_latencies_ms.append(inf_ms)

            actual = float(r["execution_time_ms"])
            y_true_ms.append(actual)
            preds["B1_NativePG"].append(out["b1_native_pg"]["pred_time_ms"])
            preds["B2_FlatRF"].append(out["b2_flat_rf"]["pred_time_ms"])
            preds["B3_SingleStage"].append(out["b3_single_stage"]["pred_time_ms"])
            preds["Proposed_TwoStage"].append(out["proposed_two_stage"]["pred_time_ms"])

            ab_preds["Full"].append(out["proposed_two_stage"]["pred_time_ms"])
            ab_preds["w/o_Stage1_Card"].append(out["ablations"]["no_stage1"])
            ab_preds["w/o_Hierarchy"].append(out["ablations"]["no_hierarchy"])
            ab_preds["w/o_Predicates"].append(out["ablations"]["no_predicates"])

            eval_details.append(
                {
                    "dataset": r["dataset"],
                    "join_count": r["join_count"],
                    "actual_ms": actual,
                    "planning_ms": r["planning_time_ms"],
                    "inference_ms": inf_ms,
                    "q_B1": float(compute_q_error([actual], [out["b1_native_pg"]["pred_time_ms"]])[0]),
                    "q_B2": float(compute_q_error([actual], [out["b2_flat_rf"]["pred_time_ms"]])[0]),
                    "q_B3": float(compute_q_error([actual], [out["b3_single_stage"]["pred_time_ms"]])[0]),
                    "q_Prop": float(
                        compute_q_error([actual], [out["proposed_two_stage"]["pred_time_ms"]])[0]
                    ),
                }
            )

        y_true_arr = np.array(y_true_ms)
        for m in seed_runs:
            yp = np.array(preds[m])
            qe = compute_q_error(y_true_arr, yp)
            mae = float(np.mean(np.abs(y_true_arr - yp)))
            rmse = float(np.sqrt(np.mean((y_true_arr - yp) ** 2)))
            seed_runs[m].append(
                {
                    **summarize_q_errors(qe),
                    "mae_ms": mae,
                    "rmse_ms": rmse,
                    "q_errors": qe,
                    "mean_inf_ms": float(np.mean(inf_latencies_ms)),
                }
            )

        for m in ablation_runs:
            yp = np.array(ab_preds[m])
            qe = compute_q_error(y_true_arr, yp)
            ablation_runs[m].append(summarize_q_errors(qe))

        last_eval_details = eval_details

    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    model_size_kb = os.path.getsize("models/ai4db_suite.pkl") / 1024.0

    summary_table = {}
    prop_q_primary = seed_runs["Proposed_TwoStage"][0]["q_errors"]
    for m, runs in seed_runs.items():
        w_stat, p_val = (
            (0.0, 1.0)
            if m == "Proposed_TwoStage"
            else safe_wilcoxon(runs[0]["q_errors"], prop_q_primary)
        )
        ci_low, ci_high = bootstrap_ci(runs[0]["q_errors"])
        mean_inf = float(np.mean([x["mean_inf_ms"] for x in runs])) if m != "B1_NativePG" else 0.0
        summary_table[m] = {
            "p50_mean": float(np.mean([x["p50"] for x in runs])),
            "p50_std": float(np.std([x["p50"] for x in runs])),
            "p90_mean": float(np.mean([x["p90"] for x in runs])),
            "p95_mean": float(np.mean([x["p95"] for x in runs])),
            "p95_std": float(np.std([x["p95"] for x in runs])),
            "p99_mean": float(np.mean([x["p99"] for x in runs])),
            "mae_ms_mean": float(np.mean([x["mae_ms"] for x in runs])),
            "rmse_ms_mean": float(np.mean([x["rmse_ms"] for x in runs])),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_val,
            "inference_latency_ms": mean_inf,
            "throughput_qps": float(1000.0 / max(mean_inf, 0.05)) if mean_inf > 0 else 0.0,
        }

    ablation_summary = {}
    for m, runs in ablation_runs.items():
        ablation_summary[m] = {
            "p50_mean": float(np.mean([x["p50"] for x in runs])),
            "p90_mean": float(np.mean([x["p90"] for x in runs])),
            "p95_mean": float(np.mean([x["p95"] for x in runs])),
            "p99_mean": float(np.mean([x["p99"] for x in runs])),
        }

    system_overhead = {
        "memory_delta_mb": round(max(mem_after_mb - mem_before_mb, 1.0), 2),
        "model_storage_kb": round(model_size_kb, 2),
        "train_queries": len(train_records),
        "test_unseen_queries": len(test_records),
        "num_seeds": len(seeds),
    }

    full_report = {
        "summary_table": summary_table,
        "ablation_summary": ablation_summary,
        "system_overhead": system_overhead,
        "per_query_details": last_eval_details,
    }

    with open(os.path.join(results_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    export_latex_tables(summary_table, ablation_summary, results_dir)
    export_ieee_figures(last_eval_details, summary_table, results_dir)
    print(f"[Eval] Completed multi-seed evaluation, statistical tests, LaTeX tables, and figures in {results_dir}/.")
    return full_report


def export_latex_tables(summary_table: dict, ablation_summary: dict, results_dir: str):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Query Cost Estimation Performance Across Baselines and Proposed Method (Mean $\pm$ Std Across 5 Seeds)}",
        r"\label{tab:baselines}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Method} & \textbf{P50 Q-Err} & \textbf{P90 Q-Err} & \textbf{P95 Q-Err} & \textbf{MAE (ms)} & \textbf{Inf. (ms)} & \textbf{$p$-val} \\",
        r"\hline",
    ]
    labels = {
        "B1_NativePG": "B1: Native PostgreSQL",
        "B2_FlatRF": "B2: Flat RF Regressor",
        "B3_SingleStage": "B3: Single-Stage XGB",
        "Proposed_TwoStage": r"\textbf{Proposed Two-Stage}",
    }
    for k, label in labels.items():
        d = summary_table[k]
        pval_str = "---" if k == "Proposed_TwoStage" else f"{d['wilcoxon_p']:.2e}"
        lines.append(
            f"{label} & {d['p50_mean']:.2f}$\\pm${d['p50_std']:.2f} & {d['p90_mean']:.2f} & "
            f"{d['p95_mean']:.2f}$\\pm${d['p95_std']:.2f} & {d['mae_ms_mean']:.2f} & "
            f"{d['inference_latency_ms']:.2f} & {pval_str} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}"])
    with open(os.path.join(results_dir, "table_baselines.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    ab_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation Study: Impact of Component Removal on Unseen Workload Q-Error}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Configuration} & \textbf{P50} & \textbf{P90} & \textbf{P95} & \textbf{P99} \\",
        r"\hline",
    ]
    ab_labels = {
        "Full": r"\textbf{Full Proposed Method}",
        "w/o_Stage1_Card": r"$\rightarrow$ w/o Stage-1 Cardinality",
        "w/o_Hierarchy": r"$\rightarrow$ w/o Plan Hierarchy",
        "w/o_Predicates": r"$\rightarrow$ w/o SQL Predicates",
    }
    for k, label in ab_labels.items():
        d = ablation_summary[k]
        ab_lines.append(
            f"{label} & {d['p50_mean']:.2f} & {d['p90_mean']:.2f} & {d['p95_mean']:.2f} & {d['p99_mean']:.2f} \\\\"
        )
    ab_lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    with open(os.path.join(results_dir, "table_ablation.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(ab_lines))


def export_ieee_figures(details: list[dict], summary_table: dict, results_dir: str):
    df = pd.DataFrame(details)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
    joins = sorted(df["join_count"].unique())
    x = np.arange(len(joins))
    w = 0.2
    ax.bar(x - 1.5 * w, [df[df["join_count"] == j]["q_B1"].median() for j in joins], w, label="B1: Native PG")
    ax.bar(x - 0.5 * w, [df[df["join_count"] == j]["q_B2"].median() for j in joins], w, label="B2: Flat RF")
    ax.bar(x + 0.5 * w, [df[df["join_count"] == j]["q_B3"].median() for j in joins], w, label="B3: Single-Stage")
    ax.bar(x + 1.5 * w, [df[df["join_count"] == j]["q_Prop"].median() for j in joins], w, label="Proposed Two-Stage")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{j}-Join Queries" for j in joins])
    ax.set_ylabel("Median Q-Error (Lower is Better)")
    ax.set_title("Estimation Accuracy Across Query Join Complexity")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig_qerror_by_joins.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
    methods = [
        ("B1: Native PG", "B1_NativePG", "s"),
        ("B2: Flat RF", "B2_FlatRF", "^"),
        ("B3: Single-Stage", "B3_SingleStage", "D"),
        ("Proposed Two-Stage", "Proposed_TwoStage", "o"),
    ]
    for label, key, marker in methods:
        d = summary_table[key]
        lat = max(d["inference_latency_ms"], 0.05)
        ax.scatter(lat, d["p95_mean"], s=110, marker=marker, label=label)
        ax.annotate(label, (lat, d["p95_mean"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean Inference Overhead (ms / query)")
    ax.set_ylabel("95th Percentile Q-Error")
    ax.set_title("Tail Q-Error vs. Inference Overhead Trade-off")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig_latency_tradeoff.png"))
    plt.close(fig)