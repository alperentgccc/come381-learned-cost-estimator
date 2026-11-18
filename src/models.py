import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.parser import (
    extract_sql_predicate_features,
    traverse_plan_bottom_up,
    build_query_flat_features,
    featurize_node_stage1,
    featurize_root_stage2,
)


class ResidualTabularNet(nn.Module):
    """PyTorch MLP with residual skip connection trained in log-space to minimize Q-Error."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.GELU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.act(h + self.block(h))
        return self.head(h).squeeze(-1)


def train_pytorch_regressor(
    X: np.ndarray, y_log: np.ndarray, seed: int = 42, epochs: int = 35, batch_size: int = 64
) -> tuple[ResidualTabularNet, StandardScaler]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_log.astype(np.float32)))
    loader = DataLoader(ds, batch_size=min(batch_size, max(8, len(ds))), shuffle=True)

    model = ResidualTabularNet(in_dim=X.shape[1], hidden_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.L1Loss()

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    return model, scaler


def predict_pytorch_regressor(
    model: ResidualTabularNet, scaler: StandardScaler, X: np.ndarray
) -> np.ndarray:
    X_scaled = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        preds_log = model(torch.from_numpy(X_scaled)).numpy()
    return np.maximum(np.expm1(np.clip(preds_log, 0.0, 20.0)), 1e-2)


class AI4DBBenchmarkSuite:
    """
    Encapsulates:
      - Baseline 1 (B1): Native PostgreSQL Optimizer
      - Baseline 2 (B2): Flat-Feature Classical Regressor (Random Forest)
      - Baseline 3 (B3): Direct Single-Stage Cost Predictor (XGBoost on raw PG Plan Rows)
      - Proposed Method: Two-Stage Cardinality-Aware Learned Cost Estimator (XGBoost + PyTorch Hybrid)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.b1_cost_scale = 1.0
        self.b2_rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1)
        self.b3_xgb = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.85, random_state=seed
        )
        self.stage1_xgb = xgb.XGBRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.85, random_state=seed
        )
        self.stage1_nn = None
        self.stage1_scaler = None

        self.stage2_xgb = xgb.XGBRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.85, random_state=seed
        )
        self.stage2_nn = None
        self.stage2_scaler = None

        self.ablation_models = {}

    def _prepare_query_structures(self, records: list[dict]) -> list[dict]:
        parsed = []
        for rec in records:
            sql_feats = extract_sql_predicate_features(rec["sql"])
            nodes = traverse_plan_bottom_up(rec["plan_json"], sql_feats)
            root = nodes[-1]
            flat_feats = build_query_flat_features(nodes, sql_feats)
            parsed.append(
                {
                    "query_id": rec.get("query_id", -1),
                    "dataset": rec.get("dataset", "unknown"),
                    "split": rec.get("split", "train_seen"),
                    "join_count": rec.get("join_count", 2),
                    "sql": rec["sql"],
                    "planning_time_ms": rec.get("planning_time_ms", 0.5),
                    "execution_time_ms": rec.get("execution_time_ms", 1.0),
                    "nodes": nodes,
                    "root": root,
                    "flat_feats": flat_feats,
                }
            )
        return parsed

    def _predict_stage1_cards(self, node_feats: np.ndarray) -> np.ndarray:
        pred_xgb = np.maximum(np.expm1(np.clip(self.stage1_xgb.predict(node_feats), 0.0, 22.0)), 1.0)
        pred_nn = np.maximum(predict_pytorch_regressor(self.stage1_nn, self.stage1_scaler, node_feats), 1.0)
        return np.sqrt(pred_xgb * pred_nn)

    def fit(self, train_records: list[dict]):
        parsed = self._prepare_query_structures(train_records)

        pg_costs = np.array([p["root"]["pg_total_cost"] for p in parsed], dtype=np.float64)
        actual_ms = np.array([p["execution_time_ms"] for p in parsed], dtype=np.float64)
        self.b1_cost_scale = float(np.median(actual_ms / np.maximum(pg_costs, 1e-3)))

        X_flat = np.vstack([p["flat_feats"] for p in parsed])
        y_time_log = np.log1p(actual_ms)
        self.b2_rf.fit(X_flat, y_time_log)

        X_nodes_list, y_nodes_log_list = [], []
        for p in parsed:
            for n in p["nodes"]:
                X_nodes_list.append(featurize_node_stage1(n))
                y_nodes_log_list.append(np.log1p(n["actual_rows"]))
        X_nodes = np.vstack(X_nodes_list)
        y_nodes_log = np.array(y_nodes_log_list, dtype=np.float32)

        self.stage1_xgb.fit(X_nodes, y_nodes_log)
        self.stage1_nn, self.stage1_scaler = train_pytorch_regressor(
            X_nodes, y_nodes_log, seed=self.seed, epochs=30
        )

        X_s2_full, X_s2_no_s1, X_s2_no_hier, X_s2_no_pred = [], [], [], []
        for p in parsed:
            node_f = np.vstack([featurize_node_stage1(n) for n in p["nodes"]])
            pred_cards = self._predict_stage1_cards(node_f)
            root_card = float(pred_cards[-1])
            child_cards = pred_cards[:-1] if len(pred_cards) > 1 else np.array([0.0])
            c_sum = float(np.sum(child_cards))
            c_max = float(np.max(child_cards))

            X_s2_full.append(featurize_root_stage2(p["root"], root_card, c_sum, c_max, "full"))
            X_s2_no_s1.append(featurize_root_stage2(p["root"], root_card, c_sum, c_max, "no_stage1"))
            X_s2_no_hier.append(featurize_root_stage2(p["root"], root_card, c_sum, c_max, "no_hierarchy"))
            X_s2_no_pred.append(featurize_root_stage2(p["root"], root_card, c_sum, c_max, "no_predicates"))

        X_s2_full = np.vstack(X_s2_full)
        X_s2_no_s1 = np.vstack(X_s2_no_s1)
        X_s2_no_hier = np.vstack(X_s2_no_hier)
        X_s2_no_pred = np.vstack(X_s2_no_pred)

        self.b3_xgb.fit(X_s2_no_s1, y_time_log)

        self.stage2_xgb.fit(X_s2_full, y_time_log)
        self.stage2_nn, self.stage2_scaler = train_pytorch_regressor(
            X_s2_full, y_time_log, seed=self.seed + 1, epochs=40
        )

        for mode, X_ab in [
            ("no_stage1", X_s2_no_s1),
            ("no_hierarchy", X_s2_no_hier),
            ("no_predicates", X_s2_no_pred),
        ]:
            m = xgb.XGBRegressor(
                n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.85, random_state=self.seed
            )
            m.fit(X_ab, y_time_log)
            self.ablation_models[mode] = m

    def predict_single_query(self, plan_json: dict, sql: str) -> dict:
        """Runs live inference for a single query across B1, B2, B3, and Proposed Method."""
        sql_feats = extract_sql_predicate_features(sql)
        nodes = traverse_plan_bottom_up(plan_json, sql_feats)
        root = nodes[-1]
        flat_feats = build_query_flat_features(nodes, sql_feats).reshape(1, -1)

        b1_card = root["pg_plan_rows"]
        b1_time_ms = max(root["pg_total_cost"] * self.b1_cost_scale, 0.01)

        b2_time_ms = float(np.maximum(np.expm1(self.b2_rf.predict(flat_feats)[0]), 0.01))

        node_f = np.vstack([featurize_node_stage1(n) for n in nodes])
        pred_cards = self._predict_stage1_cards(node_f)
        root_card = float(pred_cards[-1])
        child_cards = pred_cards[:-1] if len(pred_cards) > 1 else np.array([0.0])
        c_sum, c_max = float(np.sum(child_cards)), float(np.max(child_cards))

        x_b3 = featurize_root_stage2(root, root_card, c_sum, c_max, "no_stage1").reshape(1, -1)
        b3_time_ms = float(np.maximum(np.expm1(self.b3_xgb.predict(x_b3)[0]), 0.01))

        x_prop = featurize_root_stage2(root, root_card, c_sum, c_max, "full").reshape(1, -1)
        t_xgb = float(np.maximum(np.expm1(self.stage2_xgb.predict(x_prop)[0]), 0.01))
        t_nn = float(predict_pytorch_regressor(self.stage2_nn, self.stage2_scaler, x_prop)[0])
        prop_time_ms = float(np.sqrt(t_xgb * t_nn))

        ab_preds = {}
        for mode, m in self.ablation_models.items():
            x_ab = featurize_root_stage2(root, root_card, c_sum, c_max, mode).reshape(1, -1)
            ab_preds[mode] = float(np.maximum(np.expm1(m.predict(x_ab)[0]), 0.01))

        node_breakdown = [
            {
                "node_type": n["node_type"],
                "depth": n["depth"],
                "pg_plan_rows": round(n["pg_plan_rows"], 2),
                "learned_card": round(float(pred_cards[idx]), 2),
                "actual_rows": round(n["actual_rows"], 2),
            }
            for idx, n in enumerate(nodes)
        ]

        return {
            "b1_native_pg": {"pred_cardinality": round(b1_card, 2), "pred_time_ms": round(b1_time_ms, 3)},
            "b2_flat_rf": {"pred_time_ms": round(b2_time_ms, 3)},
            "b3_single_stage": {"pred_time_ms": round(b3_time_ms, 3)},
            "proposed_two_stage": {
                "pred_cardinality": round(root_card, 2),
                "pred_time_ms": round(prop_time_ms, 3),
            },
            "ablations": ab_preds,
            "node_breakdown": node_breakdown,
        }

    def save(self, path: str = "models/ai4db_suite.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = "models/ai4db_suite.pkl") -> "AI4DBBenchmarkSuite":
        with open(path, "rb") as f:
            return pickle.load(f)