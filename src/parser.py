import numpy as np
import sqlglot
from sqlglot import exp

OPERATOR_TYPES = [
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Heap Scan",
    "Bitmap Index Scan",
    "Hash Join",
    "Merge Join",
    "Nested Loop",
    "Aggregate",
    "Sort",
    "Hash",
    "Gather",
    "Other",
]
OP_TO_IDX = {op: idx for idx, op in enumerate(OPERATOR_TYPES)}


def extract_sql_predicate_features(sql: str) -> np.ndarray:
    """Extracts lexical & syntactic SQL features via SQLGlot."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
        tables = len(list(tree.find_all(exp.Table)))
        joins = len(list(tree.find_all(exp.Join)))
        eq_ops = len(list(tree.find_all(exp.EQ)))
        range_ops = (
            len(list(tree.find_all(exp.LT)))
            + len(list(tree.find_all(exp.LTE)))
            + len(list(tree.find_all(exp.GT)))
            + len(list(tree.find_all(exp.GTE)))
            + len(list(tree.find_all(exp.Between)))
        )
        in_ops = len(list(tree.find_all(exp.In)))
        like_ops = len(list(tree.find_all(exp.Like))) + len(list(tree.find_all(exp.ILike)))
        group_cols = len(list(tree.find_all(exp.Group)))
        agg_funcs = len(list(tree.find_all(exp.AggFunc)))
        return np.array(
            [tables, joins, eq_ops, range_ops, in_ops, like_ops, group_cols, agg_funcs],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(8, dtype=np.float32)


def traverse_plan_bottom_up(plan_node: dict, sql_feats: np.ndarray, depth: int = 0) -> list[dict]:
    """Traverses PostgreSQL EXPLAIN JSON tree bottom-up and builds node records."""
    child_records = []
    direct_children = []
    for child in plan_node.get("Plans", []):
        sub_list = traverse_plan_bottom_up(child, sql_feats, depth=depth + 1)
        child_records.extend(sub_list)
        direct_children.append(sub_list[-1])

    node_type = plan_node.get("Node Type", "Other")
    op_vec = np.zeros(len(OPERATOR_TYPES), dtype=np.float32)
    op_vec[OP_TO_IDX.get(node_type, OP_TO_IDX["Other"])] = 1.0

    pg_plan_rows = max(float(plan_node.get("Plan Rows", 1.0)), 1.0)
    pg_startup_cost = max(float(plan_node.get("Startup Cost", 0.0)), 0.0)
    pg_total_cost = max(float(plan_node.get("Total Cost", 0.01)), 0.01)
    plan_width = max(float(plan_node.get("Plan Width", 8.0)), 1.0)

    actual_rows = max(float(plan_node.get("Actual Rows", 1.0)), 1.0)
    actual_loops = max(float(plan_node.get("Actual Loops", 1.0)), 1.0)
    total_actual_rows = max(actual_rows * actual_loops, 1.0)
    actual_time_ms = max(float(plan_node.get("Actual Total Time", 0.01)) * actual_loops, 0.01)

    child_pg_rows_sum = sum(c["pg_plan_rows"] for c in direct_children)
    child_pg_rows_max = max([c["pg_plan_rows"] for c in direct_children], default=0.0)
    child_actual_rows_sum = sum(c["actual_rows"] for c in direct_children)

    struct_feats = np.array(
        [
            float(depth),
            float(len(direct_children)),
            np.log1p(plan_width),
            np.log1p(pg_startup_cost),
            np.log1p(pg_total_cost),
            np.log1p(pg_plan_rows),
            np.log1p(child_pg_rows_sum),
            np.log1p(child_pg_rows_max),
        ],
        dtype=np.float32,
    )

    node_entry = {
        "node_type": node_type,
        "depth": depth,
        "is_root": depth == 0,
        "op_vec": op_vec,
        "struct_feats": struct_feats,
        "sql_feats": sql_feats,
        "pg_plan_rows": pg_plan_rows,
        "pg_total_cost": pg_total_cost,
        "child_pg_rows_sum": child_pg_rows_sum,
        "child_pg_rows_max": child_pg_rows_max,
        "child_actual_rows_sum": child_actual_rows_sum,
        "actual_rows": total_actual_rows,
        "actual_time_ms": actual_time_ms,
    }
    child_records.append(node_entry)
    return child_records


def build_query_flat_features(nodes: list[dict], sql_feats: np.ndarray) -> np.ndarray:
    """Flattened operator histogram + SQL predicate features for Baseline 2."""
    op_counts = np.sum([n["op_vec"] for n in nodes], axis=0)
    total_nodes = np.array([float(len(nodes))], dtype=np.float32)
    return np.concatenate([op_counts, total_nodes, sql_feats], axis=0)


def featurize_node_stage1(node: dict) -> np.ndarray:
    """Features for Stage-1 Learned Cardinality Estimator."""
    return np.concatenate([node["op_vec"], node["struct_feats"], node["sql_feats"]], axis=0)


def featurize_root_stage2(
    root_node: dict,
    learned_root_card: float,
    learned_child_card_sum: float,
    learned_child_card_max: float,
    ablation_mode: str = "full",
) -> np.ndarray:
    """
    Constructs Stage-2 Cost Prediction features for the query root node.
    Supports ablation modes:
      - 'full': Learned cardinality + hierarchical structure + SQL predicates
      - 'no_stage1': Replaces learned cardinality with native PG Plan Rows
      - 'no_hierarchy': Removes tree depth & child aggregation features
      - 'no_predicates': Zeroes out SQLGlot predicate/selectivity features
    """
    op_vec = root_node["op_vec"].copy()
    struct = root_node["struct_feats"].copy()
    sql_f = root_node["sql_feats"].copy()

    if ablation_mode != "no_stage1":
        struct[5] = np.log1p(max(learned_root_card, 1.0))
        struct[6] = np.log1p(max(learned_child_card_sum, 0.0))
        struct[7] = np.log1p(max(learned_child_card_max, 0.0))

    if ablation_mode == "no_hierarchy":
        struct[0] = 0.0
        struct[1] = 0.0
        struct[6] = 0.0
        struct[7] = 0.0

    if ablation_mode == "no_predicates":
        sql_f = np.zeros_like(sql_f)

    return np.concatenate([op_vec, struct, sql_f], axis=0)