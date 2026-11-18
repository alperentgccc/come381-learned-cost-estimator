import numpy as np


def generate_uniform_queries(rng: np.random.Generator, n_per_tier: int = 80, unseen: bool = False) -> list[dict]:
    queries = []
    segments = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]
    ptypes = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
    statuses = ["O", "F", "P"]

    for tier_joins in [2, 4, 6]:
        for _ in range(n_per_tier):
            q_min = int(rng.integers(1, 25)) if not unseen else int(rng.integers(25, 45))
            q_max = q_min + int(rng.integers(5, 20))
            disc = float(rng.choice([0.02, 0.04, 0.06, 0.08]))
            seg = segments[int(rng.integers(0, len(segments)))]
            ptype = ptypes[int(rng.integers(0, len(ptypes)))]
            st = statuses[int(rng.integers(0, len(statuses)))]
            price_thr = round(float(rng.uniform(200, 1500)), 2)
            rkey = int(rng.integers(0, 5))

            if tier_joins == 2:
                sql = (
                    f"SELECT o.o_orderstatus, COUNT(*), SUM(l.l_extendedprice) "
                    f"FROM uniform_db.orders o "
                    f"JOIN uniform_db.lineitem l ON o.o_orderkey = l.l_orderkey "
                    f"WHERE l.l_quantity BETWEEN {q_min} AND {q_max} "
                    f"AND l.l_discount >= {disc} "
                    f"AND o.o_orderstatus = '{st}' "
                    f"GROUP BY o.o_orderstatus;"
                )
            elif tier_joins == 4:
                sql = (
                    f"SELECT c.c_mktsegment, COUNT(*), AVG(l.l_extendedprice) "
                    f"FROM uniform_db.customer c "
                    f"JOIN uniform_db.orders o ON c.c_custkey = o.o_custkey "
                    f"JOIN uniform_db.lineitem l ON o.o_orderkey = l.l_orderkey "
                    f"JOIN uniform_db.part p ON l.l_partkey = p.p_partkey "
                    f"WHERE c.c_mktsegment = '{seg}' "
                    f"AND p.p_type = '{ptype}' "
                    f"AND p.p_retailprice <= {price_thr} "
                    f"AND l.l_quantity BETWEEN {q_min} AND {q_max} "
                    f"GROUP BY c.c_mktsegment;"
                )
            else:
                sql = (
                    f"SELECT r.r_name, n.n_name, COUNT(*), SUM(l.l_extendedprice * (1 - l.l_discount)) "
                    f"FROM uniform_db.region r "
                    f"JOIN uniform_db.nation n ON r.r_regionkey = n.n_regionkey "
                    f"JOIN uniform_db.customer c ON n.n_nationkey = c.c_nationkey "
                    f"JOIN uniform_db.orders o ON c.c_custkey = o.o_custkey "
                    f"JOIN uniform_db.lineitem l ON o.o_orderkey = l.l_orderkey "
                    f"JOIN uniform_db.part p ON l.l_partkey = p.p_partkey "
                    f"WHERE r.r_regionkey = {rkey} "
                    f"AND c.c_mktsegment = '{seg}' "
                    f"AND p.p_retailprice <= {price_thr} "
                    f"AND l.l_quantity BETWEEN {q_min} AND {q_max} "
                    f"GROUP BY r.r_name, n.n_name;"
                )

            queries.append(
                {
                    "dataset": "uniform_db",
                    "split": "test_unseen" if unseen else "train_seen",
                    "join_count": tier_joins,
                    "sql": sql,
                }
            )
    return queries


def generate_skewed_queries(rng: np.random.Generator, n_per_tier: int = 80, unseen: bool = False) -> list[dict]:
    queries = []
    tiers = ["VIP", "GOLD", "STANDARD"]

    for tier_joins in [2, 4, 6]:
        for _ in range(n_per_tier):
            reg_id = int(rng.integers(1, 5)) if not unseen else int(rng.integers(5, 11))
            cat_id = int(rng.integers(1, 10)) if not unseen else int(rng.integers(10, 21))
            utier = tiers[int(rng.integers(0, len(tiers)))]
            loyalty_min = int(rng.integers(50, 750))
            price_min = round(float(rng.uniform(50, 1200)), 2)
            qty_max = int(rng.integers(3, 18))
            rating_min = round(float(rng.uniform(2.5, 4.5)), 2)

            if tier_joins == 2:
                sql = (
                    f"SELECT o.status_code, COUNT(*), SUM(oi.quantity) "
                    f"FROM skewed_db.orders o "
                    f"JOIN skewed_db.order_items oi ON o.order_id = oi.order_id "
                    f"WHERE o.region_id = {reg_id} "
                    f"AND oi.category_id <= {cat_id} "
                    f"AND oi.quantity <= {qty_max} "
                    f"GROUP BY o.status_code;"
                )
            elif tier_joins == 4:
                sql = (
                    f"SELECT u.tier, p.is_premium, COUNT(*), SUM(o.order_amount) "
                    f"FROM skewed_db.users u "
                    f"JOIN skewed_db.orders o ON u.user_id = o.user_id "
                    f"JOIN skewed_db.order_items oi ON o.order_id = oi.order_id "
                    f"JOIN skewed_db.products p ON oi.product_id = p.product_id "
                    f"WHERE u.region_id = {reg_id} "
                    f"AND u.tier = '{utier}' "
                    f"AND u.loyalty_score >= {loyalty_min} "
                    f"AND p.price >= {price_min} "
                    f"GROUP BY u.tier, p.is_premium;"
                )
            else:
                sql = (
                    f"SELECT r.region_code, c.category_name, COUNT(*), AVG(p.price) "
                    f"FROM skewed_db.regions r "
                    f"JOIN skewed_db.users u ON r.region_id = u.region_id "
                    f"JOIN skewed_db.orders o ON u.user_id = o.user_id "
                    f"JOIN skewed_db.order_items oi ON o.order_id = oi.order_id "
                    f"JOIN skewed_db.products p ON oi.product_id = p.product_id "
                    f"JOIN skewed_db.categories c ON p.category_id = c.category_id "
                    f"JOIN skewed_db.merchants m ON p.merchant_id = m.merchant_id "
                    f"WHERE r.region_id = {reg_id} "
                    f"AND c.category_id <= {cat_id} "
                    f"AND m.rating >= {rating_min} "
                    f"AND u.loyalty_score >= {loyalty_min} "
                    f"GROUP BY r.region_code, c.category_name;"
                )

            queries.append(
                {
                    "dataset": "skewed_db",
                    "split": "test_unseen" if unseen else "train_seen",
                    "join_count": tier_joins,
                    "sql": sql,
                }
            )
    return queries


def build_full_workload(seed: int = 42, n_train_per_tier: int = 80, n_unseen_per_tier: int = 30) -> list[dict]:
    rng = np.random.default_rng(seed)
    workload = []
    workload.extend(generate_uniform_queries(rng, n_per_tier=n_train_per_tier, unseen=False))
    workload.extend(generate_uniform_queries(rng, n_per_tier=n_unseen_per_tier, unseen=True))
    workload.extend(generate_skewed_queries(rng, n_per_tier=n_train_per_tier, unseen=False))
    workload.extend(generate_skewed_queries(rng, n_per_tier=n_unseen_per_tier, unseen=True))
    for idx, item in enumerate(workload):
        item["query_id"] = idx
    return workload