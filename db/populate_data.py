import os
import io
import numpy as np
import psycopg2

SEED = 42
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://come381:come381pass@localhost:5432/ai4db_bench"
)


def copy_rows(cur, table_name: str, columns: list[str], rows: list[tuple]):
    buf = io.StringIO()
    for r in rows:
        buf.write("\t".join(str(x) for x in r) + "\n")
    buf.seek(0)
    cols_sql = ", ".join(columns)
    cur.copy_expert(f"COPY {table_name} ({cols_sql}) FROM STDIN", buf)


def populate_uniform(cur, rng: np.random.Generator):
    cur.execute("SELECT COUNT(*) FROM uniform_db.lineitem;")
    if cur.fetchone()[0] > 0:
        print("[DB] uniform_db already populated.")
        return

    print("[DB] Populating uniform_db...")
    regions = [(i, f"REGION_{i}") for i in range(5)]
    copy_rows(cur, "uniform_db.region", ["r_regionkey", "r_name"], regions)

    nations = [(i, f"NATION_{i}", i % 5) for i in range(25)]
    copy_rows(cur, "uniform_db.nation", ["n_nationkey", "n_name", "n_regionkey"], nations)

    suppliers = [
        (i, int(rng.integers(0, 25)), round(float(rng.uniform(-500, 9500)), 2))
        for i in range(1, 501)
    ]
    copy_rows(cur, "uniform_db.supplier", ["s_suppkey", "s_nationkey", "s_acctbal"], suppliers)

    segments = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]
    customers = [
        (
            i,
            int(rng.integers(0, 25)),
            segments[int(rng.integers(0, len(segments)))],
            round(float(rng.uniform(-999, 9999)), 2),
        )
        for i in range(1, 3001)
    ]
    copy_rows(
        cur,
        "uniform_db.customer",
        ["c_custkey", "c_nationkey", "c_mktsegment", "c_acctbal"],
        customers,
    )

    ptypes = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
    parts = [
        (
            i,
            ptypes[int(rng.integers(0, len(ptypes)))],
            int(rng.integers(1, 51)),
            round(float(rng.uniform(10.0, 2000.0)), 2),
        )
        for i in range(1, 2001)
    ]
    copy_rows(
        cur,
        "uniform_db.part",
        ["p_partkey", "p_type", "p_size", "p_retailprice"],
        parts,
    )

    statuses = ["O", "F", "P"]
    priorities = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
    orders = []
    for i in range(1, 12001):
        day_offset = int(rng.integers(0, 1000))
        odate = f"2023-{(day_offset % 12) + 1:02d}-{(day_offset % 28) + 1:02d}"
        orders.append(
            (
                i,
                int(rng.integers(1, 3001)),
                statuses[int(rng.integers(0, 3))],
                round(float(rng.uniform(50.0, 50000.0)), 2),
                odate,
                priorities[int(rng.integers(0, 5))],
            )
        )
    copy_rows(
        cur,
        "uniform_db.orders",
        [
            "o_orderkey",
            "o_custkey",
            "o_orderstatus",
            "o_totalprice",
            "o_orderdate",
            "o_orderpriority",
        ],
        orders,
    )

    lineitems = []
    for okey in range(1, 12001):
        n_lines = int(rng.integers(1, 5))
        for lnum in range(1, n_lines + 1):
            day_offset = int(rng.integers(0, 1000))
            sdate = f"2023-{(day_offset % 12) + 1:02d}-{(day_offset % 28) + 1:02d}"
            lineitems.append(
                (
                    okey,
                    lnum,
                    int(rng.integers(1, 2001)),
                    int(rng.integers(1, 501)),
                    int(rng.integers(1, 51)),
                    round(float(rng.uniform(100.0, 25000.0)), 2),
                    round(float(rng.choice([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])), 2),
                    sdate,
                )
            )
    copy_rows(
        cur,
        "uniform_db.lineitem",
        [
            "l_orderkey",
            "l_linenumber",
            "l_partkey",
            "l_suppkey",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_shipdate",
        ],
        lineitems,
    )


def zipf_bounded(rng: np.random.Generator, a: float, high: int, size: int) -> np.ndarray:
    vals = rng.zipf(a, size=size)
    return np.clip(vals, 1, high)


def populate_skewed(cur, rng: np.random.Generator):
    cur.execute("SELECT COUNT(*) FROM skewed_db.order_items;")
    if cur.fetchone()[0] > 0:
        print("[DB] skewed_db already populated.")
        return

    print("[DB] Populating skewed_db (Zipfian + Correlated)...")
    regions = [(i, f"REG_{i}") for i in range(1, 11)]
    copy_rows(cur, "skewed_db.regions", ["region_id", "region_code"], regions)

    categories = [(i, f"CAT_{i}", (i % 4) + 1) for i in range(1, 21)]
    copy_rows(cur, "skewed_db.categories", ["category_id", "category_name", "tax_tier"], categories)

    m_regs = zipf_bounded(rng, 1.8, 10, 400)
    merchants = [
        (i + 1, int(m_regs[i]), round(float(np.clip(rng.normal(4.2, 0.6), 1.0, 5.0)), 2))
        for i in range(400)
    ]
    copy_rows(cur, "skewed_db.merchants", ["merchant_id", "region_id", "rating"], merchants)

    u_regs = zipf_bounded(rng, 1.9, 10, 3000)
    users = []
    for i in range(3000):
        reg = int(u_regs[i])
        if reg <= 2:
            tier = "VIP" if rng.random() < 0.75 else "GOLD"
            loyalty = int(rng.integers(700, 1000))
        else:
            tier = "STANDARD" if rng.random() < 0.80 else "GOLD"
            loyalty = int(rng.integers(50, 450))
        users.append((i + 1, reg, tier, loyalty))
    copy_rows(cur, "skewed_db.users", ["user_id", "region_id", "tier", "loyalty_score"], users)

    p_cats = zipf_bounded(rng, 1.7, 20, 2000)
    products = []
    prod_cat_map = {}
    for i in range(2000):
        cat = int(p_cats[i])
        prod_cat_map[i + 1] = cat
        is_prem = 1 if (cat <= 3 and rng.random() < 0.85) else 0
        price = (
            round(float(rng.uniform(800.0, 5000.0)), 2)
            if is_prem == 1
            else round(float(rng.uniform(15.0, 400.0)), 2)
        )
        products.append((i + 1, cat, int(zipf_bounded(rng, 1.6, 400, 1)[0]), price, is_prem))
    copy_rows(
        cur,
        "skewed_db.products",
        ["product_id", "category_id", "merchant_id", "price", "is_premium"],
        products,
    )

    o_users = zipf_bounded(rng, 1.6, 3000, 12000)
    orders = []
    for i in range(12000):
        uid = int(o_users[i])
        reg = int(u_regs[uid - 1])
        status = 1 if reg <= 2 and rng.random() < 0.85 else int(rng.integers(1, 5))
        orders.append(
            (
                i + 1,
                uid,
                reg,
                round(float(rng.exponential(450.0) + 20.0), 2),
                int(zipf_bounded(rng, 1.4, 365, 1)[0]),
                status,
            )
        )
    copy_rows(
        cur,
        "skewed_db.orders",
        ["order_id", "user_id", "region_id", "order_amount", "created_day", "status_code"],
        orders,
    )

    items = []
    for oid in range(1, 12001):
        n_items = int(rng.integers(1, 5))
        for seq in range(1, n_items + 1):
            pid = int(zipf_bounded(rng, 1.75, 2000, 1)[0])
            cat = prod_cat_map[pid]
            disc = 0.25 if cat <= 3 else round(float(rng.choice([0.0, 0.05, 0.10])), 2)
            items.append((oid, seq, pid, cat, int(zipf_bounded(rng, 1.8, 20, 1)[0]), disc))
    copy_rows(
        cur,
        "skewed_db.order_items",
        ["order_id", "item_seq", "product_id", "category_id", "quantity", "discount_rate"],
        items,
    )


def inject_data_drift(conn):
    """Injects +20% skewed records to test model sensitivity/brittleness (Stage 4)."""
    rng = np.random.default_rng(999)
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(order_id), 0) FROM skewed_db.orders;")
        max_oid = cur.fetchone()[0]
        if max_oid >= 14400:
            return
        new_orders = []
        new_items = []
        for idx in range(1, 2401):
            oid = max_oid + idx
            uid = int(rng.integers(1, 150))
            new_orders.append((oid, uid, 1, round(float(rng.uniform(1500, 9000)), 2), 10, 1))
            for seq in range(1, 4):
                pid = int(rng.integers(1, 80))
                new_items.append((oid, seq, pid, 1, int(rng.integers(5, 20)), 0.25))
        copy_rows(
            cur,
            "skewed_db.orders",
            ["order_id", "user_id", "region_id", "order_amount", "created_day", "status_code"],
            new_orders,
        )
        copy_rows(
            cur,
            "skewed_db.order_items",
            ["order_id", "item_seq", "product_id", "category_id", "quantity", "discount_rate"],
            new_items,
        )
    conn.commit()


def main():
    rng = np.random.default_rng(SEED)
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            populate_uniform(cur, rng)
            populate_skewed(cur, rng)
        conn.commit()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("VACUUM ANALYZE;")
        print("[DB] VACUUM ANALYZE completed successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()