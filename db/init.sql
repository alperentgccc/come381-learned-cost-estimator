-- ============================================================================
-- SCHEMA 1: uniform_db (Standard / Balanced Distribution - TPC-H Style)
-- ============================================================================
CREATE SCHEMA IF NOT EXISTS uniform_db;

CREATE TABLE uniform_db.region (
    r_regionkey INT PRIMARY KEY,
    r_name VARCHAR(25) NOT NULL
);

CREATE TABLE uniform_db.nation (
    n_nationkey INT PRIMARY KEY,
    n_name VARCHAR(25) NOT NULL,
    n_regionkey INT NOT NULL REFERENCES uniform_db.region(r_regionkey)
);

CREATE TABLE uniform_db.supplier (
    s_suppkey INT PRIMARY KEY,
    s_nationkey INT NOT NULL REFERENCES uniform_db.nation(n_nationkey),
    s_acctbal NUMERIC(12, 2) NOT NULL
);

CREATE TABLE uniform_db.customer (
    c_custkey INT PRIMARY KEY,
    c_nationkey INT NOT NULL REFERENCES uniform_db.nation(n_nationkey),
    c_mktsegment VARCHAR(15) NOT NULL,
    c_acctbal NUMERIC(12, 2) NOT NULL
);

CREATE TABLE uniform_db.part (
    p_partkey INT PRIMARY KEY,
    p_type VARCHAR(25) NOT NULL,
    p_size INT NOT NULL,
    p_retailprice NUMERIC(12, 2) NOT NULL
);

CREATE TABLE uniform_db.orders (
    o_orderkey INT PRIMARY KEY,
    o_custkey INT NOT NULL REFERENCES uniform_db.customer(c_custkey),
    o_orderstatus CHAR(1) NOT NULL,
    o_totalprice NUMERIC(12, 2) NOT NULL,
    o_orderdate DATE NOT NULL,
    o_orderpriority VARCHAR(15) NOT NULL
);

CREATE TABLE uniform_db.lineitem (
    l_orderkey INT NOT NULL REFERENCES uniform_db.orders(o_orderkey),
    l_linenumber INT NOT NULL,
    l_partkey INT NOT NULL REFERENCES uniform_db.part(p_partkey),
    l_suppkey INT NOT NULL REFERENCES uniform_db.supplier(s_suppkey),
    l_quantity INT NOT NULL,
    l_extendedprice NUMERIC(12, 2) NOT NULL,
    l_discount NUMERIC(4, 2) NOT NULL,
    l_shipdate DATE NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber)
);

CREATE INDEX idx_u_orders_cust ON uniform_db.orders(o_custkey);
CREATE INDEX idx_u_orders_date ON uniform_db.orders(o_orderdate);
CREATE INDEX idx_u_line_part ON uniform_db.lineitem(l_partkey);
CREATE INDEX idx_u_line_supp ON uniform_db.lineitem(l_suppkey);

-- ============================================================================
-- SCHEMA 2: skewed_db (Zipfian Skew & Strong Inter-Column Correlation)
-- ============================================================================
CREATE SCHEMA IF NOT EXISTS skewed_db;

CREATE TABLE skewed_db.regions (
    region_id INT PRIMARY KEY,
    region_code VARCHAR(10) NOT NULL
);

CREATE TABLE skewed_db.categories (
    category_id INT PRIMARY KEY,
    category_name VARCHAR(30) NOT NULL,
    tax_tier INT NOT NULL
);

CREATE TABLE skewed_db.merchants (
    merchant_id INT PRIMARY KEY,
    region_id INT NOT NULL REFERENCES skewed_db.regions(region_id),
    rating NUMERIC(3, 2) NOT NULL
);

CREATE TABLE skewed_db.users (
    user_id INT PRIMARY KEY,
    region_id INT NOT NULL REFERENCES skewed_db.regions(region_id),
    tier VARCHAR(15) NOT NULL,
    loyalty_score INT NOT NULL
);

CREATE TABLE skewed_db.products (
    product_id INT PRIMARY KEY,
    category_id INT NOT NULL REFERENCES skewed_db.categories(category_id),
    merchant_id INT NOT NULL REFERENCES skewed_db.merchants(merchant_id),
    price NUMERIC(12, 2) NOT NULL,
    is_premium INT NOT NULL
);

CREATE TABLE skewed_db.orders (
    order_id INT PRIMARY KEY,
    user_id INT NOT NULL REFERENCES skewed_db.users(user_id),
    region_id INT NOT NULL REFERENCES skewed_db.regions(region_id),
    order_amount NUMERIC(12, 2) NOT NULL,
    created_day INT NOT NULL,
    status_code INT NOT NULL
);

CREATE TABLE skewed_db.order_items (
    order_id INT NOT NULL REFERENCES skewed_db.orders(order_id),
    item_seq INT NOT NULL,
    product_id INT NOT NULL REFERENCES skewed_db.products(product_id),
    category_id INT NOT NULL REFERENCES skewed_db.categories(category_id),
    quantity INT NOT NULL,
    discount_rate NUMERIC(4, 2) NOT NULL,
    PRIMARY KEY (order_id, item_seq)
);

CREATE INDEX idx_s_orders_user ON skewed_db.orders(user_id);
CREATE INDEX idx_s_orders_day ON skewed_db.orders(created_day);
CREATE INDEX idx_s_items_prod ON skewed_db.order_items(product_id);
CREATE INDEX idx_s_items_cat ON skewed_db.order_items(category_id);