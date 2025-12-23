# create_kg.py
#
# Milestone 2 – Knowledge Graph Construction
# Step 2: read Neo4j credentials from config.txt and test the connection.
# Step 3: create constraints / indexes for main node types.
# Step 4: load CSV data and create all node types (no relationships yet).
# Step 5: load relationship pairs from CSV and create all relationships.

from neo4j import GraphDatabase
from pathlib import Path
import csv
from typing import Dict, Tuple, Any, List


CSV_PATH = "Ecommerce_KG_Optimized.csv"  # make sure this file is in the same folder as this script


# ============
# Config / Connection
# ============

def load_config(config_path: str = "config.txt") -> dict:
    """
    Load Neo4j connection details from a simple key=value config file.

    Expected format of config.txt:
        URI=neo4j://localhost:7687
        USERNAME=neo4j
        PASSWORD=your_password
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at: {config_file.resolve()}")

    config = {}
    with config_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid config line (missing '='): {line}")
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()

    required_keys = {"URI", "USERNAME", "PASSWORD"}
    missing = required_keys - set(config.keys())
    if missing:
        raise KeyError(f"Missing required config keys in config.txt: {missing}")

    return config


def get_driver(config: dict) -> GraphDatabase.driver:
    """
    Create and return a Neo4j driver using the given config dict.
    """
    uri = config["URI"]
    username = config["USERNAME"]
    password = config["PASSWORD"]

    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver


def test_connection(driver) -> None:
    """
    Run a tiny test query to ensure the Neo4j connection works.
    """
    with driver.session() as session:
        result = session.run("RETURN 1 AS result")
        record = result.single()
        if record is None:
            raise RuntimeError("No result returned from test query.")
        value = record["result"]
        print(f"✅ Neo4j connection successful, test query returned: {value}")


# ============
# Constraints
# ============

def create_constraints(driver) -> None:
    """
    Create uniqueness constraints for main node identifiers.
    Uses Neo4j modern syntax (FOR ... REQUIRE ... IS UNIQUE).
    """

    constraint_queries = [
        # Customer: unique by customer_id
        """
        CREATE CONSTRAINT customer_id_unique IF NOT EXISTS
        FOR (c:Customer)
        REQUIRE c.customer_id IS UNIQUE
        """,

        # Order: unique by order_id
        """
        CREATE CONSTRAINT order_id_unique IF NOT EXISTS
        FOR (o:Order)
        REQUIRE o.order_id IS UNIQUE
        """,

        # OrderItem: composite uniqueness on (order_id, order_item_id)
        # We will store both order_id and order_item_id on the OrderItem node.
        """
        CREATE CONSTRAINT order_item_unique IF NOT EXISTS
        FOR (oi:OrderItem)
        REQUIRE (oi.order_id, oi.order_item_id) IS UNIQUE
        """,

        # Product: unique by product_id
        """
        CREATE CONSTRAINT product_id_unique IF NOT EXISTS
        FOR (p:Product)
        REQUIRE p.product_id IS UNIQUE
        """,

        # Seller: unique by seller_id
        """
        CREATE CONSTRAINT seller_id_unique IF NOT EXISTS
        FOR (s:Seller)
        REQUIRE s.seller_id IS UNIQUE
        """,

        # Review: unique by review_id
        """
        CREATE CONSTRAINT review_id_unique IF NOT EXISTS
        FOR (r:Review)
        REQUIRE r.review_id IS UNIQUE
        """,

        # State: unique by name (e.g. 'SP', 'RJ', ...)
        """
        CREATE CONSTRAINT state_name_unique IF NOT EXISTS
        FOR (st:State)
        REQUIRE st.name IS UNIQUE
        """,
    ]

    with driver.session() as session:
        for q in constraint_queries:
            session.run(q)
            print("✅ Ensured constraint created/exists.")


# ============
# Helpers
# ============

def to_int(value: str) -> Any:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def to_float(value: str) -> Any:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def chunked(iterable: List[Dict[str, Any]], size: int = 1000):
    """Yield successive chunks from a list."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


# ============
# Load CSV into in-memory structures (nodes)
# ============

def load_ecommerce_data(csv_path: str = CSV_PATH):
    """
    Read the single denormalized CSV and build deduplicated dictionaries
    for each node type.
    """

    customers: Dict[str, Dict[str, Any]] = {}
    orders: Dict[str, Dict[str, Any]] = {}
    # key = (order_id, order_item_id)
    order_items: Dict[Tuple[str, str], Dict[str, Any]] = {}
    products: Dict[str, Dict[str, Any]] = {}
    sellers: Dict[str, Dict[str, Any]] = {}
    reviews: Dict[str, Dict[str, Any]] = {}
    states: Dict[str, Dict[str, Any]] = {}

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_file.resolve()}")

    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Safely extract values (strip whitespace)
            order_id = row.get("order_id", "").strip()
            customer_id = row.get("customer_id", "").strip()
            order_item_id = row.get("order_item_id", "").strip()
            product_id = row.get("product_id", "").strip()
            seller_id = row.get("seller_id", "").strip()
            review_id = row.get("review_id", "").strip()
            customer_state = row.get("customer_state", "").strip()

            # --- Customer ---
            if customer_id:
                customers.setdefault(customer_id, {
                    "customer_id": customer_id,
                    "customer_unique_id": row.get("customer_unique_id", "").strip(),
                    "customer_city": row.get("customer_city", "").strip(),
                    "customer_state": customer_state,
                })

            # --- Order ---
            if order_id:
                if order_id not in orders:
                    orders[order_id] = {
                        "order_id": order_id,
                        "order_status": row.get("order_status", "").strip(),
                        "order_purchase_timestamp": row.get("order_purchase_timestamp", "").strip(),
                        "order_approved_at": row.get("order_approved_at", "").strip(),
                        "order_delivered_carrier_date": row.get("order_delivered_carrier_date", "").strip(),
                        "order_delivered_customer_date": row.get("order_delivered_customer_date", "").strip(),
                        "order_estimated_delivery_date": row.get("order_estimated_delivery_date", "").strip(),
                        "delivery_delay_days": to_int(row.get("delivery_delay_days", "").strip()),
                    }

            # --- OrderItem ---
            if order_id and order_item_id:
                key = (order_id, order_item_id)
                if key not in order_items:
                    order_items[key] = {
                        "order_id": order_id,
                        "order_item_id": order_item_id,
                        "price": to_float(row.get("price", "").strip()),
                        "freight_value": to_float(row.get("freight_value", "").strip()),
                    }

            # --- Product ---
            if product_id:
                if product_id not in products:
                    # Note: CSV column is 'product_description_lenght' (typo)
                    prod_desc_len = row.get("product_description_lenght", "").strip()
                    products[product_id] = {
                        "product_id": product_id,
                        "product_category_name": row.get("product_category_name", "").strip(),
                        "product_description_length": to_int(prod_desc_len),
                        "product_photos_qty": to_int(row.get("product_photos_qty", "").strip()),
                    }

            # --- Seller ---
            if seller_id:
                sellers.setdefault(seller_id, {
                    "seller_id": seller_id
                })

            # --- Review ---
            if review_id:
                if review_id not in reviews:
                    reviews[review_id] = {
                        "review_id": review_id,
                        "review_score": to_int(row.get("review_score", "").strip()),
                        "review_comment_title": row.get("review_comment_title", "").strip(),
                        "review_comment_message": row.get("review_comment_message", "").strip(),
                        "review_creation_date": row.get("review_creation_date", "").strip(),
                        "review_length": to_int(row.get("review_length", "").strip()),
                    }

            # --- State ---
            if customer_state:
                states.setdefault(customer_state, {
                    "name": customer_state
                })

    print(f"📊 Loaded from CSV (nodes):")
    print(f"   Customers:   {len(customers)}")
    print(f"   Orders:      {len(orders)}")
    print(f"   OrderItems:  {len(order_items)}")
    print(f"   Products:    {len(products)}")
    print(f"   Sellers:     {len(sellers)}")
    print(f"   Reviews:     {len(reviews)}")
    print(f"   States:      {len(states)}")

    return customers, orders, order_items, products, sellers, reviews, states


# ============
# Node creation in Neo4j
# ============

def create_customers(driver, customers: Dict[str, Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MERGE (c:Customer {customer_id: row.customer_id})
    SET c.customer_unique_id = row.customer_unique_id,
        c.customer_city = row.customer_city,
        c.customer_state = row.customer_state
    """
    data = list(customers.values())
    with driver.session() as session:
        for batch in chunked(data, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(data)} Customer nodes.")


def create_orders(driver, orders: Dict[str, Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MERGE (o:Order {order_id: row.order_id})
    SET o.order_status = row.order_status,
        o.order_purchase_timestamp = row.order_purchase_timestamp,
        o.order_approved_at = row.order_approved_at,
        o.order_delivered_carrier_date = row.order_delivered_carrier_date,
        o.order_delivered_customer_date = row.order_delivered_customer_date,
        o.order_estimated_delivery_date = row.order_estimated_delivery_date,
        o.delivery_delay_days = row.delivery_delay_days
    """
    data = list(orders.values())
    with driver.session() as session:
        for batch in chunked(data, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(data)} Order nodes.")


def create_order_items(driver, order_items: Dict[Tuple[str, str], Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MERGE (oi:OrderItem {
        order_id: row.order_id,
        order_item_id: row.order_item_id
    })
    SET oi.price = row.price,
        oi.freight_value = row.freight_value
    """
    data = list(order_items.values())
    with driver.session() as session:
        for batch in chunked(data, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(data)} OrderItem nodes.")


def create_products(driver, products: Dict[str, Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MERGE (p:Product {product_id: row.product_id})
    SET p.product_category_name = row.product_category_name,
        p.product_description_length = row.product_description_length,
        p.product_photos_qty = row.product_photos_qty
    """
    data = list(products.values())
    with driver.session() as session:
        for batch in chunked(data, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(data)} Product nodes.")


def create_sellers(driver, sellers: Dict[str, Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MERGE (s:Seller {seller_id: row.seller_id})
    """
    data = list(sellers.values())
    with driver.session() as session:
        for batch in chunked(data, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(data)} Seller nodes.")


def create_reviews(driver, reviews: Dict[str, Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MERGE (r:Review {review_id: row.review_id})
    SET r.review_score = row.review_score,
        r.review_comment_title = row.review_comment_title,
        r.review_comment_message = row.review_comment_message,
        r.review_creation_date = row.review_creation_date,
        r.review_length = row.review_length
    """
    data = list(reviews.values())
    with driver.session() as session:
        for batch in chunked(data, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(data)} Review nodes.")


def create_states(driver, states: Dict[str, Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MERGE (st:State {name: row.name})
    """
    data = list(states.values())
    with driver.session() as session:
        for batch in chunked(data, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(data)} State nodes.")


# ============
# Load CSV relationship data (Step 5)
# ============

def load_relationship_data(csv_path: str = CSV_PATH):
    """
    Build deduplicated relationship pairs/tuples from the CSV.
    """

    rel_customer_order = set()        # (customer_id, order_id)
    rel_order_orderitem = set()       # (order_id, order_item_id)
    rel_orderitem_product = set()     # (order_id, order_item_id, product_id)
    rel_orderitem_seller = set()      # (order_id, order_item_id, seller_id)
    rel_customer_review = set()       # (customer_id, review_id)
    rel_review_order = set()          # (review_id, order_id)
    rel_customer_state = set()        # (customer_id, customer_state)

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_file.resolve()}")

    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            order_id = row.get("order_id", "").strip()
            customer_id = row.get("customer_id", "").strip()
            order_item_id = row.get("order_item_id", "").strip()
            product_id = row.get("product_id", "").strip()
            seller_id = row.get("seller_id", "").strip()
            review_id = row.get("review_id", "").strip()
            customer_state = row.get("customer_state", "").strip()

            # (Customer)-[:PLACED]->(Order)
            if customer_id and order_id:
                rel_customer_order.add((customer_id, order_id))

            # (Order)-[:CONTAINS]->(OrderItem)
            if order_id and order_item_id:
                rel_order_orderitem.add((order_id, order_item_id))

            # (OrderItem)-[:REFERS_TO]->(Product)
            if order_id and order_item_id and product_id:
                rel_orderitem_product.add((order_id, order_item_id, product_id))

            # (OrderItem)-[:SOLD_BY]->(Seller)
            if order_id and order_item_id and seller_id:
                rel_orderitem_seller.add((order_id, order_item_id, seller_id))

            # (Customer)-[:WROTE]->(Review)
            if customer_id and review_id:
                rel_customer_review.add((customer_id, review_id))

            # (Review)-[:REVIEWS]->(Order)
            if review_id and order_id:
                rel_review_order.add((review_id, order_id))

            # (Customer)-[:LOCATED_IN]->(State)
            if customer_id and customer_state:
                rel_customer_state.add((customer_id, customer_state))

    # Convert sets to list-of-dicts for UNWIND
    rel_data = {
        "customer_order": [
            {"customer_id": c, "order_id": o} for (c, o) in rel_customer_order
        ],
        "order_orderitem": [
            {"order_id": o, "order_item_id": oi} for (o, oi) in rel_order_orderitem
        ],
        "orderitem_product": [
            {"order_id": o, "order_item_id": oi, "product_id": p} for (o, oi, p) in rel_orderitem_product
        ],
        "orderitem_seller": [
            {"order_id": o, "order_item_id": oi, "seller_id": s} for (o, oi, s) in rel_orderitem_seller
        ],
        "customer_review": [
            {"customer_id": c, "review_id": r} for (c, r) in rel_customer_review
        ],
        "review_order": [
            {"review_id": r, "order_id": o} for (r, o) in rel_review_order
        ],
        "customer_state": [
            {"customer_id": c, "customer_state": st} for (c, st) in rel_customer_state
        ],
    }

    print("📊 Loaded from CSV (relationships):")
    for key, value in rel_data.items():
        print(f"   {key}: {len(value)}")

    return rel_data


# ============
# Relationship creation in Neo4j (Step 5)
# ============

def create_rel_customer_placed_order(driver, rels: List[Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MATCH (c:Customer {customer_id: row.customer_id})
    MATCH (o:Order {order_id: row.order_id})
    MERGE (c)-[:PLACED]->(o)
    """
    with driver.session() as session:
        for batch in chunked(rels, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(rels)} PLACED relationships.")


def create_rel_order_contains_orderitem(driver, rels: List[Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MATCH (o:Order {order_id: row.order_id})
    MATCH (oi:OrderItem {order_id: row.order_id, order_item_id: row.order_item_id})
    MERGE (o)-[:CONTAINS]->(oi)
    """
    with driver.session() as session:
        for batch in chunked(rels, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(rels)} CONTAINS relationships.")


def create_rel_orderitem_refers_product(driver, rels: List[Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MATCH (oi:OrderItem {order_id: row.order_id, order_item_id: row.order_item_id})
    MATCH (p:Product {product_id: row.product_id})
    MERGE (oi)-[:REFERS_TO]->(p)
    """
    with driver.session() as session:
        for batch in chunked(rels, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(rels)} REFERS_TO relationships.")


def create_rel_orderitem_soldby_seller(driver, rels: List[Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MATCH (oi:OrderItem {order_id: row.order_id, order_item_id: row.order_item_id})
    MATCH (s:Seller {seller_id: row.seller_id})
    MERGE (oi)-[:SOLD_BY]->(s)
    """
    with driver.session() as session:
        for batch in chunked(rels, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(rels)} SOLD_BY relationships.")


def create_rel_customer_wrote_review(driver, rels: List[Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MATCH (c:Customer {customer_id: row.customer_id})
    MATCH (r:Review {review_id: row.review_id})
    MERGE (c)-[:WROTE]->(r)
    """
    with driver.session() as session:
        for batch in chunked(rels, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(rels)} WROTE relationships.")


def create_rel_review_reviews_order(driver, rels: List[Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MATCH (r:Review {review_id: row.review_id})
    MATCH (o:Order {order_id: row.order_id})
    MERGE (r)-[:REVIEWS]->(o)
    """
    with driver.session() as session:
        for batch in chunked(rels, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(rels)} REVIEWS relationships.")


def create_rel_customer_located_in_state(driver, rels: List[Dict[str, Any]]):
    query = """
    UNWIND $batch AS row
    MATCH (c:Customer {customer_id: row.customer_id})
    MATCH (st:State {name: row.customer_state})
    MERGE (c)-[:LOCATED_IN]->(st)
    """
    with driver.session() as session:
        for batch in chunked(rels, size=1000):
            session.run(query, batch=batch)
    print(f"✅ Created/merged {len(rels)} LOCATED_IN relationships.")


# ============
# Main
# ============

def main():
    # Connect
    config = load_config("config.txt")
    driver = get_driver(config)

    try:
        # Step 2: sanity check
        test_connection(driver)

        # Step 3: constraints
        create_constraints(driver)

        # Step 4: load CSV + create nodes
        customers, orders, order_items, products, sellers, reviews, states = load_ecommerce_data(CSV_PATH)

        create_customers(driver, customers)
        create_orders(driver, orders)
        create_order_items(driver, order_items)
        create_products(driver, products)
        create_sellers(driver, sellers)
        create_reviews(driver, reviews)
        create_states(driver, states)

        print("🎉 Step 4 complete: all node types created.")

        # Step 5: load relationship data + create relationships
        rel_data = load_relationship_data(CSV_PATH)

        create_rel_customer_placed_order(driver, rel_data["customer_order"])
        create_rel_order_contains_orderitem(driver, rel_data["order_orderitem"])
        create_rel_orderitem_refers_product(driver, rel_data["orderitem_product"])
        create_rel_orderitem_soldby_seller(driver, rel_data["orderitem_seller"])
        create_rel_customer_wrote_review(driver, rel_data["customer_review"])
        create_rel_review_reviews_order(driver, rel_data["review_order"])
        create_rel_customer_located_in_state(driver, rel_data["customer_state"])

        print("🎉 Step 5 complete: all relationships created.")

    finally:
        driver.close()
        print("🔌 Neo4j driver closed.")


if __name__ == "__main__":
    main()
