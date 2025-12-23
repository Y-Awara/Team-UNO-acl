"""
Cypher query templates for Milestone 3 (Baseline Retrieval)

Milestone 3: Standardize output keys so baseline/embeddings/hybrid can share one kg_results schema.

Also includes:
- QUERY_EXAMPLES: 10+ example questions for grading/demo
"""

CYPHER_TEMPLATES = {

    "PRODUCT_BY_CATEGORY": """
        MATCH (p:Product)
        WHERE p.product_category_name = $category
        RETURN
            p.product_id AS product_id,
            p.product_category_name AS product_category_name
        LIMIT 20
    """,

    "PRODUCT_BY_RATING": """
        MATCH (p:Product)<-[:REFERS_TO]-(:OrderItem)<-[:CONTAINS]-(:Order)<-[:REVIEWS]-(r:Review)
        WITH p, avg(toFloat(r.review_score)) AS avg_rating, count(r) AS n_reviews
        WHERE avg_rating >= toFloat($min_rating)
        RETURN
            p.product_id AS product_id,
            p.product_category_name AS product_category_name,
            avg_rating AS avg_rating,
            n_reviews AS n_reviews
        ORDER BY avg_rating DESC, n_reviews DESC
        LIMIT 20
    """,

    "REVIEWS_FOR_PRODUCT": """
        MATCH (p:Product {product_id: $product_id})
        <-[:REFERS_TO]-(:OrderItem)<-[:CONTAINS]-(:Order)<-[:REVIEWS]-(r:Review)
        RETURN
            p.product_id AS product_id,
            r.review_score AS review_score,
            r.review_comment_message AS review_comment_message
        LIMIT 20
    """,

    "LATE_DELIVERIES": """
        MATCH (o:Order)
        WHERE o.delivery_delay_days > $min_delay
        RETURN
            o.order_id AS order_id,
            o.delivery_delay_days AS delivery_delay_days
        ORDER BY delivery_delay_days DESC
        LIMIT 20
    """,

    "DELIVERY_VS_RATING": """
        MATCH (o:Order)<-[:REVIEWS]-(r:Review)
        WHERE o.delivery_delay_days IS NOT NULL
        RETURN
            o.delivery_delay_days AS delivery_delay_days,
            avg(toFloat(r.review_score)) AS avg_rating
        ORDER BY delivery_delay_days ASC
        LIMIT 200
    """,

    "SELLER_PERFORMANCE": """
        MATCH (s:Seller)<-[:SOLD_BY]-(:OrderItem)<-[:CONTAINS]-(:Order)<-[:REVIEWS]-(r:Review)
        WITH s, avg(toFloat(r.review_score)) AS avg_rating, count(r) AS n_reviews
        RETURN
            s.seller_id AS seller_id,
            avg_rating AS avg_rating,
            n_reviews AS n_reviews
        ORDER BY avg_rating DESC, n_reviews DESC
        LIMIT 20
    """,

    "CATEGORY_INSIGHTS": """
        MATCH (p:Product)<-[:REFERS_TO]-(:OrderItem)<-[:CONTAINS]-(:Order)<-[:REVIEWS]-(r:Review)
        WITH p.product_category_name AS product_category_name,
             avg(toFloat(r.review_score)) AS avg_rating,
             count(r) AS n_reviews
        RETURN
            product_category_name AS product_category_name,
            avg_rating AS avg_rating,
            n_reviews AS n_reviews
        ORDER BY avg_rating DESC, n_reviews DESC
        LIMIT 50
    """,

    "STATE_CUSTOMER_STATS": """
        MATCH (c:Customer)-[:LOCATED_IN]->(st:State)
        RETURN
            st.name AS state,
            count(c) AS customer_count
        ORDER BY customer_count DESC
        LIMIT 50
    """,

    "LOW_RATED_PRODUCTS": """
        MATCH (p:Product)<-[:REFERS_TO]-(:OrderItem)<-[:CONTAINS]-(:Order)<-[:REVIEWS]-(r:Review)
        WITH p, avg(toFloat(r.review_score)) AS avg_rating, count(r) AS n_reviews
        WHERE avg_rating <= toFloat($max_rating)
        RETURN
            p.product_id AS product_id,
            p.product_category_name AS product_category_name,
            avg_rating AS avg_rating,
            n_reviews AS n_reviews
        ORDER BY avg_rating ASC, n_reviews DESC
        LIMIT 20
    """,

    "HIGH_RATED_PRODUCTS": """
        MATCH (p:Product)<-[:REFERS_TO]-(:OrderItem)<-[:CONTAINS]-(:Order)<-[:REVIEWS]-(r:Review)
        WITH p, avg(toFloat(r.review_score)) AS avg_rating, count(r) AS n_reviews
        WHERE avg_rating >= toFloat($min_rating)
        RETURN
            p.product_id AS product_id,
            p.product_category_name AS product_category_name,
            avg_rating AS avg_rating,
            n_reviews AS n_reviews
        ORDER BY avg_rating DESC, n_reviews DESC
        LIMIT 20
    """,

    "GENERAL_STATS": """
        MATCH (p:Product)
        RETURN
            count(p) AS total_products
    """
}


# 10+ example questions mapped to intents (for grading slides + evaluation script)
QUERY_EXAMPLES = [
    ("Products in bebes category", "PRODUCT_BY_CATEGORY", {"category": "bebes"}),
    ("List low rated products with rating 2", "LOW_RATED_PRODUCTS", {"max_rating": 2}),
    ("List high rated products with rating 5", "HIGH_RATED_PRODUCTS", {"min_rating": 5}),
    ("Products with rating at least 4", "PRODUCT_BY_RATING", {"min_rating": 4}),
    ("Show reviews for product_id 04d70d184a7c12c93cecd9c0d8b5ad69", "REVIEWS_FOR_PRODUCT",
     {"product_id": "04d70d184a7c12c93cecd9c0d8b5ad69"}),
    ("Show late deliveries more than 10 days", "LATE_DELIVERIES", {"min_delay": 10}),
    ("How does delivery delay relate to rating?", "DELIVERY_VS_RATING", {}),
    ("Top seller performance", "SELLER_PERFORMANCE", {}),
    ("Category insights", "CATEGORY_INSIGHTS", {}),
    ("Customer stats by state", "STATE_CUSTOMER_STATS", {}),
    ("General stats", "GENERAL_STATS", {}),
]
