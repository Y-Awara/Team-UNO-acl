#Query 1 — Products by category

MATCH (p:Product)
WHERE p.product_category_name = "bebes"
RETURN p.product_id
LIMIT 10;



#Query 2 — Average rating per product

MATCH (p:Product)<-[:REFERS_TO]-(:OrderItem)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
RETURN p.product_id, avg(r.review_score) AS avg_rating
ORDER BY avg_rating DESC
LIMIT 10;



#Query 3 — Late deliveries

MATCH (o:Order)
WHERE o.delivery_delay_days > 0
RETURN o.order_id, o.delivery_delay_days
ORDER BY o.delivery_delay_days DESC
LIMIT 10;



#Query 4 — Products with good ratings but late delivery

MATCH (p:Product)<-[:REFERS_TO]-(:OrderItem)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
WHERE r.review_score >= 4 AND o.delivery_delay_days > 0
RETURN p.product_id,
       avg(r.review_score) AS avg_rating,
       avg(o.delivery_delay_days) AS avg_delay
ORDER BY avg_delay DESC
LIMIT 10;



#Query 5 — Seller performance

MATCH (s:Seller)<-[:SOLD_BY]-(:OrderItem)<-[:CONTAINS]-(o:Order)
RETURN s.seller_id,
       avg(o.delivery_delay_days) AS avg_delay
ORDER BY avg_delay ASC
LIMIT 10;



#Query 6 — Reviews for a product

MATCH (p:Product {product_id: $product_id})
<-[:REFERS_TO]-(:OrderItem)<-[:CONTAINS]-(o:Order)<-[:REVIEWS]-(r:Review)
RETURN r.review_score, r.review_comment_message
LIMIT 5;



#Query 7 — Orders per state

MATCH (c:Customer)-[:LOCATED_IN]->(st:State)
MATCH (c)-[:PLACED]->(o:Order)
RETURN st.name, count(o) AS total_orders
ORDER BY total_orders DESC;



#Query 8 — Review sentiment summary

MATCH (r:Review)
RETURN r.review_score, count(*) AS count
ORDER BY r.review_score DESC;



#Query 9 — Category-level insights

MATCH (p:Product)
RETURN p.product_category_name, count(*) AS products
ORDER BY products DESC;



#Query 10 — Delivery impact rule (Milestone 2 reuse)

MATCH (o:Order)<-[:REVIEWS]-(r:Review)
WHERE o.delivery_delay_days > 0
RETURN
  avg(r.review_score) AS avg_rating_with_delay,
  count(o) AS delayed_orders;
