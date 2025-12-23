# entity_extractor.py (KG-specific, robust version)

import re

class EntityExtractor:
    def __init__(self):
        # Regex patterns per intent
        self.patterns = {
            "PRODUCT_BY_CATEGORY": r"(?:category\s+(\w+)|in\s+(\w+)\s+category|(\w+)\s+category)",
            "PRODUCT_BY_RATING": r"(?:rating\s+greater\s+than\s+(\d+)|rating\s+(\d+))",
            "REVIEWS_FOR_PRODUCT": r"product\s+([\w\d]+)",
            "LATE_DELIVERIES": r"(?:delay(?:ed)?\s+(?:more\s+than\s+)?(\d+))",
            "LOW_RATED_PRODUCTS": r"rating\s+(\d+)"
        }

        # Map regex capture to entity names
        self.entity_names = {
            "PRODUCT_BY_CATEGORY": "category",
            "PRODUCT_BY_RATING": "min_rating",
            "REVIEWS_FOR_PRODUCT": "product_id",
            "LATE_DELIVERIES": "min_delay",
            "LOW_RATED_PRODUCTS": "max_rating"
        }

    def extract(self, text, intent=None):
        """
        Extract entities from text based on intent.
        Returns a dictionary of entity_name -> value
        """
        entities = {}
        if intent and intent in self.patterns:
            pattern = self.patterns[intent]
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Pick the first non-None capture group
                value = next((g for g in match.groups() if g is not None), None)
                if value is None:
                    return entities
                if value.isdigit():
                    value = int(value)
                entities[self.entity_names[intent]] = value
        return entities


# -------------------------
# Simple Test
# -------------------------
# if __name__ == "__main__":
#     extractor = EntityExtractor()

#     queries = [
#         "Show me products in bebes category",
#         "Show me products in category Toys",
#         "Get reviews for product ec72556b5da399d24fe06338e816a9ac",
#         "Orders delayed more than 3 days",
#         "List low rated products with rating 2",
#         "Products with rating higher than 4",
#         "Orders delay 5 days",
#         "Get reviews for product abc123XYZ",
#     ]

#     intents = [
#         "PRODUCT_BY_CATEGORY",
#         "PRODUCT_BY_CATEGORY",
#         "REVIEWS_FOR_PRODUCT",
#         "LATE_DELIVERIES",
#         "LOW_RATED_PRODUCTS",
#         "PRODUCT_BY_RATING",
#         "LATE_DELIVERIES",
#         "REVIEWS_FOR_PRODUCT",
#     ]

#     for q, intent in zip(queries, intents):
#         entities = extractor.extract(q, intent)
#         print(f"Query: '{q}' → Intent: {intent} → Extracted: {entities}")
