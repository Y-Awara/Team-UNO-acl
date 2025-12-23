# intent_classifier.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class IntentClassifier:
    def __init__(self, model_path=None, vectorizer_path=None):
        self.model = None
        self.vectorizer = None

        if model_path and vectorizer_path:
            self.load(model_path, vectorizer_path)
        else:
            # Auto-train default Milestone 3 model
            self._train_default_model()

    def _train_default_model(self):
        """
        Train a small default model for Milestone 3 testing.
        """
        texts = [
            "Show me products in bebes",
            "Get reviews for product 123",
            "Which orders had delays more than 5 days?",
            "List low rated products with rating 2",
            "Give me category insights",
            "Top sellers by rating",
            "Customer count per state",
            "Products with rating higher than 4",
            "Overall stats of products"
        ]

        labels = [
            "PRODUCT_BY_CATEGORY",
            "REVIEWS_FOR_PRODUCT",
            "LATE_DELIVERIES",
            "LOW_RATED_PRODUCTS",
            "CATEGORY_INSIGHTS",
            "SELLER_PERFORMANCE",
            "STATE_CUSTOMER_STATS",
            "PRODUCT_BY_RATING",
            "GENERAL_STATS"
        ]

        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2))
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X, labels)

    def train(self, texts, labels):
        """
        Train the intent classifier on custom data.
        """
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2))
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X, labels)

    def predict(self, text):
        """
        Predict the intent of a single message.
        """
        if not self.vectorizer or not self.model:
            raise ValueError("Model and vectorizer are not trained or loaded.")
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def save(self, model_path, vectorizer_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load(self, model_path, vectorizer_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)


# ------------------------
# Test the IntentClassifier
# ------------------------
# if __name__ == "__main__":
#     classifier = IntentClassifier()  # auto-trains default model

#     test_queries = [
#         "Show me products in Toys",
#         "Reviews for product 9876",
#         "Orders delayed more than 3 days",
#         "Top sellers by rating",
#         "Customer count in SP"
#     ]

#     for q in test_queries:
#         intent = classifier.predict(q)
#         print(f"Query: '{q}' → Predicted intent: {intent}")
