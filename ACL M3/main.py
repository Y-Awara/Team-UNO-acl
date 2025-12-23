import os
import time

from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from retrieval.cypher_retrieval import KGRetriever
from retrieval.embeddings_retriever import EmbeddingsRetriever
from llm_grounding import LLMGrounding

CYPHER_ONLY_INTENTS = {
    "PRODUCT_BY_CATEGORY",
    "REVIEWS_FOR_PRODUCT",
    "LOW_RATED_PRODUCTS",
    "HIGH_RATED_PRODUCTS",
    "PRODUCT_BY_RATING",
    "LATE_DELIVERIES",
    "DELIVERY_VS_RATING",
    "SELLER_PERFORMANCE",
    "CATEGORY_INSIGHTS",
    "STATE_CUSTOMER_STATS",
    "GENERAL_STATS",
}


def dedupe_by_product_id(results: list) -> list:
    seen = set()
    deduped = []
    for r in results:
        if not isinstance(r, dict):
            deduped.append(r)
            continue
        pid = r.get("product_id")
        if pid is None:
            deduped.append(r)
            continue
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append(r)

    for i, r in enumerate(deduped, start=1):
        if isinstance(r, dict):
            r["rank"] = i
    return deduped


def retrieve_results(intent, entities, query_text, retrieval_method, retriever, embeddings_retriever, embedding_prop):
    debug = {}

    if intent in CYPHER_ONLY_INTENTS:
        results, cypher, params = retriever.run_query(intent, entities, return_debug=True)
        debug = {"cypher": cypher, "params": params}
        return results, "baseline (forced)", debug

    if retrieval_method == "baseline":
        results, cypher, params = retriever.run_query(intent, entities, return_debug=True)
        debug = {"cypher": cypher, "params": params}
        return results, "baseline", debug

    if retrieval_method == "embeddings":
        r = embeddings_retriever.search_similar_nodes(
            query_text=query_text, label="Product", top_k=10, embedding_property=embedding_prop, intent=intent
        )
        return r, "embeddings", debug

    if retrieval_method == "hybrid":
        cypher_r, cypher, params = retriever.run_query(intent, entities, return_debug=True)
        emb_r = embeddings_retriever.search_similar_nodes(
            query_text=query_text,
            label="Product",
            top_k=max(10, len(cypher_r)),
            embedding_property=embedding_prop,
            intent=intent
        )
        merged = dedupe_by_product_id(cypher_r + emb_r)
        debug = {"cypher": cypher, "params": params}
        return merged, "hybrid (deduped)", debug

    return [], "baseline", debug


if __name__ == "__main__":
    intent_classifier = IntentClassifier()
    entity_extractor = EntityExtractor()

    retriever = KGRetriever()
    embeddings_retriever = EmbeddingsRetriever(embedding_model="all-mpnet-base-v2")

    retrieval_method = "hybrid"  # "baseline" | "embeddings" | "hybrid"
    embedding_prop = "embedding_mpnet"

    user_query = "List low rated products with rating 2"

    intent = intent_classifier.predict(user_query)
    entities = entity_extractor.extract(user_query, intent)

    print(f"\nPredicted intent: {intent}")
    print(f"Extracted entities: {entities}")

    t0 = time.time()
    kg_results, effective_method, debug = retrieve_results(
        intent, entities, user_query,
        retrieval_method, retriever, embeddings_retriever, embedding_prop
    )
    retrieval_latency = time.time() - t0

    print(f"\nEffective method: {effective_method}")
    print(f"Retrieval latency: {retrieval_latency:.2f}s")

    if debug.get("cypher"):
        print("\nCypher executed:\n", debug["cypher"])
        print("\nParams:\n", debug["params"])

    print("\nKG Results (first 10):")
    for r in kg_results[:10]:
        print(r)

    hf_token = os.getenv("HF_TOKEN", "")
    llm = LLMGrounding(model="google/gemma-2-2b-it", hf_api_key=hf_token)

    t1 = time.time()
    answer = llm.summarize_results(user_query, kg_results, intent=intent)
    llm_latency = time.time() - t1

    print(f"\nLLM latency: {llm_latency:.2f}s")
    print("\nLLM Summary:\n", answer)

    retriever.close()
    embeddings_retriever.close()
