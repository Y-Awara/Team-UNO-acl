import os
import csv
import time

from retrieval.cypher_templates import QUERY_EXAMPLES
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from retrieval.cypher_retrieval import KGRetriever
from retrieval.embeddings_retriever import EmbeddingsRetriever
from llm_grounding import LLMGrounding


LLM_MODELS = [
    "google/gemma-2-2b-it",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-3B-Instruct",
]

RETRIEVAL_METHODS = ["baseline", "embeddings", "hybrid"]


def dedupe_by_product_id(results: list) -> list:
    seen = set()
    out = []
    for r in results:
        pid = r.get("product_id") if isinstance(r, dict) else None
        if pid and pid in seen:
            continue
        if pid:
            seen.add(pid)
        out.append(r)
    for i, r in enumerate(out, start=1):
        if isinstance(r, dict):
            r["rank"] = i
    return out


def retrieve(intent, entities, query_text, method, retriever, emb, embedding_prop):
    if method == "baseline":
        results, _, _ = retriever.run_query(intent, entities, return_debug=True)
        return results

    if method == "embeddings":
        return emb.search_similar_nodes(
            query_text=query_text,
            label="Product",
            top_k=10,
            embedding_property=embedding_prop,
            intent=intent
        )

    if method == "hybrid":
        base, _, _ = retriever.run_query(intent, entities, return_debug=True)
        emb_res = emb.search_similar_nodes(
            query_text=query_text,
            label="Product",
            top_k=max(10, len(base)),
            embedding_property=embedding_prop,
            intent=intent
        )
        return dedupe_by_product_id(base + emb_res)

    return []


def main():
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN not set")

    intent_classifier = IntentClassifier()
    entity_extractor = EntityExtractor()
    retriever = KGRetriever()
    emb = EmbeddingsRetriever(embedding_model="all-mpnet-base-v2")
    embedding_prop = "embedding_mpnet"

    rows = []
    for (question, gold_intent, gold_entities) in QUERY_EXAMPLES[:10]:
        intent = intent_classifier.predict(question)
        entities = entity_extractor.extract(question, intent)

        for method in RETRIEVAL_METHODS:
            t0 = time.time()
            kg_results = retrieve(intent, entities, question, method, retriever, emb, embedding_prop)
            retrieval_s = time.time() - t0

            for llm_model in LLM_MODELS:
                llm = LLMGrounding(model=llm_model, hf_api_key=hf_token)

                t1 = time.time()
                _ = llm.summarize_results(question, kg_results, intent=intent)
                llm_s = time.time() - t1

                rows.append({
                    "question": question,
                    "pred_intent": intent,
                    "entities": str(entities),
                    "retrieval_method": method,
                    "llm_model": llm_model,
                    "retrieval_latency_s": round(retrieval_s, 3),
                    "llm_latency_s": round(llm_s, 3),
                    # fill this manually after reviewing outputs:
                    "qual_score_1to5": "",
                    "notes": ""
                })

                print(f"[OK] {method} | {llm_model} | {retrieval_s:.2f}s retr | {llm_s:.2f}s llm")

    out_path = "eval_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")

    retriever.close()
    emb.close()


if __name__ == "__main__":
    main()
