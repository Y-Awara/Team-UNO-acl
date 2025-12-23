import streamlit as st
import os
import atexit
import time
from dotenv import load_dotenv

from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from retrieval.cypher_retrieval import KGRetriever
from retrieval.embeddings_retriever import EmbeddingsRetriever
from llm_grounding import LLMGrounding


load_dotenv()

# -----------------------------
# Safety: required params per intent (prevents Neo4j ParameterMissing)
# -----------------------------
REQUIRED_PARAMS = {
    "REVIEWS_FOR_PRODUCT": ["product_id"],
    "PRODUCT_BY_CATEGORY": ["category"],
    "LOW_RATED_PRODUCTS": ["max_rating"],
    "HIGH_RATED_PRODUCTS": ["min_rating"],
    "PRODUCT_BY_RATING": ["min_rating"],
    "LATE_DELIVERIES": ["min_delay"],
}

# Intents that MUST use Cypher (structured)
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

# Minimal translation layer for common Portuguese category labels (expand as needed)
CATEGORY_TRANSLATIONS = {
    "bebes": "babies",
    "cama_mesa_banho": "bed_bath_table",
    "informatica_acessorios": "computers_accessories",
    "moveis_decoracao": "furniture_decor",
    "relogios_presentes": "watches_gifts",
}

# Reverse map: English query synonyms -> KG category label
CATEGORY_SYNONYMS_TO_KG = {
    "baby": "bebes",
    "babies": "bebes",
    "infant": "bebes",
    "newborn": "bebes",
    "toddler": "bebes",
}

def normalize_category_entity_to_kg(entities: dict) -> dict:
    """
    If entity extractor returns English category like 'babies',
    convert it to the KG's Portuguese label like 'bebes' before Cypher runs.
    """
    if not entities or not entities.get("category"):
        return entities

    cat = str(entities["category"]).lower().strip()

    # English synonym -> KG Portuguese
    if cat in CATEGORY_SYNONYMS_TO_KG:
        entities["category"] = CATEGORY_SYNONYMS_TO_KG[cat]
        return entities

    # If extractor returned the English translation value (e.g., 'bed_bath_table'),
    # map back to Portuguese KG label
    for pt, en in CATEGORY_TRANSLATIONS.items():
        if cat == en.lower():
            entities["category"] = pt
            return entities

    return entities


def translate_category(cat: str) -> str:
    if not cat:
        return cat
    return CATEGORY_TRANSLATIONS.get(cat, cat)


def map_ui_retrieval_method(ui_choice: str) -> str:
    if ui_choice == "Baseline (Cypher)":
        return "baseline"
    if ui_choice == "Embeddings":
        return "embeddings"
    if ui_choice == "Hybrid":
        return "hybrid"
    return "baseline"


def lightweight_view_rows(results: list, max_rows: int = 20) -> list:
    view = []
    for r in results[:max_rows]:
        if not isinstance(r, dict):
            view.append({"value": str(r)})
            continue

        keys_priority = [
            "product_id", "product_category_name", "product_category_name_en",
            "avg_rating", "n_reviews",
            "seller_id", "order_id", "delivery_delay_days", "score", "state",
            "customer_count", "total_products", "review_score", "review_comment_message",
            "source", "intent", "rank",
        ]
        row = {}
        for k in keys_priority:
            if k in r and r[k] is not None:
                row[k] = r[k]
        view.append(row if row else r)
    return view


def add_translations(kg_results: list) -> list:
    """Add *_en fields for category labels so UI + LLM can use English."""
    out = []
    for r in kg_results:
        if not isinstance(r, dict):
            out.append(r)
            continue
        rr = dict(r)
        if "product_category_name" in rr and rr["product_category_name"]:
            rr["product_category_name_en"] = translate_category(rr["product_category_name"])
        out.append(rr)
    return out


def dedupe_by_product_id(results: list) -> list:
    """
    Deduplicate hybrid results by product_id.
    Baseline should appear first in merge, so first occurrence wins.
    """
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

    # Re-rank deterministically
    reranked = []
    for i, r in enumerate(deduped, start=1):
        if isinstance(r, dict):
            rr = dict(r)
            rr["rank"] = i
            reranked.append(rr)
        else:
            reranked.append(r)
    return reranked


def missing_required_params(intent: str, entities: dict) -> list:
    required = REQUIRED_PARAMS.get(intent, [])
    missing = []
    for k in required:
        if entities.get(k) is None:
            missing.append(k)
    return missing

import re

def infer_numeric_thresholds(intent: str, entities: dict, query_text: str):
    """
    UI-level safety net: if intent needs a number and extractor missed it,
    parse the first integer in the query and fill the expected param.
    """
    q = query_text.lower()

    # Extract first integer like 5, 10, 3...
    m = re.search(r"\b(\d+)\b", q)
    num = int(m.group(1)) if m else None

    if intent == "LATE_DELIVERIES" and entities.get("min_delay") is None and num is not None:
        entities["min_delay"] = num

    if intent == "LOW_RATED_PRODUCTS" and entities.get("max_rating") is None and num is not None:
        entities["max_rating"] = num

    if intent in ("HIGH_RATED_PRODUCTS", "PRODUCT_BY_RATING") and entities.get("min_rating") is None and num is not None:
        entities["min_rating"] = num

    return entities

def normalize_query_to_kg(intent: str, entities: dict, query_text: str):
    """
    Make the system robust for English user queries when the KG uses Portuguese categories.
    Example: "baby" => "bebes"

    We apply a HIGH PRECISION override:
    - If query contains baby-synonyms and category is not already extracted,
      set entities["category"] and route to PRODUCT_BY_CATEGORY (structured).
    """
    q = (query_text or "").lower()

    # ✅ HIGH-PRECISION SEMANTIC OVERRIDE (FOR EMBEDDINGS DEMO)
    semantic_triggers = ["similar", "recommend", "related", "like"]
    if any(word in q for word in semantic_triggers):
        # Force semantic-style intent, bypassing Cypher-only logic
        return "SEMANTIC_SEARCH", {}
    
    # If user is clearly asking for reviews of a specific product, don't override
    # (but entity extractor should have product_id anyway)
    if intent == "REVIEWS_FOR_PRODUCT":
        return intent, entities

    # If user already has category extracted, just return
    if entities.get("category"):
        return intent, entities

    for word, kg_cat in CATEGORY_SYNONYMS_TO_KG.items():
        if word in q:
            # Override to category intent for clean demo behavior
            entities["category"] = kg_cat
            return "PRODUCT_BY_CATEGORY", entities

    return intent, entities


def retrieve_with_debug(
    intent: str,
    entities: dict,
    query_text: str,
    method: str,
    retriever: KGRetriever,
    embeddings_retriever: EmbeddingsRetriever | None,
    embedding_prop: str
):
    debug = {
        "intent": intent,
        "entities": dict(entities),
        "embedding_prop": embedding_prop,
        "method": method
    }

    # 🔹 Semantic-only intent (embeddings)
    if intent == "SEMANTIC_SEARCH":
        if embeddings_retriever is None:
            return [], "embeddings unavailable", debug

        results = embeddings_retriever.search_similar_nodes(
            query_text=query_text,
            label="Product",
            top_k=10,
            embedding_property=embedding_prop,
        )
        return results, "embeddings (semantic)", debug


    # Validate required parameters BEFORE Cypher (prevents Neo4j ParameterMissing)
    missing = missing_required_params(intent, entities)
    if missing:
        debug["param_error"] = f"Missing required params for intent={intent}: {missing}"

        # If user selected embeddings/hybrid, fallback to embeddings (semantic)
        if method in ("embeddings", "hybrid") and embeddings_retriever is not None:
            results = embeddings_retriever.search_similar_nodes(
                query_text=query_text,
                label="Product",
                top_k=10,
                embedding_property=embedding_prop,
            )
            return results, f"embeddings (fallback: missing {missing})", debug

        # Baseline-only: block retrieval gracefully
        return [], f"baseline blocked (missing {missing})", debug

    # Force baseline for structured intents
    if intent in CYPHER_ONLY_INTENTS and method == "embeddings":
        results, cypher, params = retriever.run_query(intent, entities, return_debug=True)
        debug["cypher"] = cypher
        debug["params"] = params
        return results, "baseline (forced by intent)", debug

    # Baseline
    if method == "baseline":
        results, cypher, params = retriever.run_query(intent, entities, return_debug=True)
        debug["cypher"] = cypher
        debug["params"] = params
        return results, "baseline", debug

    # Embeddings
    if method == "embeddings":
        if embeddings_retriever is None:
            return [], "embeddings unavailable (not initialized)", debug
        results = embeddings_retriever.search_similar_nodes(
            query_text=query_text,
            label="Product",
            top_k=10,
            embedding_property=embedding_prop,
        )
        return results, "embeddings", debug

    # Hybrid
    if method == "hybrid":
        # Even for hybrid, if intent is unstructured, we can still pull baseline
        cypher_results, cypher, params = retriever.run_query(intent, entities, return_debug=True)
        debug["cypher"] = cypher
        debug["params"] = params

        if embeddings_retriever is None:
            return cypher_results, "hybrid (embeddings unavailable)", debug

        emb_results = embeddings_retriever.search_similar_nodes(
            query_text=query_text,
            label="Product",
            top_k=max(10, len(cypher_results) if isinstance(cypher_results, list) else 10),
            embedding_property=embedding_prop,
        )
        merged = cypher_results + emb_results
        merged = dedupe_by_product_id(merged)
        return merged, "hybrid (deduped)", debug

    return [], "baseline", debug


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Graph-RAG Assistant", layout="wide")
st.title("🧠 Graph-RAG Assistant")
st.write("Milestone 3 – Knowledge Graph + LLM Integration")

ui_retrieval_choice = st.sidebar.selectbox(
    "Retrieval Method",
    ["Baseline (Cypher)", "Embeddings", "Hybrid"]
)
retrieval_method = map_ui_retrieval_method(ui_retrieval_choice)

embedding_model_choice = st.sidebar.selectbox(
    "Embedding Model (Embeddings / Hybrid)",
    ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]
)

# -----------------------------
# LLM Model (Supported-only dropdown)
# -----------------------------
SUPPORTED_LLM_OPTIONS = [
    "Gemma 2B (Fast, Deployed)",
    "Mistral 7B (Comparison)",
    "LLaMA 3 8B (Comparison)",
]

llm_model_mapping = {
    "Gemma 2B (Fast, Deployed)": "google/gemma-2-2b-it",
    "Mistral 7B (Comparison)": "mistralai/Mistral-7B-Instruct-v0.2",
    "LLaMA 3 8B (Comparison)": "meta-llama/Meta-Llama-3-8B-Instruct",
}


llm_model = st.sidebar.selectbox("LLM Model", SUPPORTED_LLM_OPTIONS)
selected_model_id = llm_model_mapping[llm_model]


user_query = st.text_input("Enter your question:", placeholder="e.g. Products in bebes category")

intent_classifier = IntentClassifier()
entity_extractor = EntityExtractor()
retriever = KGRetriever()

# Lazy init embeddings retriever ONLY when needed + cache per embedding model
embeddings_retriever = None
if retrieval_method in ("embeddings", "hybrid"):
    key = f"emb_retriever_{embedding_model_choice}"
    if key not in st.session_state:
        st.session_state[key] = EmbeddingsRetriever(embedding_model=embedding_model_choice)
    embeddings_retriever = st.session_state[key]

# Cleanup ONCE
def _cleanup():
    try:
        retriever.close()
    except Exception:
        pass

    for k, v in list(st.session_state.items()):
        if k.startswith("emb_retriever_"):
            try:
                v.close()
            except Exception:
                pass

atexit.register(_cleanup)

hf_token = os.getenv("HF_TOKEN", "")

llm = None
llm_error = None

try:
    llm = LLMGrounding(
        model=selected_model_id,
        hf_api_key=hf_token
    )

except Exception as e:
    llm_error = str(e)


st.sidebar.markdown("### Debug")
st.sidebar.write("HF_TOKEN loaded:", "✅" if hf_token else "❌")
st.sidebar.write("Selected LLM:", selected_model_id)
if llm_error:
    st.sidebar.error(llm_error)

embedding_prop = "embedding_mpnet" if "mpnet" in embedding_model_choice.lower() else "embedding_minilm"

if st.button("🔍 Ask") and user_query:
    intent = intent_classifier.predict(user_query)
    entities = entity_extractor.extract(user_query, intent)
    intent, entities = normalize_query_to_kg(intent, entities, user_query)
    entities = infer_numeric_thresholds(intent, entities, user_query)
    entities = normalize_category_entity_to_kg(entities)  # ✅ ADD THIS LINE


    st.markdown("### 🔍 Predicted Intent & Entities")
    st.write(f"Intent: `{intent}`")
    st.write(f"Entities: `{entities}`")

    t0 = time.time()
    kg_results, effective_method, debug_info = retrieve_with_debug(
        intent=intent,
        entities=entities,
        query_text=user_query,
        method=retrieval_method,
        retriever=retriever,
        embeddings_retriever=embeddings_retriever,
        embedding_prop=embedding_prop,
    )
    retrieval_latency = time.time() - t0

    # Add translation fields for English assistant requirement
    kg_results = add_translations(kg_results)

    st.markdown("### 🔗 Retrieved Results (Deterministic)")
    st.caption(f"Effective retrieval: **{effective_method}** | Retrieval latency: **{retrieval_latency:.2f}s**")

    if debug_info.get("param_error"):
        st.warning(debug_info["param_error"])

    if not kg_results:
        st.warning("No results found from KG retrieval.")
        st.stop()

    st.dataframe(lightweight_view_rows(kg_results, max_rows=25), use_container_width=True)

    # Show Cypher + params for screenshots (baseline/hybrid)
    if debug_info.get("cypher"):
        st.markdown("### 🧾 Cypher Query (Executed)")
        st.code(debug_info["cypher"], language="cypher")
        st.markdown("### 🧾 Cypher Params")
        st.code(debug_info.get("params", {}), language="json")

    st.markdown("### 🧾 Raw kg_results (debug / screenshots)")
    st.code(kg_results)

    # LLM grounded summary (no re-listing)
    if llm:
        t1 = time.time()
        summary = llm.summarize_results(
            user_query,
            kg_results,
            intent=intent,
            notes="UI already displayed the full list deterministically; summarize only."
        )
        llm_latency = time.time() - t1

        st.markdown("### ✅ LLM Grounded Summary (No re-listing)")
        st.success(summary)
        st.caption(
            f"LLM: {selected_model_id} | "
            f"Embedding: {embedding_model_choice if effective_method.startswith(('embeddings', 'hybrid')) else 'N/A'} | "
            f"LLM latency: {llm_latency:.2f}s"
        )
    else:
        st.warning("LLM not available (check HF_TOKEN).")
