from llm_grounding import LLMGrounding
import os

kg_results = [
    {
        "source": "baseline",
        "intent": "PRODUCT_BY_CATEGORY",
        "rank": 1,
        "product_id": "d70f38e7f79c630f8ea00c993897042c",
        "product_category_name": "bebes"
    }
]

user_query = "Show me products in bebes category"

llm = LLMGrounding(hf_api_key=os.getenv("HF_TOKEN"), model="google/gemma-2-2b-it")

summary = llm.summarize_results(user_query, kg_results, intent="PRODUCT_BY_CATEGORY")

print("LLM Summary:")
print(summary)
