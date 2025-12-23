import os
import json
from openai import OpenAI


class LLMGrounding:
    def __init__(self, model="google/gemma-2-2b-it", hf_api_key=None):
        self.model = model
        if hf_api_key is None:
            hf_api_key = os.getenv("HF_TOKEN")

        if not hf_api_key:
            raise ValueError("Hugging Face API key not provided. Set HF_TOKEN environment variable.")

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_api_key
        )

    @staticmethod
    def _safe_json(results, max_chars: int = 12000) -> str:
        s = json.dumps(results, ensure_ascii=False)
        return s[:max_chars]

    def summarize_results(self, query: str, kg_results: list, intent: str = "", notes: str = "") -> str:
        """
        Milestone requirement:
        - Structured prompt: Context / Persona / Task
        - Grounded: use ONLY KG_RESULTS
        - UI does deterministic listing; LLM only summarizes
        """
        kg_json = self._safe_json(kg_results)

        prompt = f"""
[CONTEXT]
You are given KG_RESULTS from a Neo4j knowledge graph retrieval system.
KG_RESULTS is the ONLY source of truth. If something is not in KG_RESULTS, you must not claim it.
KG_RESULTS is provided as JSON and may contain product IDs, categories, ratings, counts, delays, etc.
Some category labels may be non-English; if a translation is provided in the fields, prefer the English label.

[PERSONA]
You are a careful, concise data assistant for an academic demo.
You prioritize correctness and transparency.
You never hallucinate and you never omit constraints.

[TASK]
Given the USER_QUERY and KG_RESULTS:
1) Provide a short grounded summary of what KG_RESULTS shows.
2) Do NOT re-list every item in full (the UI already displays them).
3) Do NOT add new items not present in KG_RESULTS.
4) If KG_RESULTS is empty, say no results were found and what the user could try next.

Intent: {intent}
Notes (optional): {notes}

[KG_RESULTS JSON]
{kg_json}

[USER_QUERY]
{query}

Write the grounded summary:
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Only use provided KG_RESULTS. Be strict and grounded."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()
