from __future__ import annotations

from neo4j import GraphDatabase
from typing import Dict, Any, List, Optional, Tuple, Union

from retrieval.cypher_templates import CYPHER_TEMPLATES


def load_config(path: str = "config.txt") -> dict:
    config = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                config[k] = v
    return config


def get_driver(config_path: str = "config.txt"):
    config = load_config(config_path)
    return GraphDatabase.driver(
        config["URI"],
        auth=(config["USERNAME"], config["PASSWORD"])
    )


class KGRetriever:
    """
    Baseline (Cypher) KG retriever.

    - Standardized kg_results: adds source, intent, rank
    - Optionally returns cypher+params for UI screenshots (return_debug=True)
    """

    def __init__(self, neo4j_config_path: str = "config.txt"):
        self.config_path = neo4j_config_path
        self.driver = get_driver(neo4j_config_path)

    def close(self):
        self.driver.close()

    def run_query(
        self,
        intent: str,
        entities: Optional[Dict[str, Any]] = None,
        return_debug: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
        if entities is None:
            entities = {}

        if intent not in CYPHER_TEMPLATES:
            raise ValueError(f"No template for intent: {intent}")

        cypher = CYPHER_TEMPLATES[intent]
        params = dict(entities)

        rows: List[Dict[str, Any]] = []
        with self.driver.session() as session:
            records = session.run(cypher, **params)
            for record in records:
                rows.append(dict(record))

        standardized: List[Dict[str, Any]] = []
        for idx, r in enumerate(rows, start=1):
            out = {"source": "baseline", "intent": intent, "rank": idx}
            out.update(r)
            standardized.append(out)

        if return_debug:
            return standardized, cypher, params

        return standardized
