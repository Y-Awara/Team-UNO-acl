from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


class EmbeddingsRetriever:
    """
    Embedding-based retriever for Neo4j nodes.

    Milestone 3 GUARANTEES:
    - NEVER returns full nodes
    - NEVER returns embedding arrays
    - ALWAYS returns lightweight, standardized dicts
    - Safe even if Cypher is modified later
    """

    def __init__(
        self,
        embedding_model: str,
        neo4j_config_path: str = "config.txt",
        device: Optional[str] = None,
    ):
        self.model_name = embedding_model
        self.device = device or "cpu"

        # ✅ CRITICAL FIX:
        # Pass device into the constructor so SentenceTransformer handles device placement correctly.
        # Do NOT call .to() after initialization (can trigger meta tensor errors).
        try:
            self.model = SentenceTransformer(embedding_model, device=self.device)
        except Exception:
            # Hard fallback: CPU always stable for demos
            self.device = "cpu"
            self.model = SentenceTransformer(embedding_model, device="cpu")

        # Warmup (lightweight)
        _ = self.model.encode(["warmup"])

        self.driver = self._get_driver(neo4j_config_path)

    # -------------------
    # Neo4j connection
    # -------------------
    def _get_driver(self, path: str):
        config = {}
        with open(path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    config[k] = v

        return GraphDatabase.driver(
            config["URI"],
            auth=(config["USERNAME"], config["PASSWORD"]),
        )

    def close(self):
        self.driver.close()

    # -------------------
    # Embedding helper
    # -------------------
    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    # -------------------
    # Store embeddings (offline step)
    # -------------------
    def store_node_embeddings(
        self,
        label: str = "Product",
        node_id_property: str = "product_id",
        text_properties: List[str] = ["product_category_name"],
        embedding_property: str = "embedding",
        batch_size: int = 50,
        limit: int = 2000,
        skip_if_exists: bool = True,
    ):
        where_clause = ""
        if skip_if_exists:
            where_clause = f"WHERE n.{embedding_property} IS NULL"

        with self.driver.session() as session:
            query = f"""
            MATCH (n:{label})
            {where_clause}
            RETURN
                n.{node_id_property} AS node_id,
                {", ".join([f"n.{p} AS {p}" for p in text_properties])}
            LIMIT $limit
            """

            records = session.run(query, limit=limit)

            nodes = []
            for r in records:
                node_id = r["node_id"]
                text = " ".join(
                    str(r[p]) for p in text_properties if r.get(p) is not None
                )
                if text.strip():
                    nodes.append((node_id, text))

            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                print(f"Embedding batch {i // batch_size + 1}")

                for node_id, text in batch:
                    emb = self.embed_text(text)
                    session.run(
                        f"""
                        MATCH (n:{label} {{ {node_id_property}: $node_id }})
                        SET n.{embedding_property} = $embedding
                        """,
                        node_id=node_id,
                        embedding=emb,
                    )

    # -------------------
    # Similarity search (SAFE + LIGHTWEIGHT)
    # -------------------
    def search_similar_nodes(
        self,
        query_text: str,
        label: str = "Product",
        top_k: int = 5,
        embedding_property: str = "embedding",
        id_property: str = "product_id",
        category_property: str = "product_category_name",
    ) -> List[Dict[str, Any]]:
        """
        Returns standardized kg_results:

        [
          {
            "source": "embeddings",
            "rank": 1,
            "product_id": "...",
            "product_category_name": "...",
            "score": 0.87
          }
        ]
        """
        query_emb = self.embed_text(query_text)

        with self.driver.session() as session:
            cypher = f"""
            MATCH (n:{label})
            WHERE n.{embedding_property} IS NOT NULL
              AND n.{id_property} IS NOT NULL
            WITH
                n.{id_property} AS product_id,
                n.{category_property} AS product_category_name,
                gds.similarity.cosine(n.{embedding_property}, $query_emb) AS score
            RETURN
                product_id,
                product_category_name,
                score
            ORDER BY score DESC
            LIMIT $top_k
            """

            records = session.run(
                cypher,
                query_emb=query_emb,
                top_k=top_k,
            )

            allowed_keys = {
                "product_id",
                "product_category_name",
                "score",
            }

            results: List[Dict[str, Any]] = []
            for idx, r in enumerate(records, start=1):
                clean = {k: r.get(k) for k in allowed_keys}
                results.append(
                    {
                        "source": "embeddings",
                        "rank": idx,
                        **clean,
                    }
                )

            return results


if __name__ == "__main__":

    retriever_minilm = EmbeddingsRetriever("all-MiniLM-L6-v2")
    retriever_minilm.store_node_embeddings(
        embedding_property="embedding_minilm",
        limit=2000,
    )
    retriever_minilm.close()
    print("✅ MiniLM embeddings stored")

    retriever_mpnet = EmbeddingsRetriever("all-mpnet-base-v2")
    retriever_mpnet.store_node_embeddings(
        embedding_property="embedding_mpnet",
        limit=2000,
    )
    retriever_mpnet.close()
    print("✅ MPNet embeddings stored")
