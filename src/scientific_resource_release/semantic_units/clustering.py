from __future__ import annotations

import logging
from typing import List, Optional

from .llm_client import SemanticUnitLLMClient, parse_json_from_response
from .schemas import FactUnit


logger = logging.getLogger(__name__)

TARGET_MIN_CLUSTERS = 4
TARGET_MAX_CLUSTERS = 6


def _choose_n_clusters(n_facts: int) -> int:
    if n_facts <= TARGET_MIN_CLUSTERS:
        return max(1, n_facts)
    return max(1, min(TARGET_MAX_CLUSTERS, max(TARGET_MIN_CLUSTERS, n_facts // 5)))


def cluster_facts_kmeans(
    facts: List[FactUnit],
    n_clusters: Optional[int] = None,
) -> List[List[FactUnit]]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        raise ImportError("KMeans clustering requires scikit-learn")

    if not facts:
        return []

    statements = [fact.statement for fact in facts]
    n = len(statements)
    n_clusters = min(n_clusters or _choose_n_clusters(n), n)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(statements)
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(matrix)

    clusters = [[] for _ in range(n_clusters)]
    for fact, label in zip(facts, labels):
        clusters[int(label)].append(fact)
    return clusters


def cluster_facts_llm(
    facts: List[FactUnit],
    llm_client: Optional[SemanticUnitLLMClient] = None,
    n_clusters: Optional[int] = None,
) -> List[List[FactUnit]]:
    if not facts:
        return []
    if not llm_client or not llm_client.enabled:
        return cluster_facts_kmeans(facts, n_clusters)

    n = len(facts)
    n_clusters = min(n_clusters or _choose_n_clusters(n), n)
    numbered = "\n".join("%d. %s" % (index, fact.statement) for index, fact in enumerate(facts))
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at grouping atomic facts from one academic paper into theme-based clusters. "
                "Output only a JSON object with one key assignments mapping each fact index string to a cluster index."
            ),
        },
        {
            "role": "user",
            "content": (
                "There are %d facts. Group them into exactly %d clusters with cluster indices 0 to %d.\n\nFacts:\n%s\n\n"
                "Output one JSON object with key assignments mapping each fact index string to one integer cluster index."
            ) % (n, n_clusters, n_clusters - 1, numbered),
        },
    ]

    try:
        raw_response, _, _ = llm_client.call_llm(messages, use_json_mode=True, call_name="llm_clustering")
        data = parse_json_from_response(raw_response)
    except Exception as exc:
        logger.warning("LLM clustering failed: %s", exc)
        data = None

    assignments_raw = data.get("assignments") if isinstance(data, dict) else None
    if assignments_raw is None:
        return cluster_facts_kmeans(facts, n_clusters)

    if isinstance(assignments_raw, list):
        labels = [int(value) for value in assignments_raw[:n]]
    else:
        labels = [int(assignments_raw.get(str(index), 0)) for index in range(n)]

    clusters = [[] for _ in range(n_clusters)]
    for fact, label in zip(facts, labels):
        fixed_label = int(label) % n_clusters
        clusters[fixed_label].append(fact)
    return clusters


def cluster_facts(
    facts: List[FactUnit],
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    llm_client: Optional[SemanticUnitLLMClient] = None,
) -> List[List[FactUnit]]:
    if method == "llm":
        return cluster_facts_llm(facts, llm_client=llm_client, n_clusters=n_clusters)
    return cluster_facts_kmeans(facts, n_clusters=n_clusters)