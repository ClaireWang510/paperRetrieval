from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class SemanticScholarClient:
    timeout_seconds: int = 20

    def fetch_arxiv_metadata(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        fields = "citationCount,influentialCitationCount,fieldsOfStudy,publicationDate,venue"
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"ARXIV:{arxiv_id}?fields={fields}"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "scientific-resource-release/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return {
                    "citationCount": payload.get("citationCount"),
                    "influentialCitationCount": payload.get("influentialCitationCount"),
                    "fieldsOfStudy": payload.get("fieldsOfStudy") or [],
                    "published_date": payload.get("publicationDate"),
                    "venue": payload.get("venue"),
                }
        except urllib.error.HTTPError as exc:
            logger.warning("Semantic Scholar HTTP error for %s: %s", arxiv_id, exc)
            return None
        except urllib.error.URLError as exc:
            logger.warning("Semantic Scholar URL error for %s: %s", arxiv_id, exc)
            return None
        except Exception as exc:  # pragma: no cover
            logger.warning("Semantic Scholar unexpected error for %s: %s", arxiv_id, exc)
            return None
