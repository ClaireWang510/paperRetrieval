from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from scientific_resource_release.models import SemanticUnitConfig


logger = logging.getLogger(__name__)


class SemanticUnitLLMClient:
    def __init__(self, config: SemanticUnitConfig):
        self.config = config
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool((self.config.llm_api_key or "").strip())

    def _get_client(self):
        if not self.enabled:
            raise RuntimeError("LLM API key is not configured")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=(self.config.llm_api_key or "").strip(),
                base_url=(self.config.llm_base_url or "https://api.openai.com/v1").rstrip("/"),
                timeout=self.config.llm_timeout_seconds,
            )
        return self._client

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        use_json_mode: bool = True,
        call_name: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        client = self._get_client()
        for attempt in range(max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.config.llm_model or "gpt-4o-mini",
                    "messages": messages,
                    "temperature": 0.0,
                }
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                return content, input_tokens, output_tokens
            except Exception as exc:
                logger.warning(
                    "Semantic unit LLM call failed (%s, attempt %s/%s): %s",
                    call_name or "unknown",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Semantic unit LLM call failed after retries")


def parse_json_from_response(raw: str) -> Optional[Any]:
    text = (raw or "").strip()
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*([\[{][\s\S]*?[\]}])\s*```", text)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"[\[{][\s\S]*[\]}]", text)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None