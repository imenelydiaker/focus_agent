import os
from typing import Any

import requests


class TextEmbeddingClient:
    def __init__(self):
        self.endpoint = os.getenv("TEXT_EMBEDDING_ENDPOINT", "http://localhost:8080/embed")
        self.timeout_s = 30
        self.max_inputs_per_request = 64
        self.max_chars_per_request = 20_000

    def _post_embed(self, inputs: list[str]) -> list[Any]:
        response = requests.post(
            self.endpoint,
            json={"inputs": inputs},
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_s,
        )
        response.raise_for_status()

        payload = response.json()
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "embeddings" in payload:
            return payload["embeddings"]
        raise ValueError("Unexpected /embed response format.")

    def embed(self, inputs: list[str]) -> list[Any]:
        if not inputs:
            return []

        all_embeddings: list[Any] = []
        batch: list[str] = []
        batch_chars = 0

        for text in inputs:
            text_chars = len(text)

            would_exceed_count = len(batch) >= self.max_inputs_per_request
            would_exceed_chars = (batch_chars + text_chars) > self.max_chars_per_request and len(
                batch
            ) > 0

            if would_exceed_count or would_exceed_chars:
                all_embeddings.extend(self._post_embed(batch))
                batch = []
                batch_chars = 0

            batch.append(text)
            batch_chars += text_chars

        if batch:
            all_embeddings.extend(self._post_embed(batch))

        return all_embeddings
