from focus_agent.retriever.text_embedding_client import TextEmbeddingClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_embed_batches_by_max_inputs(monkeypatch):
    posted_batches = []

    def fake_post(_endpoint, json, headers, timeout):
        _ = headers, timeout
        batch = json["inputs"]
        posted_batches.append(batch)
        payload = [[float(i)] for i, _text in enumerate(batch)]
        return _FakeResponse(payload)

    monkeypatch.setattr("focus_agent.retriever.text_embedding_client.requests.post", fake_post)

    client = TextEmbeddingClient()
    client.max_inputs_per_request = 2
    client.max_chars_per_request = 10_000

    inputs = ["a", "b", "c", "d", "e"]
    embeddings = client.embed(inputs)

    assert posted_batches == [["a", "b"], ["c", "d"], ["e"]]
    assert len(embeddings) == len(inputs)


def test_embed_batches_by_max_chars(monkeypatch):
    posted_batches = []

    def fake_post(_endpoint, json, headers, timeout):
        _ = headers, timeout
        batch = json["inputs"]
        posted_batches.append(batch)
        payload = [[float(len(text))] for text in batch]
        return _FakeResponse(payload)

    monkeypatch.setattr("focus_agent.retriever.text_embedding_client.requests.post", fake_post)

    client = TextEmbeddingClient()
    client.max_inputs_per_request = 64
    client.max_chars_per_request = 5

    inputs = ["abcd", "efgh", "ijklmnop"]
    embeddings = client.embed(inputs)

    assert posted_batches == [["abcd"], ["efgh"], ["ijklmnop"]]
    assert [item[0] for item in embeddings] == [4.0, 4.0, 8.0]


def test_embed_empty_inputs_skips_request(monkeypatch):
    calls = 0

    def fake_post(_endpoint, json, headers, timeout):
        _ = _endpoint, json, headers, timeout
        nonlocal calls
        calls += 1
        return _FakeResponse([])

    monkeypatch.setattr("focus_agent.retriever.text_embedding_client.requests.post", fake_post)

    client = TextEmbeddingClient()
    embeddings = client.embed([])

    assert embeddings == []
    assert calls == 0
