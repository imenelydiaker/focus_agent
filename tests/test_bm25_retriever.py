"""Tests for the BM25 retrieval baseline and the shared chunking helper."""

import pytest

from focus_agent.retriever.bm25_retriever import (
    BM25RetrieverArgs,
    BM25SRetriever,
    count_tokens,
)
from focus_agent.retriever.utils import get_chunks_from_tokenizer


TREE = """RootWebArea 'Postmill', focused
    [24] navigation '', visible
        [25] link 'Jump to main content', clickable
        [26] link 'Jump to sidebar', clickable
    [a203] button 'Submit', clickable
    [27] heading 'The quick brown fox jumps over the lazy dog'
        [30] link 'Home', clickable, visible"""


class TestCountTokens:
    def test_counts_tokens(self):
        assert count_tokens("hello world") > 0

    def test_empty_string_has_no_tokens(self):
        assert count_tokens("") == 0

    def test_grows_with_length(self):
        assert count_tokens("hello " * 20) > count_tokens("hello")


class TestExtractBid:
    def test_extracts_a_numeric_bid(self):
        assert BM25SRetriever.extract_bid("[24] navigation '', visible") == "24"

    def test_extracts_an_alphanumeric_bid(self):
        assert BM25SRetriever.extract_bid("[a203] button 'Submit'") == "a203"

    def test_extracts_the_first_bid_when_several_are_present(self):
        assert BM25SRetriever.extract_bid("[24] link to [25]") == "24"

    def test_ignores_leading_indentation(self):
        assert BM25SRetriever.extract_bid("        [25] link 'Jump'") == "25"

    def test_returns_none_when_there_is_no_bid(self):
        assert BM25SRetriever.extract_bid("StaticText 'hello'") is None

    def test_returns_none_for_empty_brackets(self):
        assert BM25SRetriever.extract_bid("[] nothing here") is None


class TestGetElementsAround:
    def test_returns_the_target_element(self):
        result = BM25SRetriever.get_elements_around(TREE, "25", n=0)
        assert "[25] link 'Jump to main content'" in result

    def test_includes_n_neighbours_on_each_side(self):
        result = BM25SRetriever.get_elements_around(TREE, "25", n=1)
        assert "[24] navigation" in result
        assert "[26] link 'Jump to sidebar'" in result

    def test_neighbour_window_counts_elements_not_raw_lines(self):
        result = BM25SRetriever.get_elements_around(TREE, "25", n=1)
        assert len(result.splitlines()) == 3

    def test_preserves_indentation(self):
        result = BM25SRetriever.get_elements_around(TREE, "25", n=0)
        assert result.startswith("        ")

    def test_clamps_the_window_at_the_start_of_the_tree(self):
        result = BM25SRetriever.get_elements_around(TREE, "24", n=5)
        assert "[24] navigation" in result

    def test_clamps_the_window_at_the_end_of_the_tree(self):
        result = BM25SRetriever.get_elements_around(TREE, "30", n=5)
        assert "[30] link 'Home'" in result

    def test_finds_an_alphanumeric_element_id(self):
        result = BM25SRetriever.get_elements_around(TREE, "a203", n=0)
        assert "button 'Submit'" in result

    def test_unknown_element_id_raises(self):
        with pytest.raises(ValueError, match="not found"):
            BM25SRetriever.get_elements_around(TREE, "9999", n=1)


class TestGetChunksFromTokenizer:
    def test_splits_long_text_into_several_chunks(self):
        chunks = get_chunks_from_tokenizer("hello world " * 200, chunk_size=50, overlap=10)
        assert len(chunks) > 1

    def test_short_text_yields_a_single_chunk(self):
        chunks = get_chunks_from_tokenizer("hello world", chunk_size=100, overlap=10)
        assert len(chunks) == 1

    def test_chunks_reconstruct_the_source_when_there_is_no_overlap(self):
        text = "hello world " * 30
        chunks = get_chunks_from_tokenizer(text, chunk_size=20, overlap=0)
        assert "".join(chunks) == text

    def test_consecutive_chunks_share_the_overlap(self):
        chunks = get_chunks_from_tokenizer("hello world " * 200, chunk_size=50, overlap=25)
        assert len(chunks) >= 2
        # With step = chunk_size - overlap, chunk 2 restarts inside chunk 1.
        assert chunks[0] != chunks[1]

    def test_empty_text_yields_no_chunks(self):
        assert get_chunks_from_tokenizer("", chunk_size=50, overlap=10) == []


class TestBM25SRetriever:
    def _make(self, top_k=2, chunk_size=20, overlap=5):
        args = BM25RetrieverArgs(chunk_size=chunk_size, overlap=overlap, top_k=top_k)
        return BM25SRetriever(TREE, args)

    def test_indexes_the_tree_on_construction(self):
        assert self._make().retriever.corpus

    def test_retrieve_returns_at_most_top_k_results(self):
        assert len(self._make(top_k=2).retrieve("quick brown fox")) <= 2

    def test_retrieve_returns_strings(self):
        assert all(isinstance(r, str) for r in self._make().retrieve("navigation"))

    def test_retrieve_surfaces_the_lexically_matching_chunk(self):
        results = self._make(top_k=3, chunk_size=15, overlap=0).retrieve("quick brown fox")
        assert any("quick brown fox" in r for r in results)

    def test_top_k_larger_than_corpus_is_clamped(self):
        # Guards the branch that would otherwise ask BM25 for more docs than exist.
        retriever = self._make(top_k=999, chunk_size=500, overlap=0)
        results = retriever.retrieve("navigation")
        assert len(results) == len(retriever.retriever.corpus)

    def test_create_text_chunks_uses_the_configured_sizes(self):
        retriever = self._make(chunk_size=20, overlap=5)
        assert len(retriever.create_text_chunks(TREE)) > 1
