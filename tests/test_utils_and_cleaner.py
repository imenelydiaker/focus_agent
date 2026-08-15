"""Tests for the summary/message helpers, the heuristic cleaner, and constants."""

import pytest

from focus_agent.constants import (
    FILTERS,
    OBS_PROMPT_AXTREE_PREFIX,
    OBS_PROMPT_HTML_PREFIX,
)
from focus_agent.utils import (
    clean_chat_messages,
    concat_user_messages,
    reformat_summary,
)


SUMMARY_PREFIX = "Here is a one-sentence summary of the content:"
ALT_PREFIX = "Here is a summary of the content in one sentence:"


class TestReformatSummary:
    def test_replaces_the_preamble_with_the_summary_itself(self):
        text = f"\t<collapsed 5 lines>\n\t{SUMMARY_PREFIX}\n\tThe page lists posts.\n\t</collapsed>"
        result = reformat_summary(text)
        assert SUMMARY_PREFIX not in result
        assert "The page lists posts." in result

    def test_keeps_the_surrounding_markers(self):
        text = f"\t<collapsed 5 lines>\n\t{SUMMARY_PREFIX}\n\tThe page lists posts.\n\t</collapsed>"
        result = reformat_summary(text)
        assert result.splitlines()[0].strip() == "<collapsed 5 lines>"
        assert result.splitlines()[-1].strip() == "</collapsed>"

    def test_collapses_preamble_and_summary_into_one_line(self):
        text = f"\t{SUMMARY_PREFIX}\n\tThe page lists posts."
        assert len(reformat_summary(text).splitlines()) == 1

    def test_preserves_indentation(self):
        text = f"\t{SUMMARY_PREFIX}\n\tThe page lists posts."
        assert reformat_summary(text).startswith("\t")

    def test_handles_the_alternate_preamble_wording(self):
        text = f"\t{ALT_PREFIX}\n\tThe page lists posts."
        result = reformat_summary(text)
        assert ALT_PREFIX not in result
        assert "The page lists posts." in result

    def test_text_without_a_preamble_is_left_alone(self):
        assert reformat_summary("line a\nline b") == "line a\nline b"

    def test_drops_a_blank_line(self):
        assert reformat_summary("a\n\nb") == "a\nb"

    def test_empty_input(self):
        assert reformat_summary("") == ""


class TestConcatUserMessages:
    def test_joins_plain_user_messages(self):
        messages = [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
        assert concat_user_messages(messages) == "first\nsecond"

    def test_ignores_system_messages(self):
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        assert concat_user_messages(messages) == "hi"

    def test_ignores_assistant_messages(self):
        messages = [{"role": "assistant", "content": "reply"}, {"role": "user", "content": "hi"}]
        assert concat_user_messages(messages) == "hi"

    def test_extracts_text_parts_from_multipart_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
                ],
            }
        ]
        assert concat_user_messages(messages) == "look at this"

    def test_skips_image_payloads(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}}
                ],
            }
        ]
        assert "base64" not in concat_user_messages(messages)

    def test_empty_list_yields_empty_string(self):
        assert concat_user_messages([]) == ""


class TestCleanChatMessages:
    def test_drops_assistant_turns(self):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "reply"}]
        assert clean_chat_messages(messages) == [{"role": "user", "content": "hi"}]

    def test_drops_the_missing_action_retry_prompt(self):
        messages = [{"role": "user", "content": "Missing the key <action> in the answer."}]
        assert clean_chat_messages(messages) == []

    def test_drops_the_missing_think_retry_prompt(self):
        messages = [{"role": "user", "content": "Missing the key <think> in the answer"}]
        assert clean_chat_messages(messages) == []

    def test_keeps_ordinary_user_turns(self):
        messages = [{"role": "user", "content": "upvote the newest post"}]
        assert clean_chat_messages(messages) == messages

    def test_keeps_system_turns(self):
        messages = [{"role": "system", "content": "you are an agent"}]
        assert clean_chat_messages(messages) == messages

    def test_preserves_order_of_surviving_turns(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "noise"},
            {"role": "user", "content": "second"},
        ]
        assert [m["content"] for m in clean_chat_messages(messages)] == ["first", "second"]

    def test_empty_list_yields_empty_list(self):
        assert clean_chat_messages([]) == []


class TestConstants:
    def test_filters_map_short_names_to_model_ids(self):
        assert FILTERS["4o"] == "gpt-4o"

    def test_filter_values_are_strings(self):
        assert all(isinstance(v, str) for v in FILTERS.values())

    def test_axtree_prefix_explains_the_bid_convention(self):
        assert "bid" in OBS_PROMPT_AXTREE_PREFIX

    def test_axtree_prefix_labels_the_section(self):
        assert "## AXTree:" in OBS_PROMPT_AXTREE_PREFIX

    def test_html_prefix_labels_the_section(self):
        assert "## HTML:" in OBS_PROMPT_HTML_PREFIX


class TestGenericAgentHeuristicCleaner:
    """The cleaner strips bid-less lines on top of the parent's preprocessing.

    GenericAgent's constructor builds a live chat model, so the base class is
    stubbed out — this keeps the test offline and credential-free.
    """

    @pytest.fixture
    def agent(self, monkeypatch):
        from agentlab.agents.generic_agent.generic_agent import GenericAgent
        from focus_agent.agents.generic_agent_heuristic_cleaner import (
            GenericAgentHeuristicCleaner,
        )

        monkeypatch.setattr(GenericAgent, "__init__", lambda self, *a, **k: None)
        monkeypatch.setattr(GenericAgent, "obs_preprocessor", lambda self, obs: obs)
        return GenericAgentHeuristicCleaner(None, None)

    def test_keeps_lines_with_a_bid(self, agent):
        obs = {"axtree_txt": "RootWebArea 'App'\n    [1] button 'Go'"}
        assert "[1] button 'Go'" in agent.obs_preprocessor(obs)["axtree_txt"]

    def test_drops_lines_without_a_bid(self, agent):
        obs = {"axtree_txt": "RootWebArea 'App'\n    [1] button 'Go'\n    StaticText 'noise'"}
        assert "noise" not in agent.obs_preprocessor(obs)["axtree_txt"]

    def test_keeps_the_root_line(self, agent):
        obs = {"axtree_txt": "RootWebArea 'App'\n    StaticText 'noise'"}
        assert agent.obs_preprocessor(obs)["axtree_txt"].startswith("RootWebArea 'App'")

    def test_returns_the_same_observation_object(self, agent):
        obs = {"axtree_txt": "RootWebArea 'App'", "other": "untouched"}
        result = agent.obs_preprocessor(obs)
        assert result is obs
        assert result["other"] == "untouched"
