"""Tests for retriever prompt construction.

Prompts are assembled locally and never sent anywhere here, so these tests need
no credentials and make no network calls.
"""

import numpy as np
import pytest

from focus_agent.retriever.focus_prompt import (
    BidExtractorPrompt,
    FocusDefenderPrompt,
    FocusPrompt,
    FocusPromptFlags,
    FocusSystemPrompt,
    NeutralFocusPrompt,
    RestrictiveFocusPrompt,
)


GOAL = "Upvote the newest post in the deeplearning subreddit"
TREE = "   1 RootWebArea 'Postmill'\n   2     [24] navigation '', visible"
HISTORY = "step 0: clicked [12]"


def flags(**overrides):
    base = dict(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
        sanitize_attacks=False,
    )
    base.update(overrides)
    return FocusPromptFlags(**base)


def build(cls=FocusPrompt, screenshot=None, extra_instruction="", **flag_overrides):
    return cls(
        goal=GOAL,
        tree=TREE,
        screenshot=screenshot,
        history=HISTORY,
        extra_instruction=extra_instruction,
        flags=flags(**flag_overrides),
    )


def user_text(prompt):
    return "\n".join(m["content"] for m in prompt.prompt if isinstance(m["content"], str))


class TestFocusSystemPrompt:
    def test_states_the_retriever_role(self):
        assert "extract" in FocusSystemPrompt().prompt.lower()


class TestFocusPromptMessages:
    def test_opens_with_a_system_message(self):
        assert build().prompt[0]["role"] == "system"

    def test_minimal_prompt_is_system_plus_user(self):
        assert len(build().prompt) == 2

    def test_includes_the_goal(self):
        assert GOAL in user_text(build())

    def test_includes_the_observation(self):
        assert TREE in user_text(build())

    def test_includes_the_answer_format_contract(self):
        # The parser downstream depends on this exact envelope.
        assert "<answer>" in user_text(build())

    def test_asks_for_reasoning(self):
        assert "<think>" in user_text(build())

    def test_extra_instruction_is_included_when_given(self):
        assert "BE CAREFUL" in user_text(build(extra_instruction="BE CAREFUL"))

    def test_no_extra_instruction_by_default(self):
        assert build().extra_instruction == ""

    def test_none_extra_instruction_becomes_empty_string(self):
        assert build(extra_instruction=None).extra_instruction == ""


class TestFocusPromptFlags:
    def test_history_is_omitted_by_default(self):
        assert HISTORY not in user_text(build())

    def test_history_is_included_when_enabled(self):
        assert HISTORY in user_text(build(use_history=True))

    def test_abstract_example_is_omitted_by_default(self):
        assert len(build().prompt) == 2

    def test_abstract_example_adds_a_message(self):
        assert len(build(use_abstract_example=True).prompt) == 3

    def test_concrete_example_adds_a_message(self):
        assert len(build(use_concrete_example=True).prompt) == 3

    def test_both_examples_add_two_messages(self):
        assert len(build(use_abstract_example=True, use_concrete_example=True).prompt) == 4

    def test_concrete_example_shows_a_worked_answer(self):
        text = user_text(build(use_concrete_example=True))
        assert "[(10,12), (123, 210)]" in text

    def test_screenshot_attaches_an_image_message(self):
        prompt = build(screenshot=np.zeros((4, 4, 3), dtype=np.uint8), use_screenshot=True)
        image_parts = [
            part
            for message in prompt.prompt
            if isinstance(message["content"], list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_sanitize_attacks_requests_attack_line_ranges(self):
        assert "<attack_lines>" in user_text(build(sanitize_attacks=True))

    def test_sanitize_attacks_is_off_by_default(self):
        assert "<attack_lines>" not in user_text(build())


class TestGetFullTextPrompt:
    def test_flattens_messages_into_one_string(self):
        assert GOAL in build().get_full_text_prompt()

    def test_includes_the_system_message(self):
        assert "extract" in build().get_full_text_prompt().lower()

    def test_skips_binary_image_payloads(self):
        prompt = build(screenshot=np.zeros((4, 4, 3), dtype=np.uint8), use_screenshot=True)
        text = prompt.get_full_text_prompt()
        assert "Here is a screenshot" in text
        assert "base64" not in text


class TestPromptVariants:
    def test_restrictive_asks_to_prune_aggressively(self):
        assert "restrictive" in user_text(build(RestrictiveFocusPrompt)).lower()

    def test_default_asks_to_be_extensive(self):
        assert "extensive" in user_text(build()).lower()

    def test_neutral_takes_neither_side(self):
        text = user_text(build(NeutralFocusPrompt)).lower()
        assert "be restrictive" not in text
        assert "be extensive" not in text

    def test_variants_still_carry_the_goal_and_tree(self):
        for cls in (RestrictiveFocusPrompt, NeutralFocusPrompt):
            text = user_text(build(cls))
            assert GOAL in text and TREE in text

    def test_variants_keep_the_answer_contract(self):
        for cls in (RestrictiveFocusPrompt, NeutralFocusPrompt):
            assert "<answer>" in user_text(build(cls))


class TestFocusDefenderPrompt:
    def test_warns_about_attacks(self):
        assert "attack" in user_text(build(FocusDefenderPrompt)).lower()

    def test_includes_the_goal_and_observation(self):
        text = user_text(build(FocusDefenderPrompt))
        assert GOAL in text and TREE in text

    def test_opens_with_a_system_message(self):
        assert build(FocusDefenderPrompt).prompt[0]["role"] == "system"

    def test_screenshot_attaches_an_image_message(self):
        prompt = build(
            FocusDefenderPrompt,
            screenshot=np.zeros((4, 4, 3), dtype=np.uint8),
            use_screenshot=True,
        )
        assert any(isinstance(m["content"], list) for m in prompt.prompt)


class TestBidExtractorPrompt:
    def test_builds_a_prompt_carrying_the_goal(self):
        assert GOAL in user_text(build(BidExtractorPrompt))

    def test_includes_the_observation(self):
        assert TREE in user_text(build(BidExtractorPrompt))
