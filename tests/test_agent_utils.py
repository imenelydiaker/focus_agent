"""Tests for the AxTree helpers used to build the retriever's prompt."""

from focus_agent.agents.utils import (
    add_line_numbers_to_tree,
    get_nb_tokens,
    remove_no_bid_lines,
)


TREE = """RootWebArea 'Postmill', focused
    [24] navigation '', visible
        [25] link 'Jump to main content', clickable"""


class TestAddLineNumbersToTree:
    """The retriever answers in line numbers, so numbering must be exact."""

    def test_numbering_starts_at_one(self):
        result = add_line_numbers_to_tree(TREE)
        assert result.splitlines()[0].startswith("   1 ")

    def test_every_line_is_numbered_consecutively(self):
        result = add_line_numbers_to_tree(TREE).splitlines()
        assert [line.split()[0] for line in result] == ["1", "2", "3"]

    def test_line_content_is_preserved_after_the_number(self):
        result = add_line_numbers_to_tree(TREE).splitlines()
        assert result[0].endswith("RootWebArea 'Postmill', focused")

    def test_indentation_is_preserved(self):
        result = add_line_numbers_to_tree(TREE).splitlines()
        assert "    [24] navigation" in result[1]
        assert "        [25] link" in result[2]

    def test_line_count_is_unchanged(self):
        assert len(add_line_numbers_to_tree(TREE).splitlines()) == len(TREE.splitlines())

    def test_numbers_are_right_aligned_to_four_columns(self):
        # Keeps the tree readable past line 999.
        many = "\n".join(f"line {i}" for i in range(1, 12))
        result = add_line_numbers_to_tree(many).splitlines()
        assert result[0].startswith("   1 ")
        assert result[9].startswith("  10 ")

    def test_surrounding_blank_lines_are_stripped(self):
        result = add_line_numbers_to_tree("\n\nRootWebArea\n\n")
        assert result == "   1 RootWebArea"

    def test_numbering_round_trips_with_pruning(self):
        # A range the retriever reports must select the line a human would count.
        from focus_agent.retriever.focus_utils import FocusUtils

        numbered = add_line_numbers_to_tree(TREE)
        assert numbered.splitlines()[1].split()[0] == "2"
        kept = FocusUtils.remove_lines(TREE, [2])
        assert kept.splitlines()[1] == "    [24] navigation '', visible"


class TestRemoveNoBidLines:
    def test_keeps_lines_that_have_a_bid(self):
        result = remove_no_bid_lines(TREE)
        assert "[24] navigation" in result
        assert "[25] link" in result

    def test_drops_lines_with_no_bid(self):
        tree = "RootWebArea 'Postmill'\n    [24] navigation\n    StaticText 'hello'"
        result = remove_no_bid_lines(tree)
        assert "StaticText" not in result

    def test_always_keeps_the_root_line(self):
        # The root carries no bid but anchors the tree.
        tree = "RootWebArea 'Postmill'\n    StaticText 'hello'"
        assert remove_no_bid_lines(tree).splitlines()[0] == "RootWebArea 'Postmill'"

    def test_root_is_kept_exactly_once(self):
        result = remove_no_bid_lines(TREE)
        assert result.count("RootWebArea") == 1

    def test_indentation_is_preserved(self):
        result = remove_no_bid_lines(TREE)
        assert "        [25] link 'Jump to main content', clickable" in result

    def test_single_line_tree_survives(self):
        assert remove_no_bid_lines("RootWebArea 'Postmill'") == "RootWebArea 'Postmill'"

    def test_requires_both_brackets_not_just_a_closing_one(self):
        # Regression: the check was `"[" and "]" in line`, which Python folds to
        # `"]" in line`, so a stray closing bracket was enough to keep a line.
        tree = "RootWebArea 'Postmill'\n    StaticText 'a] stray bracket'"
        assert "stray bracket" not in remove_no_bid_lines(tree)

    def test_stray_opening_bracket_alone_is_not_a_bid(self):
        tree = "RootWebArea 'Postmill'\n    StaticText 'a [ stray bracket'"
        assert "stray bracket" not in remove_no_bid_lines(tree)

    def test_empty_input_does_not_raise(self):
        # Regression: indexing splitlines()[0] crashed on empty input.
        assert remove_no_bid_lines("") == ""


class TestGetNbTokens:
    def test_counts_tokens_in_a_string(self):
        assert get_nb_tokens("hello world") > 0

    def test_empty_string_has_no_tokens(self):
        assert get_nb_tokens("") == 0

    def test_longer_text_costs_more_tokens(self):
        assert get_nb_tokens("hello world " * 10) > get_nb_tokens("hello world")

    def test_measures_the_saving_from_pruning(self):
        # The metric the whole approach is judged on.
        from focus_agent.retriever.focus_utils import FocusUtils

        pruned = FocusUtils.remove_lines(TREE, [1])
        assert get_nb_tokens(pruned) < get_nb_tokens(TREE)
