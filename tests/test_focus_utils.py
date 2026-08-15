"""Tests for the tree-pruning utilities that back FocusAgent's observation step."""

import pytest

from focus_agent.retriever.focus_utils import FocusUtils


TREE = """RootWebArea 'Postmill', focused
    [24] navigation '', visible
        [25] link 'Jump to main content', clickable
        [26] link 'Jump to sidebar', clickable
    [27] navigation '', visible
        [30] link 'Home', clickable, visible"""


class TestRemoveLines:
    def test_keeps_only_selected_lines(self):
        result = FocusUtils.remove_lines(TREE, [1, 2])
        lines = result.splitlines()
        assert lines[0] == "RootWebArea 'Postmill', focused"
        assert lines[1] == "    [24] navigation '', visible"

    def test_line_numbers_are_1_indexed(self):
        # Selecting 1 must yield the first line, not the second.
        result = FocusUtils.remove_lines(TREE, [1])
        assert result.splitlines()[0] == "RootWebArea 'Postmill', focused"

    def test_dropped_span_is_replaced_by_a_counted_placeholder(self):
        result = FocusUtils.remove_lines(TREE, [1])
        # Lines 2-6 are dropped as one contiguous run of 5.
        assert "... pruned 5 lines ..." in result

    def test_separate_gaps_get_separate_placeholders(self):
        result = FocusUtils.remove_lines(TREE, [1, 4])
        assert result.count("... pruned") == 2
        assert "... pruned 2 lines ..." in result  # lines 2-3
        assert "... pruned 2 lines ..." in result  # lines 5-6

    def test_selecting_every_line_is_a_no_op(self):
        result = FocusUtils.remove_lines(TREE, list(range(1, 7)))
        assert result == TREE
        assert "pruned" not in result

    def test_selecting_nothing_collapses_the_whole_tree(self):
        result = FocusUtils.remove_lines(TREE, [])
        assert result == "... pruned 6 lines ..."

    def test_out_of_range_line_numbers_are_ignored(self):
        result = FocusUtils.remove_lines(TREE, [1, 999])
        assert "... pruned 5 lines ..." in result

    def test_empty_tree_yields_empty_string(self):
        assert FocusUtils.remove_lines("", [1]) == ""

    def test_pruning_actually_shrinks_the_observation(self):
        result = FocusUtils.remove_lines(TREE, [1, 2])
        assert len(result) < len(TREE)


class TestRemoveLinesKeepStructure:
    def test_bid_strategy_keeps_the_bid_and_drops_the_rest(self):
        result = FocusUtils.remove_lines_keep_structure(TREE, [1], strategy="bid")
        lines = result.splitlines()
        assert lines[0] == "RootWebArea 'Postmill', focused"  # kept verbatim
        assert lines[1] == "    [24] ... removed ..."

    def test_structure_is_preserved_line_for_line(self):
        # Unlike remove_lines, every input line survives as one output line.
        result = FocusUtils.remove_lines_keep_structure(TREE, [1], strategy="bid")
        assert len(result.splitlines()) == len(TREE.splitlines())

    def test_indentation_is_preserved(self):
        result = FocusUtils.remove_lines_keep_structure(TREE, [1], strategy="bid")
        # Line 3 is nested two levels deep in the source tree.
        assert result.splitlines()[2].startswith("        ")

    def test_bid_role_strategy_keeps_both_bid_and_role(self):
        result = FocusUtils.remove_lines_keep_structure(TREE, [1], strategy="bid+role")
        assert result.splitlines()[1] == "    [24] navigation ... removed ..."

    def test_bid_role_strategy_keeps_role_only_when_line_has_no_bid(self):
        tree = "RootWebArea 'Postmill', focused\n    StaticText 'hello'"
        result = FocusUtils.remove_lines_keep_structure(tree, [1], strategy="bid+role")
        assert result.splitlines()[1] == "    StaticText"

    def test_selected_lines_are_untouched(self):
        result = FocusUtils.remove_lines_keep_structure(TREE, [2], strategy="bid")
        assert result.splitlines()[1] == "    [24] navigation '', visible"

    def test_role_only_strategy_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            FocusUtils.remove_lines_keep_structure(TREE, [1], strategy="role")
