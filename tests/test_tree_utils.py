"""Tests for the AxTree data structure: parsing, indexing, and restructuring.

Pure tree manipulation — no LLM, no network, no credentials. The one method
that takes a summarizer is exercised with a plain stub callable.
"""

import pytest

from focus_agent.agents.tree_utils import AxTree, Node


TREE = """RootWebArea 'Postmill', focused
    [24] navigation 'Main', visible
        [25] link 'Jump to main content', clickable
        [26] link 'Jump to sidebar', clickable
    [27] heading 'Featured', level=2
        [30] link 'Home', clickable, visible"""


@pytest.fixture
def tree():
    return AxTree(TREE)


@pytest.fixture
def numbered_tree():
    return AxTree(TREE, add_line_numbers=True)


class TestNode:
    def test_add_child_links_the_node(self):
        parent, child = Node(node_id="1", line="a"), Node(node_id="2", line="b")
        parent.add_child(child)
        assert parent.children == [child]

    def test_children_default_to_an_empty_list(self):
        assert Node(node_id="1", line="a").children == []

    def test_nodes_do_not_share_a_mutable_default(self):
        a, b = Node(node_id="1", line="a"), Node(node_id="2", line="b")
        a.add_child(Node(node_id="3", line="c"))
        assert b.children == []

    def test_attributes_default_to_an_empty_dict(self):
        assert Node(node_id="1", line="a").attributes == {}

    def test_counts_itself_when_it_has_no_children(self):
        assert Node(node_id="1", line="a").get_nb_nodes_in_subtree() == 1

    def test_counts_the_whole_subtree(self, tree):
        assert tree.root.get_nb_nodes_in_subtree() == 6

    def test_preorder_traversal_visits_parents_before_children(self, tree):
        ids = [n.node_id for n in tree.root.preorder_traversal()]
        assert ids == ["Root", "24", "25", "26", "27", "30"]

    def test_to_string_round_trips_the_source_lines(self, tree):
        assert tree.root.to_string().rstrip("\n") == TREE

    def test_to_string_includes_line_numbers_when_present(self, numbered_tree):
        assert numbered_tree.root.to_string().startswith("1 RootWebArea")

    def test_token_count_of_a_leaf_is_its_own(self):
        node = Node(line="[1] button 'Go'")
        assert node.get_n_tokens_subtree() > 0

    def test_token_count_accumulates_over_the_subtree(self, tree):
        root_only = Node(line=tree.root.line)
        assert tree.root.get_n_tokens_subtree() > root_only.get_n_tokens_subtree()

    def test_display_prints_every_node(self, tree, capsys):
        tree.root.display()
        assert len(capsys.readouterr().out.strip().splitlines()) == 6

    def test_display_axtree_prints_the_original_lines(self, tree, capsys):
        tree.root.display_axtree()
        assert "RootWebArea 'Postmill', focused" in capsys.readouterr().out


class TestParsing:
    def test_builds_one_node_per_line(self, tree):
        assert len(tree) == 6

    def test_indentation_becomes_hierarchy(self, tree):
        assert [c.node_id for c in tree.root.children] == ["24", "27"]
        assert [c.node_id for c in tree.root.children[0].children] == ["25", "26"]

    def test_extracts_the_bid(self, tree):
        assert tree.get_node_by_id("25") is not None

    def test_extracts_the_role(self, tree):
        assert tree.get_node_by_id("25").role == "link"

    def test_extracts_the_accessible_name(self, tree):
        assert tree.get_node_by_id("25").name == "Jump to main content"

    def test_unlabelled_root_is_named_root(self, tree):
        assert tree.root.node_id == "Root"

    def test_root_role_and_name_are_parsed(self, tree):
        assert tree.root.role == "RootWebArea"
        assert tree.root.name == "Postmill"

    def test_keeps_the_original_line(self, tree):
        assert tree.get_node_by_id("25").line == "        [25] link 'Jump to main content', clickable"

    def test_counts_tokens_per_node(self, tree):
        assert tree.get_node_by_id("25").n_tokens > 0

    def test_parses_attributes_that_end_in_a_delimiter(self):
        t = AxTree("RootWebArea 'App'\n    [1] textbox 'Search', focused=True, visible")
        assert t.get_node_by_id("1").attributes == {"focused": "True"}

    def test_line_numbers_are_off_by_default(self, tree):
        assert tree.root.line_number is None

    def test_line_numbers_are_1_indexed_when_enabled(self, numbered_tree):
        assert numbered_tree.root.line_number == 1
        assert numbered_tree.get_node_by_id("24").line_number == 2

    def test_trailing_whitespace_does_not_create_nodes(self):
        assert len(AxTree(TREE + "\n\n  ")) == 6

    def test_single_line_tree(self):
        t = AxTree("RootWebArea 'App'")
        assert len(t) == 1
        assert t.root.children == []


class TestLookups:
    def test_get_node_by_id(self, tree):
        assert tree.get_node_by_id("30").name == "Home"

    def test_get_node_by_id_returns_none_when_absent(self, tree):
        assert tree.get_node_by_id("9999") is None

    def test_get_node_by_line_number(self, numbered_tree):
        assert numbered_tree.get_node_by_line_number(3).node_id == "25"

    def test_get_node_by_line_number_returns_none_when_absent(self, numbered_tree):
        assert numbered_tree.get_node_by_line_number(999) is None

    def test_line_number_lookup_is_empty_without_numbering(self, tree):
        assert tree.get_node_by_line_number(1) is None

    def test_len_matches_traversal(self, tree):
        assert len(tree) == len(tree.root.preorder_traversal())


class TestReplaceNode:
    def test_swaps_the_targeted_node(self, tree):
        tree.replace_node("25", Node(node_id="25", role="link", line="        [25] REPLACED"))
        assert "REPLACED" in tree.root.to_string()

    def test_leaves_siblings_alone(self, tree):
        tree.replace_node("25", Node(node_id="25", line="        [25] REPLACED"))
        assert "[26] link 'Jump to sidebar'" in tree.root.to_string()

    def test_replacing_a_subtree_root_drops_its_children(self, tree):
        tree.replace_node("24", Node(node_id="24", line="    [24] COLLAPSED"))
        assert "[25]" not in tree.root.to_string()

    def test_unknown_id_leaves_the_tree_unchanged(self, tree):
        before = tree.root.to_string()
        tree.replace_node("9999", Node(node_id="9999", line="nope"))
        assert tree.root.to_string() == before

    def test_can_replace_the_root(self, tree):
        tree.replace_node("Root", Node(node_id="Root", line="NEW ROOT"))
        assert tree.root.to_string() == "NEW ROOT\n"


class TestCollapsedNodes:
    COLLAPSED = (
        "RootWebArea 'App'\n"
        "    [1] generic ''\n"
        "        [2] div 'collapsed content'\n"
        "    [3] button 'Go'"
    )

    def test_reports_the_parent_of_a_collapsed_child(self):
        t = AxTree(self.COLLAPSED)
        assert [n.node_id for n, _ in t.get_collapsed_nodes()] == ["1"]

    def test_reports_the_level_of_the_parent(self):
        t = AxTree(self.COLLAPSED)
        assert t.get_collapsed_nodes()[0][1] == 1

    def test_ids_variant_returns_bare_ids(self):
        assert AxTree(self.COLLAPSED).get_collapsed_nodes_ids() == ["1"]

    def test_no_collapsed_markers_yields_nothing(self, tree):
        assert tree.get_collapsed_nodes() == []
        assert tree.get_collapsed_nodes_ids() == []


class TestSummarizeAndReplace:
    def test_replaces_the_subtree_with_a_summary(self, tree):
        sub = tree.get_node_by_id("24")
        _, summarized = tree.summarize_and_replace(tree, sub, summarizer=lambda text: "SUMMARY")
        assert "SUMMARY" in summarized.to_string()

    def test_summarizer_receives_the_subtree_text(self, tree):
        seen = {}
        tree.summarize_and_replace(
            tree, tree.get_node_by_id("24"), summarizer=lambda t: seen.setdefault("text", t)
        )
        assert "[25] link 'Jump to main content'" in seen["text"]

    def test_records_how_many_lines_were_collapsed(self, tree):
        # The navigation subtree has two children.
        _, summarized = tree.summarize_and_replace(
            tree, tree.get_node_by_id("24"), summarizer=lambda t: "S"
        )
        assert "<collapsed 2 lines>" in summarized.to_string()

    def test_emits_a_closing_marker(self, tree):
        _, summarized = tree.summarize_and_replace(
            tree, tree.get_node_by_id("24"), summarizer=lambda t: "S"
        )
        assert "</collapsed>" in summarized.to_string()

    def test_the_original_tree_is_updated_in_place(self, tree):
        tree.summarize_and_replace(tree, tree.get_node_by_id("24"), summarizer=lambda t: "SUMMARY")
        assert "SUMMARY" in tree.root.to_string()

    def test_untouched_branches_survive(self, tree):
        tree.summarize_and_replace(tree, tree.get_node_by_id("24"), summarizer=lambda t: "S")
        assert "[30] link 'Home'" in tree.root.to_string()


class TestChunkTree:
    BIG = "RootWebArea 'App'\n" + "\n".join(
        f"    [{i}] section 'S{i}'\n"
        + "\n".join(f"        [{i}{j}] link 'L{i}{j}'" for j in range(3))
        for i in range(1, 4)
    )

    def test_splits_a_large_tree_into_subtrees(self):
        chunks = AxTree(self.BIG).chunk_tree(max_n_descendants=4, min_n_tokens_of_chunk=1)
        assert [c.node_id for c in chunks] == ["1", "2", "3"]

    def test_chunks_are_nodes_of_the_tree(self):
        chunks = AxTree(self.BIG).chunk_tree(max_n_descendants=4, min_n_tokens_of_chunk=1)
        assert all(isinstance(c, Node) for c in chunks)

    def test_a_tree_that_fits_yields_the_root_itself(self, tree):
        chunks = tree.chunk_tree(max_n_descendants=100, min_n_tokens_of_chunk=1)
        assert chunks == [tree.root]

    def test_token_floor_filters_out_small_chunks(self):
        chunks = AxTree(self.BIG).chunk_tree(max_n_descendants=4, min_n_tokens_of_chunk=10_000)
        # Nothing clears the floor, so the whole tree comes back as one chunk.
        assert len(chunks) == 1

    def test_rejects_a_max_descendant_count_below_one(self, tree):
        with pytest.raises(ValueError):
            tree.chunk_tree(max_n_descendants=0, min_n_tokens_of_chunk=1)

    def test_chunk_to_string_renders_id_role_and_name(self, tree):
        rendered = tree.chunk_to_string(tree.get_node_by_id("24"))
        assert rendered.splitlines()[0] == "- ID: 24, Role: navigation, Name: 'Main'"

    def test_chunk_to_string_indents_descendants(self, tree):
        rendered = tree.chunk_to_string(tree.get_node_by_id("24"))
        assert rendered.splitlines()[1].startswith("  - ID: 25")

    def test_chunk_to_string_covers_the_whole_subtree(self, tree):
        assert len(tree.chunk_to_string(tree.root).splitlines()) == 6


class TestTrimTree:
    """`k` sets how many levels above the target the trim happens.

    At that level every node not on the path to the target has its children
    dropped; the node itself and the path to the target always survive.
    """

    DEEP = "\n".join(
        [
            "RootWebArea 'App'",
            "    [1] section 'A'",
            "        [11] link 'A1'",
            "            [111] span 'A1a'",
            "    [2] section 'B'",
            "        [21] link 'B1'",
            "            [211] span 'B1a'",
        ]
    )

    @pytest.fixture
    def deep(self):
        return AxTree(self.DEEP)

    def _ids(self, tree):
        return [n.node_id for n in tree.root.preorder_traversal()]

    def test_returns_the_root(self, deep, capsys):
        assert deep.trim_tree("11", k=0).node_id == "Root"

    def test_shrinks_the_tree(self, deep, capsys):
        before = deep.get_tree_size_and_depth()[0]
        deep.trim_tree("11", k=0)
        assert deep.get_tree_size_and_depth()[0] < before

    def test_keeps_the_full_path_to_the_target(self, deep, capsys):
        deep.trim_tree("11", k=0)
        assert {"Root", "1", "11"} <= set(self._ids(deep))

    def test_keeps_the_targets_own_descendants(self, deep, capsys):
        deep.trim_tree("11", k=0)
        assert "111" in self._ids(deep)

    def test_drops_descendants_of_the_off_path_sibling(self, deep, capsys):
        deep.trim_tree("11", k=0)
        assert "211" not in self._ids(deep)

    def test_the_off_path_node_itself_survives(self, deep, capsys):
        # Only its children are cleared, so the tree shape stays legible.
        deep.trim_tree("11", k=0)
        assert "21" in self._ids(deep)

    def test_larger_k_trims_higher_and_removes_more(self, deep, capsys):
        deep.trim_tree("11", k=1)
        assert self._ids(deep) == ["Root", "1", "11", "111", "2"]

    def test_trimming_at_a_leaf_level_changes_nothing(self, deep, capsys):
        # Nodes level with a leaf target have no children left to drop.
        deep.trim_tree("111", k=0)
        assert deep.get_tree_size_and_depth()[0] == 7

    def test_unknown_target_returns_none(self, deep, capsys):
        assert deep.trim_tree("9999", k=1) is None

    def test_unknown_target_leaves_the_tree_intact(self, deep, capsys):
        before = deep.root.to_string()
        deep.trim_tree("9999", k=1)
        assert deep.root.to_string() == before


class TestTreeSizeAndDepth:
    def test_reports_size_and_depth(self, tree):
        assert tree.get_tree_size_and_depth() == (6, 3)

    def test_a_lone_node_is_size_one_depth_one(self):
        assert AxTree("RootWebArea 'App'").get_tree_size_and_depth() == (1, 1)

    def test_accepts_an_explicit_subtree_root(self, tree):
        assert tree.get_tree_size_and_depth(tree.get_node_by_id("24")) == (3, 2)

    def test_size_matches_the_node_count(self, tree):
        assert tree.get_tree_size_and_depth()[0] == len(tree)
