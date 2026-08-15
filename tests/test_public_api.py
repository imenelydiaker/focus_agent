"""The package's published import surface.

These names are what users of the PyPI distribution import, so removing or
renaming one is a breaking change. This module pins them.
"""

import pytest

import focus_agent


AGENTS = ["FocusAgent", "EmbeddingRetrieverAgent", "BM25RetrieverAgent"]
ARGS = ["FocusAgentArgs", "EmbeddingRetrieverAgentArgs", "BM25RetrieverAgentArgs"]


class TestPackageRoot:
    @pytest.mark.parametrize("name", AGENTS)
    def test_agent_is_importable_from_the_package_root(self, name):
        assert hasattr(focus_agent, name)

    @pytest.mark.parametrize("name", ARGS)
    def test_args_class_is_importable_from_the_package_root(self, name):
        assert hasattr(focus_agent, name)

    def test_retriever_prompt_flags_are_exported(self):
        # Needed to configure FocusAgent's retriever.
        assert hasattr(focus_agent, "FocusPromptFlags")

    def test_all_is_declared(self):
        assert focus_agent.__all__

    def test_everything_in_all_actually_exists(self):
        missing = [n for n in focus_agent.__all__ if not hasattr(focus_agent, n)]
        assert missing == []

    @pytest.mark.parametrize("name", AGENTS + ARGS)
    def test_public_names_are_listed_in_all(self, name):
        assert name in focus_agent.__all__

    def test_version_is_exposed(self):
        assert isinstance(focus_agent.__version__, str)

    def test_version_matches_the_distribution_metadata(self):
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("focus_agent")
        except PackageNotFoundError:
            pytest.skip("package not installed")
        assert focus_agent.__version__ == installed


class TestAgentsSubpackage:
    @pytest.mark.parametrize("name", AGENTS + ARGS)
    def test_name_is_also_available_on_the_agents_subpackage(self, name):
        from focus_agent import agents

        assert hasattr(agents, name)

    def test_root_and_subpackage_expose_the_same_objects(self):
        from focus_agent import agents

        for name in AGENTS + ARGS:
            assert getattr(focus_agent, name) is getattr(agents, name)

    def test_heuristic_cleaner_is_exported_from_the_subpackage(self):
        from focus_agent import agents

        assert hasattr(agents, "GenericAgentHeuristicCleaner")


class TestAgentsAreUsable:
    """Exported names must be the real classes, not modules or placeholders."""

    @pytest.mark.parametrize("name", AGENTS + ARGS)
    def test_exported_name_is_a_class(self, name):
        assert isinstance(getattr(focus_agent, name), type)

    @pytest.mark.parametrize(
        "agent,args",
        list(zip(AGENTS, ARGS)),
        ids=AGENTS,
    )
    def test_each_agent_pairs_with_its_args_class(self, agent, args):
        args_cls = getattr(focus_agent, args)
        assert hasattr(args_cls, "make_agent")

    def test_agents_derive_from_the_agentlab_base(self):
        from agentlab.agents.generic_agent.generic_agent import GenericAgent

        for name in AGENTS:
            assert issubclass(getattr(focus_agent, name), GenericAgent)
