"""Validation of the shipped agent configurations.

These are plain declarations, but a typo in a model key or a missing retriever
only surfaces at launch time — after a benchmark has already been scheduled.
Constructing them here is cheap and catches that early. No model is ever
instantiated, so no credentials are required.
"""

import pytest

from focus_agent.agents import agent_configs as ac
from focus_agent.agents.bm25_retriever_agent import BM25RetrieverAgentArgs
from focus_agent.agents.embedding_retriever_agent import EmbeddingRetrieverAgentArgs
from focus_agent.agents.focus_agent import FocusAgentArgs
from focus_agent.ablation import (
    history_agent_configs,
    prompts_agent_configs,
    retriever_history_agent_configs,
    structure_agent_configs,
)
from focus_agent.llm_configs import MODEL_CONFIGS_DICT


def configs_of(module, kind):
    return [
        (name, obj)
        for name, obj in vars(module).items()
        if name.isupper() and isinstance(obj, kind)
    ]


FOCUS_CONFIGS = configs_of(ac, FocusAgentArgs)
ABLATION_MODULES = [
    history_agent_configs,
    prompts_agent_configs,
    retriever_history_agent_configs,
    structure_agent_configs,
]
ABLATION_CONFIGS = [c for m in ABLATION_MODULES for c in configs_of(m, FocusAgentArgs)]
ALL_FOCUS = FOCUS_CONFIGS + ABLATION_CONFIGS


class TestConfigsExist:
    def test_focus_configs_are_declared(self):
        assert len(FOCUS_CONFIGS) > 10

    def test_ablation_configs_are_declared(self):
        assert len(ABLATION_CONFIGS) >= 6

    def test_embedding_and_bm25_baselines_are_declared(self):
        assert configs_of(ac, EmbeddingRetrieverAgentArgs)
        assert configs_of(ac, BM25RetrieverAgentArgs)


@pytest.mark.parametrize("name,config", ALL_FOCUS, ids=[n for n, _ in ALL_FOCUS])
class TestEveryFocusConfig:
    def test_has_an_acting_model(self, name, config):
        assert config.chat_model_args is not None

    def test_has_a_retriever_model(self, name, config):
        # Without this the agent cannot run its first stage.
        assert config.retriever_chat_model_args is not None

    def test_has_retriever_prompt_flags(self, name, config):
        assert config.retriever_prompt_flags is not None

    def test_has_agent_prompt_flags(self, name, config):
        assert config.flags is not None

    def test_has_a_name(self, name, config):
        assert config.agent_name

    def test_uses_a_known_retriever_type(self, name, config):
        assert config.retriever_type in {"line", "defender", "restrictive", "neutral"}

    def test_uses_a_known_structure_strategy(self, name, config):
        assert config.strategy in {None, "bid", "role", "bid+role"}

    def test_models_come_from_the_model_registry(self, name, config):
        # Model args are unhashable dataclasses, so compare against a list.
        known = list(MODEL_CONFIGS_DICT.values())
        assert config.chat_model_args in known
        assert config.retriever_chat_model_args in known


class TestConfigNaming:
    @pytest.mark.parametrize(
        "module", [ac] + ABLATION_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1]
    )
    def test_agent_names_are_unique_within_a_module(self, module):
        # Two configs sharing a name would write to the same result directory.
        names = [c.agent_name for _, c in configs_of(module, FocusAgentArgs)]
        assert len(names) == len(set(names))

    def test_agent_names_are_non_empty(self):
        assert all(c.agent_name.strip() for _, c in ALL_FOCUS)

    def test_agent_names_have_no_path_separators(self):
        # agent_name becomes a result directory name.
        assert all("/" not in c.agent_name for _, c in ALL_FOCUS)

    def test_agent_names_have_no_whitespace(self):
        assert all(" " not in c.agent_name for _, c in ALL_FOCUS)


class TestAblationAxes:
    def test_structure_ablation_enables_keep_structure(self):
        cfg = structure_agent_configs.FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_STRUCTURE
        assert cfg.keep_structure is True

    def test_structure_ablation_sets_a_strategy(self):
        cfg = structure_agent_configs.FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_STRUCTURE
        assert cfg.strategy is not None

    def test_history_ablation_contrasts_the_two_settings(self):
        # The axis is whether the *retriever* sees the interaction history; both
        # configs deliberately share the same acting-agent flags.
        no_history = history_agent_configs.FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_NO_HISTORY
        with_history = history_agent_configs.FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_HISTORY
        assert no_history.retriever_prompt_flags.use_history is False
        assert with_history.retriever_prompt_flags.use_history is True

    def test_history_ablation_holds_the_acting_agent_fixed(self):
        no_history = history_agent_configs.FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_NO_HISTORY
        with_history = history_agent_configs.FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_HISTORY
        assert no_history.chat_model_args is with_history.chat_model_args
        assert no_history.flags is with_history.flags

    def test_retriever_history_ablation_toggles_the_retriever_flag(self):
        cfg = retriever_history_agent_configs.FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_HISTORY
        assert cfg.retriever_prompt_flags.use_history is True

    def test_prompt_ablations_select_the_wording_variants(self):
        restrictive = prompts_agent_configs.FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_RESTRICTIVE
        neutral = prompts_agent_configs.FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_NEUTRAL
        assert restrictive.retriever_type == "restrictive"
        assert neutral.retriever_type == "neutral"


class TestBaselineConfigs:
    def test_bm25_chunk_size_suffixes_match_their_configuration(self):
        for suffix in (50, 400, 1000):
            cfg = getattr(ac, f"BM25_RETRIEVER_AGENT_{suffix}")
            assert cfg.retriever_args.chunk_size == suffix

    def test_bm25_configs_declare_a_top_k(self):
        for _, cfg in configs_of(ac, BM25RetrieverAgentArgs):
            assert cfg.retriever_args.top_k > 0

    def test_embedding_configs_declare_retriever_args(self):
        for _, cfg in configs_of(ac, EmbeddingRetrieverAgentArgs):
            assert cfg.retriever_args is not None

    def test_defender_configs_use_the_defender_retriever(self):
        defenders = [(n, c) for n, c in FOCUS_CONFIGS if "DEFENDER" in n]
        assert defenders
        assert all(c.retriever_type == "defender" for _, c in defenders)
