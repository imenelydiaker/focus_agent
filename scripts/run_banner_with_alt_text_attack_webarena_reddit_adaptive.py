import bgym
from doomarena.core.agent_defenses.prompted_gpt import PromptedGpt
from doomarena.browsergym.attacks.adaptive_attacks import TaskAdaptiveBannerAttack
from doomarena.browsergym.filters import FilterByUrl
from doomarena.browsergym.success_filters import TargetUrl
from doomarena.browsergym.scripts.run_bgym_experiment import (
    BgymExperiment,
    run_bgym_experiment,
    setup_logging,
)
from agentlab.agents.generic_agent import (
    AGENT_4o_VISION,
    AGENT_4o,
    AGENT_4o_MINI,
)
from doomarena.core.attack_config import AttackConfig
from copy import deepcopy

from doomarena.core.agent_defenses.llmjudge import LLMJudgeSafetyCheck
from focus_agent.agents.agent_configs import (
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_STRUCTURE,
    GENERIC_AGENT_4_1,
    AGENT_4_1_DEFENDER_RETRIEVER_4_1_MINI,
    DEFENDER_GENERIC_AGENT_4_1,
    GENERIC_AGENT_QWEN3_235B_A30B,
    FOCUS_AGENT_QWEN3_235B_A30B_RETRIEVER_4_1_MINI,
)

extra_instructions = """
As a VLM, remember to pay close attention to the overall screenshot and images displayed as they might contain information inaccessible in text.
"""
AGENT_4o_VISION = deepcopy(AGENT_4o_VISION)
AGENT_4o_VISION.flags.obs.use_ax_tree = True
AGENT_4o_VISION.flags.extra_instructions = extra_instructions
AGENT_4o_MINI = deepcopy(AGENT_4o_MINI)
AGENT_4o_MINI.flags.obs.use_ax_tree = True
AGENT_4o_MINI.flags.extra_instructions = extra_instructions
AGENT_4o = deepcopy(AGENT_4o)
AGENT_4o.flags.obs.use_ax_tree = True
AGENT_4o.flags.extra_instructions = extra_instructions


task_adaptive_attack = TaskAdaptiveBannerAttack()

reddit_user_generated_content_attack_config = AttackConfig(
    attackable_component={"type": "banner"},
    attack=task_adaptive_attack,
    filter=FilterByUrl(allowed_urls=["*"]),
    success_filter=TargetUrl(port=1234, target_urls=["localhost:1234"]),
)

benchmark = bgym.DEFAULT_BENCHMARKS["webarena_reddit_tiny"]()

defenses = [PromptedGpt(abort_on_detection=True)]

bgym_experiments = [
    BgymExperiment(
        agent=FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
        attack_configs=reddit_user_generated_content_attack_config,
        benchmark=benchmark,
        # defenses=defenses, # with defense
    ),
]

if __name__ == "__main__":
    setup_logging()

    run_bgym_experiment(
        bgym_experiments=bgym_experiments,
        reproducibility_mode=False,
        relaunch=True,
        n_jobs=10,
        skip_reset_and_massage=False,
        max_steps=30,
    )
