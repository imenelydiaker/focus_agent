import bgym
from doomarena.browsergym.filters import FilterByUrl
from doomarena.browsergym.success_filters import TargetUrl
from doomarena.browsergym.scripts.run_bgym_experiment import (
    BgymExperiment,
    run_bgym_experiment,
    setup_logging,
)
from doomarena.core.agent_defenses.prompted_gpt import PromptedGpt

from focus_agent.agents.agent_configs import (
    GENERIC_AGENT_4_1,
    DEFENDER_GENERIC_AGENT_4_1,
    AGENT_4_1_DEFENDER_RETRIEVER_4_1_MINI,
    GENERIC_AGENT_QWEN3_235B_A30B,
    AGENT_QWEN3_235B_A30B_DEFENDER_RETRIEVER_4_1_MINI,
    DEFENDER_GENERIC_AGENT_QWEN3_235B_A22B,
)

benchmark = bgym.DEFAULT_BENCHMARKS["webarena_reddit_tiny"]()

defenses = [PromptedGpt(abort_on_detection=True)]

bgym_experiments = [
    BgymExperiment(
        agent=GENERIC_AGENT_4_1,
        attack_configs=None,  # no attack
        benchmark=benchmark,
        # defenses=defenses, # with defense layer
    ),
    BgymExperiment(
        agent=DEFENDER_GENERIC_AGENT_QWEN3_235B_A22B,
        attack_configs=None,  # no attack
        benchmark=benchmark,
        # defenses=defenses, # with defense layer
    ),
    BgymExperiment(
        agent=AGENT_4_1_DEFENDER_RETRIEVER_4_1_MINI,
        attack_configs=None,  # no attack
        benchmark=benchmark,
        # defenses=defenses, # with defense layer
    ),
    BgymExperiment(
        agent=AGENT_QWEN3_235B_A30B_DEFENDER_RETRIEVER_4_1_MINI,
        attack_configs=None,  # no attack
        benchmark=benchmark,
        # defenses=defenses, # with defense layer
    ),
]

if __name__ == "__main__":
    setup_logging()

    run_bgym_experiment(
        bgym_experiments=bgym_experiments,
        reproducibility_mode=False,
        relaunch=False,
        n_jobs=4,
        skip_reset_and_massage=False,
        max_steps=30,  # Reduce max steps for quicker testing
    )
