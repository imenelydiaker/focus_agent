import numpy as np

import bgym
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.metadata.utils import task_metadata
from browsergym.experiments.benchmark.utils import make_env_args_list_from_repeat_tasks
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS

from doomarena.core.attack_config import AttackConfig
from doomarena.browsergym.attacks.adaptive_attacks import (
    TaskGoalDiversionBannerAttack,
    _load_all_webarena_tasks,
    build_diversion_map,
)
from doomarena.core.agent_defenses.prompted_gpt import PromptedGpt
from doomarena.browsergym.filters import StartAtStepFilter, FilterByUrl
from doomarena.browsergym.success_filters import DivertedTaskIntentHeuristic
from doomarena.browsergym.scripts.run_bgym_experiment import (
    BgymExperiment,
    run_bgym_experiment,
    setup_logging,
)
from focus_agent.agents.agent_configs import (
    GENERIC_AGENT_4_1, 
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI, 
    DEFENDER_GENERIC_AGENT_4_1,
    DEFENDER_GENERIC_AGENT_QWEN3_235B_A22B,
    GENERIC_AGENT_QWEN3_235B_A30B, 
    FOCUS_AGENT_QWEN3_235B_A30B_RETRIEVER_4_1_MINI
)

# Build the task->sibling mapping once so both the attack and the success
# filter use the exact same sibling for every task (reproducibility).
shared_map = build_diversion_map(_load_all_webarena_tasks())

# Attack: inject a banner that displays a slightly-different task goal.
goal_diversion_attack_config = AttackConfig(
    attackable_component={"type": "banner"},
    attack=TaskGoalDiversionBannerAttack(diversion_map=shared_map),
    filter=StartAtStepFilter(start_step=0),  # Fire every step (always visible)
    success_filter=DivertedTaskIntentHeuristic(diversion_map=shared_map, type="llm")
)

benchmark = "webarena_reddit_tiny"
# benchmark = "webarena_reddit_tests"

match benchmark:
    case "webarena_reddit_tiny":
        benchmark = bgym.DEFAULT_BENCHMARKS["webarena_reddit_tiny"]()
    case "webarena_reddit_tests":
        benchmark = Benchmark(
            name="webarena_reddit_tests",
            high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
            is_multi_tab=False,
            supports_parallel_seeds=False,
            backends=["webarena"],
            env_args_list=make_env_args_list_from_repeat_tasks(
                task_list=[
                    "webarena.407",
                    "webarena.408",
                    "webarena.648",
                    "webarena.726",
                ],
                max_steps=30,
                n_repeats=1,
                seeds_rng=np.random.RandomState(42),
            ),
            task_metadata=task_metadata("webarena"),
        )

defenses = [PromptedGpt(abort_on_detection=True)]

bgym_experiments = [
    BgymExperiment(
        agent=GENERIC_AGENT_4_1,
        attack_configs=goal_diversion_attack_config,
        benchmark=benchmark,
        defenses=defenses,
    ),
]

if __name__ == "__main__":
    setup_logging()

    run_bgym_experiment(
        bgym_experiments=bgym_experiments,
        reproducibility_mode=False,
        relaunch=False,
        n_jobs=4,
        max_steps=30,
        skip_reset_and_massage=False,
    )
