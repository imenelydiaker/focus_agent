from copy import deepcopy

import numpy as np

import bgym
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.metadata.utils import task_metadata
from browsergym.experiments.benchmark.utils import make_env_args_list_from_repeat_tasks
from browsergym.experiments.benchmark.metadata.utils import task_metadata
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS

from doomarena.core.attack_config import AttackConfig
from doomarena.browsergym.attacks.adaptive_attacks import (
    TaskGoalDiversionPopupAttack,
    _load_all_webarena_tasks,
    build_diversion_map,
)
from doomarena.browsergym.filters import StartAtStepFilter, EveryNStepsFilter
from doomarena.browsergym.success_filters import DivertedTaskSuccessFilter
from doomarena.browsergym.scripts.run_bgym_experiment import (
    BgymExperiment,
    run_bgym_experiment,
    setup_logging,
)
from focus_agent.agents.agent_configs import (
    GENERIC_AGENT_4_1,
    GENERIC_AGENT_QWEN3_235B_A30B,
    DEFENDER_GENERIC_AGENT_4_1,
    AGENT_4_1_DEFENDER_RETRIEVER_4_1_MINI,
)

# Build the task->sibling mapping once so both the attack and the success
# filter use the exact same sibling for every task (reproducibility).
_diversion_map = build_diversion_map(_load_all_webarena_tasks())

# Attack: inject a popup that presents a slightly-different task goal.
goal_diversion_attack_config = AttackConfig(
    attackable_component={"type": "popup-element"},
    attack=TaskGoalDiversionPopupAttack(diversion_map=_diversion_map),
    filter=EveryNStepsFilter(n=3),  # Fire every 3 steps
    success_filter=DivertedTaskSuccessFilter(diversion_map=_diversion_map),
)

benchmark = "webarena_reddit_tiny"
# benchmark = "webarena_reddit_tests" # small subset for testing

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

bgym_experiments = [
    BgymExperiment(
        agent=GENERIC_AGENT_4_1,
        attack_configs=goal_diversion_attack_config,
        benchmark=benchmark,
    ),
    BgymExperiment(
        agent=GENERIC_AGENT_QWEN3_235B_A30B,
        attack_configs=goal_diversion_attack_config,
        benchmark=benchmark,
    ),
    BgymExperiment(
        agent=DEFENDER_GENERIC_AGENT_4_1,
        attack_configs=goal_diversion_attack_config,
        benchmark=benchmark,
    ),
    BgymExperiment(
        agent=AGENT_4_1_DEFENDER_RETRIEVER_4_1_MINI,
        attack_configs=goal_diversion_attack_config,
        benchmark=benchmark,
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
