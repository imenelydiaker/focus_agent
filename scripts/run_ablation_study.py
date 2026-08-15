"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import os
import logging

from agentlab.experiments.study import make_study, Study
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.utils import (
    make_env_args_list_from_repeat_tasks,
    make_env_args_list_from_workarena_curriculum,
)
from browsergym.experiments.benchmark.metadata.utils import task_metadata
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS

from focus_agent.agents.agent_configs import (
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
    GENERIC_AGENT_4_1,
)

from focus_agent.ablation.history_agent_configs import (
    GENERIC_AGENT_4_1_THINK_HISTORY,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_HISTORY,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_NO_HISTORY,
)

from focus_agent.ablation.structure_agent_configs import (
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_STRUCTURE,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_STRUCTURE_BID_ROLE,
)

from focus_agent.ablation.retriever_history_agent_configs import (
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_HISTORY,
)

from focus_agent.ablation.prompts_agent_configs import (
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_RESTRICTIVE,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_NEUTRAL,
)


logging.getLogger().setLevel(logging.INFO)

# Tree Structure Ablation
structure_agent_args = [
    GENERIC_AGENT_4_1,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_STRUCTURE,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_STRUCTURE_BID_ROLE,
]

# History Ablation
history_agent_args = [
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_HISTORY,  # History thoughts + actions
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_NO_HISTORY,  # No history
]

# Retriever History Ablation
retriever_agent_args = [
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_HISTORY,
]

# Prompts Ablation
prompts_agent_args = [
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,  # Regular is extensive
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_RESTRICTIVE,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_NEUTRAL,
]

# Choose the benchmark to run on
benchmark_name = "webarena_reddit"
# benchmark_name = "workarena_l1_tiny"

match benchmark_name:
    case "webarena_reddit":
        from doomarena.browsergym.webarena_subsets import get_webarena_subset

        benchmark = get_webarena_subset(start_url="__REDDIT__")

    case "workarena_l1_tiny":
        benchmark = Benchmark(
            name="workarena_l1_tiny",
            high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["workarena"],
            is_multi_tab=False,
            supports_parallel_seeds=True,
            backends=["workarena"],
            env_args_list=make_env_args_list_from_workarena_curriculum(
                level="l1",
                task_category_filter=None,
                meta_seed=42,  # meta seed for evaluation curriculum
                max_steps=15,
                curriculum_type="agent",
                seeds_l1=10,
            ),
            task_metadata=task_metadata("workarena"),
        )

# Set agents to run
agent_args = structure_agent_args  # history_agent_args, structure_agent_args, retriever_agent_args, prompts_agent_args

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

# Number of parallel jobs
n_jobs = 14  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores


if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = make_study(
            agent_args=agent_args,
            benchmark=benchmark,
            logging_level_stdout=logging.WARNING,
        )

    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        strict_reproducibility=reproducibility_mode,
        n_relaunch=3,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
