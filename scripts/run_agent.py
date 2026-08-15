"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import os
import logging
from typing import List, Literal, Optional

import requests
import numpy as np
import random

import bgym
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.utils import make_env_args_list_from_fixed_seeds
from browsergym.experiments.benchmark.metadata.utils import task_metadata

from agentlab.experiments.study import make_study, Study
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.utils import make_env_args_list_from_repeat_tasks
from browsergym.experiments.benchmark.metadata.utils import task_metadata
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS

from focus_agent.agents.agent_configs import (
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B,
    FOCUS_AGENT_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_STRUCTURE,
    FOCUS_AGENT_4_1_RETRIEVER_4_1,
    GENERIC_AGENT_4_1,
    GENERIC_AGENT_4_1_MINI,
    AGENT_4_1_DEFENDER_RETRIEVER_4_1_MINI,
    AGENT_4_1_DEFENDER_RETRIEVER_4_1,
    AGENT_4_1_DEFENDER_RETRIEVER_4_1_WITH_STRUCTURE,
    EMBEDDING_RETRIEVER_AGENT,
    EMBEDDING_RETRIEVER_AGENT_LARGE,
    BM25_RETRIEVER_AGENT,
    GENERIC_AGENT_4_1_5K,
    GENERIC_AGENT_QWEN3_235B_A30B,
    GENERIC_AGENT_QWEN2_5_72B,
    GENERIC_AGENT_QWEN3_MAX,
    FOCUS_AGENT_QWEN2_5_72B_RETRIEVER_QWEN3_235B_A22B,
    FOCUS_AGENT_QWEN3_235B_A30B_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_QWEN3_235B_A30B_RETRIEVER_5_MINI   ,
    FOCUS_AGENT_4_1_RETRIEVER_5_MINI,
    DEFENDER_GENERIC_AGENT_4_1,
    AGENT_4_1_SATNITIZE_DEFENDER_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_8B,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_4B_THINKING,
    FOCUS_AGENT_QWEN3_235B_A30B_QWEN3_4B_THINKING,
    FOCUS_AGENT_QWEN3_235B_A30B_QWEN3_8B,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_32B,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_235B_A22B,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_5_9B,
    FOCUS_AGENT_QWEN3_235B_A30B_RETRIEVER_QWEN3_5_9B,

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

# choose your agent or provide a new agent
# agent_args = [
#     GENERIC_AGENT_4_1,
#     BM25_RETRIEVER_AGENT
#     EMBEDDING_RETRIEVAL_AGENT,
#     FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
# ]

agent_args = [
    GENERIC_AGENT_QWEN3_235B_A30B,
    FOCUS_AGENT_QWEN3_235B_A30B_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_QWEN3_235B_A30B_RETRIEVER_5_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_8B,
    FOCUS_AGENT_QWEN3_235B_A30B_QWEN3_8B,
    FOCUS_AGENT_QWEN2_5_72B_RETRIEVER_QWEN3_235B_A22B,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_32B,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_5_9B,
    FOCUS_AGENT_QWEN3_235B_A30B_RETRIEVER_QWEN3_5_9B,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_235B_A22B,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_4B_THINKING,
    FOCUS_AGENT_QWEN3_235B_A30B_QWEN3_4B_THINKING,
    GENERIC_AGENT_4_1_MINI,
    GENERIC_AGENT_4_1,
    GENERIC_AGENT_4_1_5K,
    BM25_RETRIEVER_AGENT,
    EMBEDDING_RETRIEVER_AGENT,
    EMBEDDING_RETRIEVER_AGENT_LARGE,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B,
    FOCUS_AGENT_4_1_RETRIEVER_QWEN3_5_27B,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_5_MINI,
]


# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
benchmark = "workarena_l1"
# benchmark = "webarena"
# benchmark = "webarena_tiny"
# benchmark = "workarena_l1_tiny"
# benchmark = "webarena_reddit"
# benchmark = "webarena_shopping_subset5"
# benchmark = "webarena_reddit_tests"

if isinstance(benchmark, str) and benchmark == "webarena_reddit_tests":
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
                "webarena.409",
            ],
            max_steps=10,
            n_repeats=1,
            seeds_rng=np.random.RandomState(42),
        ),
        task_metadata=task_metadata("webarena"),
    )

# Create WorkArena L1 tiny
if isinstance(benchmark, str) and benchmark == "workarena_l1_tiny":
    benchmark = Benchmark(
        name="workarena_l1_tiny",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["workarena"],
        is_multi_tab=False,
        supports_parallel_seeds=False,
        backends=["workarena"],
        env_args_list=make_env_args_list_from_repeat_tasks(
            task_list=[
                "workarena.servicenow.all-menu",
                "workarena.servicenow.create-problem",
                "workarena.servicenow.create-user",
                "workarena.servicenow.create-hardware-asset",
                "workarena.servicenow.order-development-laptop-p-c",
                "workarena.servicenow.order-developer-laptop",
                "workarena.servicenow.order-ipad-mini",
                "workarena.servicenow.order-loaner-laptop",
                "workarena.servicenow.multi-chart-min-max-retrieval",
                "workarena.servicenow.filter-asset-list",
                "workarena.servicenow.filter-change-request-list",
                "workarena.servicenow.sort-asset-list",
                "workarena.servicenow.sort-user-list",
                "workarena.servicenow.knowledge-base-search",
            ],
            max_steps=15,
            n_repeats=2,
            seeds_rng=np.random.RandomState(42),
        ),
        task_metadata=task_metadata("workarena"),
    )


if isinstance(benchmark, str) and benchmark == "webarena_reddit":
    # from doomarena.browsergym.webarena_subsets import get_webarena_subset
    def get_webarena_subset(
        max_tasks: int = 9999999,
        name: str = "webarena_reddit_tiny",
        shuffle: Optional[int] = None,
        start_url: Literal["__REDDIT__", "__SHOPPING__", ""] = "__REDDIT__",
        max_steps: int = 30,
        fixed_seeds: List[int] = [0],
    ) -> Benchmark:
        """
        Creates a configurable subset of the WebArena benchmark.

        Args:
            max_tasks: Maximum number of tasks to include
            name: Name for the benchmark
            shuffle: Random seed for shuffling tasks, or None for no shuffling
            start_url: Filter to include only tasks with this string in URL
            max_steps: Maximum number of steps per episode
            fixed_seeds: List of seeds for environment initialization

        Returns:
            Benchmark: A configured WebArena benchmark
        """
        # URL to the JSON file
        json_url = (
            "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json"
        )

        # Fetch the JSON data from the URL
        response = requests.get(json_url)
        data = response.json()

        # List to store filtered task_ids
        filtered_task_ids = []

        # Loop through each element in the list
        for item in data:
            # Check if start_url contains the filter string
            if start_url in item.get("start_url", "").upper():
                # Add the task_id to the list
                filtered_task_ids.append(item["task_id"])

        # Create a list of tasks based on the output
        task_list = [f"webarena.{task_id}" for task_id in filtered_task_ids]

        # Shuffle the task list if requested
        if shuffle is not None:
            rnd = random.Random(shuffle)
            rnd.shuffle(task_list)
            print(f"Shuffled task list with seed {shuffle}")

        # Apply max_tasks limit
        task_list = task_list[:max_tasks]
        filter_name = start_url if start_url else "all"

        print(
            f"Benchmark '{name}': Selected {len(task_list)} tasks from {len(filtered_task_ids)} "
            f"{filter_name}-filtered tasks (max available: {len(data)})"
        )

        # Create the Benchmark configuration
        benchmark = Benchmark(
            name=name,
            high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
            is_multi_tab=True,
            supports_parallel_seeds=True,
            backends=["webarena"],
            env_args_list=make_env_args_list_from_fixed_seeds(
                task_list=task_list,
                max_steps=max_steps,
                fixed_seeds=fixed_seeds,
            ),
            task_metadata=task_metadata("webarena"),
        )

        return benchmark

    benchmark = get_webarena_subset(name="webarena_reddit_tiny", start_url="__REDDIT__")

if isinstance(benchmark, str) and benchmark == "webarena_shopping_subset5":
    from doomarena.browsergym.webarena_subsets import get_webarena_subset

    benchmark = get_webarena_subset(
        max_tasks=5,
        name="webarena_shopping_subset5",
        start_url="__SHOPPING__",
    )


if benchmark == "webarena":
    from browsergym.experiments.benchmark.utils import make_env_args_list_from_fixed_seeds
    from browsergym.experiments.benchmark.metadata.utils import (
        task_metadata,
        task_list_from_metadata,
    )

    split = "test"
    tm = task_metadata("webarena")
    b = Benchmark(
        name=f"webarena_{split}",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["webarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=task_list_from_metadata(tm),
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=tm,
    )
    benchmark = b.subset_from_split(split)


# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

# Number of parallel jobs
n_jobs = 5 # Make sure to use 1 job when debugging in VSCode
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

    # study.avg_step_timeout = 60 * 3
    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        strict_reproducibility=reproducibility_mode,
        n_relaunch=3,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
