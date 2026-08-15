"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging

from agentlab.experiments.study import make_study, Study
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.utils import (
    make_env_args_list_from_workarena_curriculum,
)
from browsergym.experiments.benchmark.metadata.utils import task_metadata
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS

from focus_agent.agents.agent_configs import (
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
    EMBEDDING_RETRIEVER_AGENT,
    GENERIC_AGENT_4_1,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B_50,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B_100,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B_400,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B_1000,
    BM25_RETRIEVER_AGENT,
    BM25_RETRIEVER_AGENT_50,
    BM25_RETRIEVER_AGENT_100,
    BM25_RETRIEVER_AGENT_400,
    BM25_RETRIEVER_AGENT_1000,
)


logging.getLogger().setLevel(logging.INFO)

# choose your agent or provide a new agent
baseline_args = [
    GENERIC_AGENT_4_1,
    BM25_RETRIEVER_AGENT,
    EMBEDDING_RETRIEVER_AGENT,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
]

retrievers_args = [
    BM25_RETRIEVER_AGENT_50,
    BM25_RETRIEVER_AGENT_100,
    BM25_RETRIEVER_AGENT, # 200
    BM25_RETRIEVER_AGENT_400,
    BM25_RETRIEVER_AGENT_1000,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B_50,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B_100,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B, # 200
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B_400,
    EMBEDDING_RETRIEVER_AGENT_QWEN_8B_1000,
]

# Create WorkArena L1 tiny
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

agent_args = retrievers_args  # retrievers_args, baseline_args

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

# Number of parallel jobs
n_jobs = 10  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores


if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        # relaunch an existing study
        # study = Study.load("path/to/study")
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
