# run_webwalker_study.py
from agentlab.agents.generic_agent import GenericAgentArgs
from agentlab.experiments.study import Study, make_study

from focus_agent.llm_configs import MODEL_CONFIGS_DICT
from focus_agent.agents.agent_configs import FLAGS_GPT_4o

from focus_agent.benchmarks.webwalker import WebWalkerBenchmark
from focus_agent.agents.agent_configs import (
    GENERIC_AGENT_4_1,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_5_MINI,
)

# Pick your agent
agents = [
    GENERIC_AGENT_4_1,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_5_MINI,
]

benchmark = WebWalkerBenchmark(
    split="main",
    start=0,
    limit=None,
    # start=43,
    # limit=1,
    task_seed=42,
    max_steps=30,
    headless=True,
)

relaunch = False  # set to True to relaunch an existing study instead of starting a new one

if relaunch:
    #  relaunch an existing study
    # study = Study.load("path/to/study")
    study = Study.load_most_recent(contains=None)
    study.find_incomplete(include_errors=True)
else:
    # Launch the study
    study = make_study(
        benchmark=benchmark,
        agent_args=agents,
        comment="WebWalkerQA eval run",
    )

study.run(n_jobs=5)  # parallel with joblib/ray
