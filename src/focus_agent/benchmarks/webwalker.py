from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from browsergym.core.env import BrowserEnv
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS

from agentlab.benchmarks.abstract_env import AbstractBenchmark, AbstractEnvArgs, AbstractEnv
from focus_agent.benchmarks.webwalker_task import WebWalkerTask, get_dataset


@dataclass
class WebWalkerEnvArgs(AbstractEnvArgs):
    task_name: str
    task_index: int
    task_seed: int = 0
    max_steps: int = 30
    headless: bool = True

    def make_env(self, action_mapping, exp_dir: Path, use_raw_page_output: bool = False, **kwargs):
        return WebWalkerEnv(
            task_entrypoint=WebWalkerTask,
            task_kwargs={"task_id": self.task_index},
            headless=self.headless,
            action_mapping=action_mapping,
            use_raw_page_output=use_raw_page_output,
            max_steps=self.max_steps,
        )


class WebWalkerEnv(AbstractEnv, BrowserEnv):
    """
    Environment for WebWalker tasks.
    This environment inherits from browsergym.core.env.BrowserEnv
    and rewrites step() method to stop the agent when it reaches a max
    number of steps.
    """

    def __init__(
        self,
        task_entrypoint: WebWalkerTask,
        task_kwargs: dict,
        headless: bool,
        action_mapping,
        use_raw_page_output: bool,
        max_steps: int,
        **kwargs,
    ):
        self.max_steps = max_steps
        BrowserEnv.__init__(
            self,
            task_entrypoint=task_entrypoint,
            task_kwargs=task_kwargs,
            headless=headless,
            action_mapping=action_mapping,
            use_raw_page_output=use_raw_page_output,
        )

    def reset(self, seed: int = None):
        self._step_count = 0
        return BrowserEnv.reset(self, seed=seed)

    def step(self, action):
        self._step_count += 1
        obs, reward, terminated, truncated, info = BrowserEnv.step(self, action)
        if self._step_count >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

    def close(self):
        return BrowserEnv.close(self)


class WebWalkerBenchmark(AbstractBenchmark):
    name: str = "webwalkerqa"
    high_level_action_set_args: Any = DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"]
    is_multi_tab: bool = True
    supports_parallel_seeds: bool = False
    backends: list[str] = ["webarena"]
    split: str = "main"
    start: int = 0
    limit: int | None = None
    task_seed: int = 0
    max_steps: int = 30
    headless: bool = True
    dataset_name: str = "callanwu/WebWalkerQA"
    env_args_list: list[WebWalkerEnvArgs] | None = None

    def model_post_init(self, __context: Any) -> None:
        dataset = get_dataset()
        end = None if self.limit is None else self.start + self.limit
        selected_indices = range(
            self.start, len(dataset) if end is None else min(end, len(dataset))
        )

        self.env_args_list = [
            WebWalkerEnvArgs(
                task_name=f"webwalkerqa.{index}",
                task_index=index,
                task_seed=self.task_seed,
                max_steps=self.max_steps,
                headless=self.headless,
            )
            for index in selected_indices
        ]
