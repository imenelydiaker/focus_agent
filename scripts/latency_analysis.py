#!/usr/bin/env python3
"""
Latency analysis: FocusAgent vs GenericAgent on WorkArena L1.

Runs all 33 WorkArena-L1 tasks sequentially with 1 seed (seed=0) and
measures wall-clock LLM latency per step:

  GenericAgent:   one agent-LLM call per step
  FocusAgent:     one retriever-LLM call + one agent-LLM call per step

Usage:
    cd focus_agent
    .venv/bin/python scripts/latency_analysis.py

Outputs:
    latency_results/latency_results.json   raw per-step data
    Printed comparison table
"""

import json
import logging
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Optional

import bgym
from browsergym.experiments.loop import EnvArgs

from agentlab.agents.generic_agent.generic_agent import GenericAgent

from focus_agent.agents.agent_configs import (
    GENERIC_AGENT_4_1,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI,
)
from focus_agent.agents.focus_agent import FocusAgent

# ── config ─────────────────────────────────────────────────────────────────

SEED = 0
MAX_STEPS = 15
RESULTS_DIR = Path("latency_results")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

WORKARENA_L1_TASKS = [
    "workarena.servicenow.all-menu",
    "workarena.servicenow.create-change-request",
    "workarena.servicenow.create-hardware-asset",
    "workarena.servicenow.create-incident",
    "workarena.servicenow.create-problem",
    "workarena.servicenow.create-user",
    "workarena.servicenow.filter-asset-list",
    "workarena.servicenow.filter-change-request-list",
    "workarena.servicenow.filter-hardware-list",
    "workarena.servicenow.filter-incident-list",
    "workarena.servicenow.filter-service-catalog-item-list",
    "workarena.servicenow.filter-user-list",
    "workarena.servicenow.impersonation",
    "workarena.servicenow.knowledge-base-search",
    "workarena.servicenow.multi-chart-min-max-retrieval",
    "workarena.servicenow.multi-chart-value-retrieval",
    "workarena.servicenow.order-apple-mac-book-pro15",
    "workarena.servicenow.order-apple-watch",
    "workarena.servicenow.order-developer-laptop",
    "workarena.servicenow.order-development-laptop-p-c",
    "workarena.servicenow.order-ipad-mini",
    "workarena.servicenow.order-ipad-pro",
    "workarena.servicenow.order-loaner-laptop",
    "workarena.servicenow.order-sales-laptop",
    "workarena.servicenow.order-standard-laptop",
    "workarena.servicenow.single-chart-min-max-retrieval",
    "workarena.servicenow.single-chart-value-retrieval",
    "workarena.servicenow.sort-asset-list",
    "workarena.servicenow.sort-change-request-list",
    "workarena.servicenow.sort-hardware-list",
    "workarena.servicenow.sort-incident-list",
    "workarena.servicenow.sort-service-catalog-item-list",
    "workarena.servicenow.sort-user-list",
]

assert len(WORKARENA_L1_TASKS) == 33, "Expected 33 tasks"


# ── timing wrapper ─────────────────────────────────────────────────────────


class TimedLLM:
    """Wraps any LLM callable and accumulates wall-clock time per step."""

    def __init__(self, llm):
        self._llm = llm
        self._step_s = 0.0
        self._n_calls = 0

    def reset(self):
        self._step_s = 0.0
        self._n_calls = 0

    @property
    def step_s(self) -> float:
        """Total LLM time accumulated since last reset() call."""
        return self._step_s

    @property
    def n_calls(self) -> int:
        return self._n_calls

    def __call__(self, messages, **kwargs):
        t0 = time.perf_counter()
        result = self._llm(messages, **kwargs)
        self._step_s += time.perf_counter() - t0
        self._n_calls += 1
        return result

    # Delegate stats and any other attribute to the underlying model.
    def get_stats(self):
        return self._llm.get_stats()

    def __getattr__(self, name):
        return getattr(self._llm, name)


# ── timed agents ───────────────────────────────────────────────────────────


class TimedGenericAgent(GenericAgent):
    """GenericAgent that records per-step agent-LLM latency in extra_info."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t_agent = TimedLLM(self.chat_llm)
        self.chat_llm = self._t_agent

    def get_action(self, obs):
        self._t_agent.reset()
        action, info = super().get_action(obs)
        if info.extra_info is None:
            info.extra_info = {}
        info.extra_info["latency_retriever_s"] = None
        info.extra_info["latency_agent_s"] = self._t_agent.step_s
        info.extra_info["latency_total_s"] = self._t_agent.step_s
        return action, info


class TimedFocusAgent(FocusAgent):
    """FocusAgent that records retriever + agent LLM latency per step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t_retriever = TimedLLM(self.retriever_chat_model)
        self.retriever_chat_model = self._t_retriever
        self._t_agent = TimedLLM(self.chat_llm)
        self.chat_llm = self._t_agent

    def get_action(self, obs):
        self._t_retriever.reset()
        self._t_agent.reset()
        action, info = super().get_action(obs)
        if info.extra_info is None:
            info.extra_info = {}
        info.extra_info["latency_retriever_s"] = self._t_retriever.step_s
        info.extra_info["latency_agent_s"] = self._t_agent.step_s
        info.extra_info["latency_total_s"] = self._t_retriever.step_s + self._t_agent.step_s
        return action, info


# ── experiment runner ──────────────────────────────────────────────────────


def _make_timed_agent(agent_args):
    """Instantiate the correct timed agent from base agent_args."""
    if isinstance(agent_args, type):
        raise ValueError("Pass an instance, not a class.")

    original_make = agent_args.make_agent

    if "FocusAgent" in type(agent_args).__name__:
        def make():
            return TimedFocusAgent(
                chat_model_args=agent_args.chat_model_args,
                flags=agent_args.flags,
                retriever_chat_model_args=agent_args.retriever_chat_model_args,
                retriever_prompt_flags=agent_args.retriever_prompt_flags,
                keep_structure=agent_args.keep_structure,
                strategy=agent_args.strategy,
                max_retry=agent_args.max_retry,
                benchmark=agent_args.benchmark,
                sanitize_attacks=agent_args.sanitize_attacks,
                retriever_type=agent_args.retriever_type,
                extra_instruction=agent_args.extra_instruction,
            )
    else:
        def make():
            return TimedGenericAgent(
                chat_model_args=agent_args.chat_model_args,
                flags=agent_args.flags,
                max_retry=agent_args.max_retry,
            )

    return make


def run_task(
    agent_args,
    task_name: str,
    seed: int = 0,
    max_steps: int = 15,
    tmp_dir: Optional[Path] = None,
) -> list[dict]:
    """Run one task end-to-end and return a list of per-step timing dicts.

    Each dict has keys: step, latency_retriever_s, latency_agent_s, latency_total_s.
    latency_retriever_s is None for GenericAgent.
    Returns an empty list on error.
    """
    tmp_dir = tmp_dir or Path("/tmp/latency_analysis_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env_args = EnvArgs(
        task_name=task_name,
        task_seed=seed,
        max_steps=max_steps,
        headless=True,
        record_video=False,
    )

    make_agent = _make_timed_agent(agent_args)
    step_timings = []

    agent = None
    env = None
    try:
        agent = make_agent()
        env = env_args.make_env(
            action_mapping=agent.action_set.to_python_code,
            exp_dir=tmp_dir,
        )

        obs, _ = env.reset(seed=seed)
        if agent.obs_preprocessor:
            obs = agent.obs_preprocessor(obs)

        terminated = truncated = False
        step = 0

        while not (terminated or truncated) and step < max_steps:
            action, info = agent.get_action(obs.copy())

            extra = info.extra_info or {}
            step_timings.append(
                {
                    "step": step,
                    "latency_retriever_s": extra.get("latency_retriever_s"),
                    "latency_agent_s": extra.get("latency_agent_s"),
                    "latency_total_s": extra.get("latency_total_s"),
                }
            )

            if action is None:
                break

            obs, _reward, terminated, truncated, _ = env.step(action)
            if agent.obs_preprocessor:
                obs = agent.obs_preprocessor(obs)
            step += 1

    except Exception:
        logger.error(f"Error on task {task_name}:\n{traceback.format_exc()}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return step_timings


# ── analysis helpers ───────────────────────────────────────────────────────


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])


def _stats(values: list[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "median": _percentile(clean, 50),
        "p90": _percentile(clean, 90),
        "p95": _percentile(clean, 95),
    }


def print_comparison(results: dict):
    """Print a side-by-side latency comparison table."""

    def collect(agent_key: str, field: str):
        vals = []
        for task_data in results[agent_key]["tasks"].values():
            for step in task_data["steps"]:
                v = step.get(field)
                if v is not None:
                    vals.append(v)
        return vals

    agents = list(results.keys())
    print("\n" + "=" * 80)
    print(f"  LATENCY ANALYSIS  —  WorkArena L1  —  seed={SEED}")
    print("=" * 80)

    header = f"{'Metric':<30}" + "".join(f"{a:>22}" for a in agents)
    print(header)
    print("-" * 80)

    for field, label in [
        ("latency_retriever_s", "Retriever LLM (s)"),
        ("latency_agent_s", "Agent LLM (s)"),
        ("latency_total_s", "Total per step (s)"),
    ]:
        row = f"  {label:<28}"
        for agent_key in agents:
            vals = collect(agent_key, field)
            s = _stats(vals)
            if s["mean"] is None:
                row += f"{'N/A':>22}"
            else:
                row += f"  mean={s['mean']:6.2f}  p95={s['p95']:6.2f}"
        print(row)

    print("-" * 80)

    # Per-task step count
    print(f"\n  {'Task':<44}" + "".join(f"  {'steps':>6}  {'total(s)':>10}" for _ in agents))
    hdr2 = f"  {'':44}" + "".join(f"  {a:>18}" for a in agents)
    print(hdr2)
    print("-" * 80)

    all_tasks = sorted(
        set(t for a in results.values() for t in results[a]["tasks"])
    )
    for task in all_tasks:
        short = task.replace("workarena.servicenow.", "")
        row = f"  {short:<44}"
        for agent_key in agents:
            task_data = results[agent_key]["tasks"].get(task)
            if task_data is None or not task_data["steps"]:
                row += f"  {'ERR':>6}  {'':>10}"
                continue
            steps = task_data["steps"]
            totals = [s["latency_total_s"] for s in steps if s["latency_total_s"] is not None]
            row += f"  {len(steps):>6}  {sum(totals):>9.1f}s"
        print(row)

    print("=" * 80)

    # Overhead summary
    if len(agents) == 2:
        a_generic = next((a for a in agents if "generic" in a.lower()), agents[0])
        a_retriever = next((a for a in agents if "retriever" in a.lower()), agents[1])
        gen_totals = collect(a_generic, "latency_total_s")
        ret_totals = collect(a_retriever, "latency_total_s")
        ret_retriever = collect(a_retriever, "latency_retriever_s")
        if gen_totals and ret_totals:
            gen_mean = sum(gen_totals) / len(gen_totals)
            ret_mean = sum(ret_totals) / len(ret_totals)
            retr_mean = sum(ret_retriever) / len(ret_retriever) if ret_retriever else 0
            overhead = ret_mean - gen_mean
            overhead_pct = 100 * overhead / gen_mean if gen_mean else 0
            print(f"\n  Retriever overhead per step:  {overhead:+.2f}s  ({overhead_pct:+.1f}%)")
            print(f"  Retriever LLM share of total: {100 * retr_mean / ret_mean:.1f}%")
        print()


# ── main ───────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Latency analysis: Focus vs Generic")
    parser.add_argument(
        "--focus-agent-only",
        action="store_true",
        help="Skip GenericAgent and load its results from a previously saved JSON instead.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = RESULTS_DIR / "tmp"
    out_file = RESULTS_DIR / "latency_results.json"

    benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l1"]()

    retriever_args = deepcopy(FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI)
    retriever_args.set_benchmark(benchmark, demo_mode=False)
    retriever_key = (
        f"FocusAgent-{retriever_args.chat_model_args.model_name}"
        f"+{retriever_args.retriever_chat_model_args.model_name}"
    )

    # Load previously saved results (keeps generic results when --focus-agent-only)
    if out_file.exists():
        with open(out_file) as f:
            results: dict = json.load(f)
        logger.info(f"Loaded existing results from {out_file} ({list(results.keys())})")
    else:
        results = {}

    if not args.focus_agent_only:
        generic_args = deepcopy(GENERIC_AGENT_4_1)
        generic_args.set_benchmark(benchmark, demo_mode=False)
        generic_key = f"GenericAgent-{generic_args.chat_model_args.model_name}"
        experiments = [(generic_key, generic_args), (retriever_key, retriever_args)]
    else:
        experiments = [(retriever_key, retriever_args)]

    total_tasks = len(WORKARENA_L1_TASKS)

    for agent_key, agent_args in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Agent: {agent_key}")
        logger.info(f"{'='*60}")
        results[agent_key] = {"agent": agent_key, "tasks": {}}

        for i, task_name in enumerate(WORKARENA_L1_TASKS, 1):
            logger.info(f"  [{i}/{total_tasks}] {task_name}")
            step_timings = run_task(
                agent_args=agent_args,
                task_name=task_name,
                seed=SEED,
                max_steps=MAX_STEPS,
                tmp_dir=tmp_dir,
            )
            results[agent_key]["tasks"][task_name] = {
                "steps": step_timings,
                "n_steps": len(step_timings),
            }
            if step_timings:
                totals = [s["latency_total_s"] for s in step_timings if s["latency_total_s"]]
                mean_total = sum(totals) / len(totals) if totals else 0
                retr_vals = [s["latency_retriever_s"] for s in step_timings if s["latency_retriever_s"]]
                mean_retr = sum(retr_vals) / len(retr_vals) if retr_vals else 0
                logger.info(
                    f"      steps={len(step_timings)}  "
                    f"mean_total={mean_total:.2f}s  "
                    f"mean_retriever={mean_retr:.2f}s"
                )
            else:
                logger.warning(f"      No steps recorded (task may have failed).")

            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"      Saved incrementally to {out_file}")

    # Print comparison table
    print_comparison(results)


if __name__ == "__main__":
    main()
