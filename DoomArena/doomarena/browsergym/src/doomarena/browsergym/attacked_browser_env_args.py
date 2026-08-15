import json
import logging
import traceback
from pathlib import Path

from traitlets import default
from .attack_gateway import (
    BrowserGymAttackGateway,
)
from doomarena.core.attack_config.get_attack_config import (
    get_attack_config,
)
from agentlab.experiments.loop import EnvArgs
from dataclasses import dataclass, field
from typing import Optional
from browsergym.core.chat import Chat
from browsergym.experiments.loop import (
    save_package_versions,
    _send_chat_info,
    StepInfo,
    _is_debugging,
)
from agentlab.experiments.loop import ExpArgs

try:
    from agentlab.agents.tapeagent import TapeAgent, save_tape
except ImportError:
    TapeAgent = None

from doomarena.core.agent_defenses.base import AttackSafetyCheck

logger = logging.getLogger(__name__)


@dataclass
class AttackedBrowserEnvArgs(EnvArgs):
    benchmark_name: str = ""  # track the benchmark name for easier data aggregation
    attack_configs: Optional[tuple] = ()  # Add this field to support the parameter
    defenses: list[AttackSafetyCheck] = field(default_factory=list)
    abort_on_successful_attack: bool = True

    def make_env(self, action_mapping, exp_dir, exp_task_kwargs: dict = {}, **kwargs):
        """
        Instantiates the BrowserGym environment corresponding to the arguments (with some tweaks).

        Args:
            action_mapping: overrides the action mapping of the environment.
            exp_dir: will set some environment parameters (e.g., record_video_dir) with respect to the directory where the experiment is running.
            exp_task_kwargs: use with caution! Will override task parameters to experiment-specific values. Useful to set different server configs for different experiments, or output file paths within the experiment's folder (e.g., assistantbench).
        """

        env = super().make_env(action_mapping, exp_dir, exp_task_kwargs, **kwargs)
        abort_on_detection = False
        for defense in self.defenses:
            if defense.abort_on_detection:
                abort_on_detection = True
        return BrowserGymAttackGateway(
            env,
            self.attack_configs,
            task_name=self.task_name,
            defenses=self.defenses,
            abort_on_detection=abort_on_detection,
            abort_on_successful_attack=self.abort_on_successful_attack,
        )


class AttackExpArgs(ExpArgs):

    def run(self):
        """Run the experiment while bridging StepInfo.agent_info into env.step()."""

        self._set_logger()
        save_package_versions(self.exp_dir)

        episode_info = []
        env, step_info, err_msg, stack_trace = None, None, None, None
        try:
            logger.info(f"Running experiment {self.exp_name} in:\n  {self.exp_dir}")
            agent = self.agent_args.make_agent()
            logger.debug("Agent created.")

            env = self.env_args.make_env(
                action_mapping=agent.action_set.to_python_code,
                exp_dir=self.exp_dir,
                use_raw_page_output=False,
            )

            logger.debug("Environment created.")

            step_info = StepInfo(step=0)
            episode_info = [step_info]
            step_info.from_reset(
                env, seed=self.env_args.task_seed, obs_preprocessor=agent.obs_preprocessor
            )
            logger.debug("Environment reset.")

            while not step_info.is_done:
                logger.debug(f"Starting step {step_info.step}.")
                action = step_info.from_action(agent)
                logger.debug(f"Agent chose action:\n {action}")

                if action is None:
                    step_info.truncated = True

                step_info.save_step_info(
                    self.exp_dir, save_screenshot=self.save_screenshot, save_som=self.save_som
                )
                logger.debug("Step info saved.")

                _send_chat_info(env.unwrapped.chat, action, step_info.agent_info)
                logger.debug("Chat info sent.")

                if action is None:
                    logger.debug("Agent returned None action. Ending episode.")
                    break

                if hasattr(env, "set_pending_agent_info"):
                    env.set_pending_agent_info(step_info.agent_info)

                step_info = StepInfo(step=step_info.step + 1)
                episode_info.append(step_info)

                logger.debug("Sending action to environment.")
                step_info.from_step(env, action, obs_preprocessor=agent.obs_preprocessor)
                logger.debug("Environment stepped.")

        except Exception as e:
            err_msg = f"Exception uncaught by agent or environment in task {self.env_args.task_name}.\n{type(e).__name__}:\n{e}"
            stack_trace = traceback.format_exc()

            self.err_msg = err_msg
            self.stack_trace = stack_trace

            logger.warning(err_msg + "\n" + stack_trace)
            if _is_debugging() and self.enable_debug:
                logger.warning("Debug mode is enabled. Raising the error.")
                raise

        finally:
            try:
                if step_info is not None:
                    step_info.save_step_info(
                        self.exp_dir, save_screenshot=self.save_screenshot, save_som=self.save_som
                    )
            except Exception as e:
                logger.error(f"Error while saving step info in the finally block: {e}")
            try:
                if (
                    not err_msg
                    and len(episode_info) > 0
                    and not (episode_info[-1].terminated or episode_info[-1].truncated)
                ):
                    e = KeyboardInterrupt("Early termination??")
                    err_msg = f"Exception uncaught by agent or environment in task {self.env_args.task_name}.\n{type(e).__name__}:\n{e}"
                logger.info("Saving summary info.")
                self.save_summary_info(episode_info, self.exp_dir, err_msg, stack_trace)
            except Exception as e:
                logger.error(f"Error while saving summary info in the finally block: {e}")
            try:
                if env is not None:
                    env.close()
            except Exception as e:
                logger.error(f"Error while closing the environment in the finally block: {e}")
            try:
                self._unset_logger()
            except Exception as e:
                logger.error(f"Error while unsetting the logger in the finally block: {e}")

    def save_summary_info(
        self,
        episode_info: list[StepInfo],
        exp_dir,
        err_msg,
        stack_trace,
    ):
        super().save_summary_info(episode_info, exp_dir, err_msg, stack_trace)

        attack_summary_info = {
            "successful_attacks": [],
            "attack_successful": False,
            "successful_attack_contents": [],
            "triggered_defenses": [],
            "attack_undetected": True,
            "success_filter_results": {},
        }
        for i, step_info in enumerate(episode_info):

            if step_info.task_info and "success_filter_results" in step_info.task_info:
                success_filter_results: dict = step_info.task_info["success_filter_results"]
                for key, value in success_filter_results.items():
                    attack_summary_info['success_filter_results'].setdefault(key, False)
                    if value:
                        attack_summary_info['success_filter_results'][key] = True

            if step_info.task_info and "successful_attacks" in step_info.task_info:
                if len(step_info.task_info["successful_attacks"]) > 0:
                    attack_summary_info["attack_successful"] = True

                for attack_name in step_info.task_info["successful_attacks"]:
                    attack_summary_info["successful_attacks"].append((i, attack_name))
            if (
                step_info.task_info
                and "successful_attack_contents" in step_info.task_info
            ):
                attack_summary_info["successful_attack_contents"].append(
                    (i, step_info.task_info["successful_attack_contents"])
                )
            if step_info.task_info and "triggered_defenses" in step_info.task_info:
                if len(step_info.task_info["triggered_defenses"]) > 0:
                    attack_summary_info["attack_undetected"] = False

                attack_summary_info["triggered_defenses"].append(
                    (i, step_info.task_info["triggered_defenses"])
                )

        with open(exp_dir / "attack_summary_info.json", "w") as f:
            json.dump(attack_summary_info, f, indent=4)

        logger.info(
            f"Attack summary info saved to {exp_dir / 'attack_summary_info.json'}"
        )
