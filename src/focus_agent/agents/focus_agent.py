from dataclasses import dataclass
from copy import copy
from email.mime import text
import logging
from pydoc import text
import os
from typing import Literal
import ast
import re
from functools import partial

import bgym
from browsergym.experiments import Agent, AgentInfo
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.llm.chat_api import ChatModelArgs
from agentlab.llm.llm_utils import (
    ParseError,
    parse_html_tags_raise,
    retry,
    extract_code_blocks,
)
import agentlab.agents.dynamic_prompting as dp


from focus_agent.retriever.focus_prompt import (
    FocusPrompt,
    FocusDefenderPrompt,
    FocusPromptFlags,
    RestrictiveFocusPrompt,
    NeutralFocusPrompt,
)
from focus_agent.retriever.focus_utils import FocusUtils
from focus_agent.llm_configs import MODEL_CONFIGS_DICT
from focus_agent.agents.utils import get_nb_tokens, add_line_numbers_to_tree
from focus_agent.agents.tree_utils import AxTree


@dataclass
class FocusAgentArgs(GenericAgentArgs):
    flags: GenericPromptFlags = None
    chat_model_args: ChatModelArgs = None
    retriever_chat_model_args: ChatModelArgs = None
    retriever_prompt_flags: FocusPromptFlags = None
    max_retry: int = 4
    keep_structure: bool = False
    strategy: Literal["bid", "role", "bid+role"] = None
    benchmark: str = None
    sanitize_attacks: bool = False
    retriever_type: Literal["line", "defender", "bid_extractor", "restrictive", "neutral"] = "line"
    extra_instruction: str = None

    agent_name: str = None

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            if self.agent_name == None:
                self.agent_name = f"FocusAgent-{self.chat_model_args.model_name}".replace(
                    "/", "_"
                )
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        self.benchmark = benchmark.name
        super().set_benchmark(benchmark, demo_mode)

    def set_reproducibility_mode(self):
        super().set_reproducibility_mode()
        self.retriever_chat_model_args.temperature = 0

    def make_agent(self) -> Agent:
        return FocusAgent(
            self.chat_model_args,
            self.flags,
            retriever_chat_model_args=self.retriever_chat_model_args,
            retriever_prompt_flags=self.retriever_prompt_flags,
            keep_structure=self.keep_structure,
            strategy=self.strategy,
            max_retry=self.max_retry,
            benchmark=self.benchmark,
            sanitize_attacks=self.sanitize_attacks,
            retriever_type=self.retriever_type,
            extra_instruction=self.extra_instruction,
        )


class FocusAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: ChatModelArgs,
        flags: GenericPromptFlags,
        retriever_chat_model_args: ChatModelArgs,
        retriever_prompt_flags: FocusPromptFlags,
        keep_structure: bool = False,
        strategy: str = None,
        max_retry: int = 4,
        benchmark: str = None,
        sanitize_attacks: bool = True,
        retriever_type: str = None,
        extra_instruction: str = None,
    ):
        chat_model_args.temperature = 0  # Set temperature to 0 for deterministic behavior
        super().__init__(chat_model_args, flags, max_retry)

        self.retriever_prompt_flags = retriever_prompt_flags
        self.retriever_chat_model_args = retriever_chat_model_args
        self.extra_instruction = extra_instruction or ""
        self.keep_structure = keep_structure
        self.strategy = strategy
        self.benchmark = benchmark
        self.sanitize_attacks = sanitize_attacks
        self.retriever_type = retriever_type
        self.retriever_chat_model = self.retriever_chat_model_args.make_model()

        self.sanitize_attacks = False if self.retriever_type != "defender" else sanitize_attacks

    @staticmethod
    def clean_list(text: str) -> str:
        """Clean the answer string by removing anything before or after the returned list."""
        match = re.search(r"\[[^\]]*\]", text, re.DOTALL)
        if not match:
            raise ParseError(f"Could not find a list in the model answer: {text}")

        list_str = match.group(0).strip()

        try:
            parsed = ast.literal_eval(list_str)
        except (ValueError, SyntaxError) as exc:
            raise ParseError(f"Could not parse list from model answer: {text}") from exc

        if not isinstance(parsed, list):
            raise ParseError(f"Parsed object is not a list: {parsed}")

        normalized = []
        for item in parsed:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], int)
                and isinstance(item[1], int)
            ):
                normalized.append((item[0], item[1]))
            else:
                raise ParseError(f"Invalid line range tuple in parsed list: {item}")

        return str(normalized)

    @staticmethod
    def _parse_answer(response) -> dict:
        """Parse retriever response: <answer> tags, dict, code block, or bare list formats."""
        if not response:
            raise ParseError(f"Empty or None response from retriever model: {response!r}")

        # 1. <answer> tag format
        if "<answer>" in response:
            try:
                return parse_html_tags_raise(response, keys=["answer"], merge_multiple=True)
            except ParseError:
                raise ParseError(
                    "Could not parse the <answer> tags from your response. "
                    "Ensure the format is exactly <answer>[(start, end), ...]</answer>."
                )

        # 2. Dict format: {'answer': '[(65, 68)]'}
        try:
            match = re.search(r"\{[^{}]*'answer'[^{}]*\}", response, re.DOTALL)
            candidate = match.group(0) if match else response.strip()
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
        except (ValueError, SyntaxError):
            pass

        # 3. Code block format: ```json\n[(65, 68)]\n``` or ```\n[(65, 68)]\n```
        code_match = re.search(r"```[a-zA-Z]*\s*([\s\S]*?)\s*```", response)
        if code_match:
            return {"answer": code_match.group(1).strip()}

        # 4. Bare list anywhere in the response: [(65, 68)]
        list_match = re.search(r"\[[^\]]*\]", response)
        if list_match:
            return {"answer": list_match.group(0).strip()}

        raise ParseError(
            "Your response is missing <answer> tags. "
            "Please wrap your answer in <answer>[(start, end), ...]</answer> tags."
        )

    @staticmethod
    def _parse_answer_with_validation(response) -> dict:
        """Parse and validate the full retriever response, with descriptive retry messages."""
        answer_dict = FocusAgent._parse_answer(response)
        try:
            FocusAgent.clean_list(answer_dict["answer"])
        except ParseError:
            raise ParseError(
                "The list inside your <answer> tags is malformed. "
                "Please provide a valid Python list of integer tuple pairs, "
                "e.g. <answer>[(1, 5), (10, 15)]</answer>."
            )
        return answer_dict

    @staticmethod
    def sanitize_attack_lines(tree: str, attack_line_ranges: list[tuple[int, int]]) -> list[str]:
        """Sanitize attack lines by keeping only bid and role.

        Args:
            tree: The tree text with line numbers
            attack_line_ranges: List of (start, end) tuples indicating line ranges containing attacks

        Returns:
            List of sanitized lines with only bid and role preserved
        """
        lines = tree.splitlines()
        sanitized_lines = []

        # Get all line numbers that are in attack ranges
        attack_line_numbers = set()
        for start, end in attack_line_ranges:
            attack_line_numbers.update(range(start, end + 1))

        for line_num in sorted(attack_line_numbers):
            if line_num <= len(lines):
                line = lines[line_num - 1]  # Convert to 0-indexed
                indentation = line[: len(line) - len(line.lstrip())]

                # Extract bid and role from the line
                parts = line.split()
                if len(parts) >= 2 and "[" in parts[0] and "]" in parts[0]:
                    # Line has a bid like [123]
                    bid = parts[0]
                    role = parts[1]
                    sanitized_line = f"{indentation}{bid} {role} ... sanitized ..."
                elif len(parts) >= 1:
                    # Line doesn't have a bid, just keep the role
                    role = parts[0]
                    sanitized_line = f"{indentation}{role} ... sanitized ..."
                else:
                    # Empty line or whitespace only
                    sanitized_line = f"{indentation}... sanitized ..."

                sanitized_lines.append(sanitized_line)

        return sanitized_lines

    def _make_retriever_prompt(self, obs: dict):
        """Create the appropriate retriever prompt for a given tree chunk."""
        extra_instruction = self.extra_instruction if "webarena" in self.benchmark else ""
        numbered_tree = add_line_numbers_to_tree(obs["axtree_txt"])
        match self.retriever_type:
            case "line":
                return FocusPrompt(
                    goal=obs["goal"],
                    tree=numbered_tree,
                    screenshot=obs["screenshot"],
                    history=obs["history"],
                    extra_instruction=extra_instruction,
                    flags=self.retriever_prompt_flags,
                )
            case "defender":
                return FocusDefenderPrompt(
                    goal=obs["goal"],
                    tree=numbered_tree,
                    screenshot=obs["screenshot"],
                    history=obs["history"],
                    extra_instruction=extra_instruction,
                    flags=self.retriever_prompt_flags,
                )
            case "bid_extractor":
                raise NotImplementedError(
                    "bid extractor is not implemented yet. Please use 'default' or 'defender'."
                )
            case "restrictive":
                return RestrictiveFocusPrompt(
                    goal=obs["goal"],
                    tree=numbered_tree,
                    screenshot=obs["screenshot"],
                    history=obs["history"],
                    extra_instruction=extra_instruction,
                    flags=self.retriever_prompt_flags,
                )
            case "neutral":
                return NeutralFocusPrompt(
                    goal=obs["goal"],
                    tree=numbered_tree,
                    screenshot=obs["screenshot"],
                    history=obs["history"],
                    extra_instruction=extra_instruction,
                    flags=self.retriever_prompt_flags,
                )
            case _:
                raise ValueError(f"Unknown retriever type: {self.retriever_type}")

    def _run_retriever(self, prompt) -> dict:
        """Run the retriever LLM on a prompt and return the parsed answer dict."""
        if self.sanitize_attacks:
            return retry(
                self.retriever_chat_model,
                prompt,
                n_retry=3,
                parser=partial(
                    parse_html_tags_raise,
                    keys=["think", "answer", "attack_lines"],
                    merge_multiple=True,
                ),
            )
        else:
            return retry(
                self.retriever_chat_model,
                prompt,
                n_retry=3,
                parser=self._parse_answer_with_validation,
            )
            # return retry(
            #     self.retriever_chat_model,
            #     prompt,
            #     n_retry=3,
            #     parser=partial(
            #         parse_html_tags_raise,
            #         keys=["answer"], #"think"
            #         merge_multiple=True,
            #     ),
            # )
            

    def get_new_obs(self, obs: dict) -> dict:
        extra_info = {}

        if self.retriever_type == "defender":
            if self.sanitize_attacks:
                logging.info("Overriding sanitize_attacks to True for defender retriever prompt.")
                self.retriever_prompt_flags.sanitize_attacks = True

        self.focus_prompt = self._make_retriever_prompt(obs)

        answer_dict = self._run_retriever(self.focus_prompt.prompt)

        extra_info["retriever_answer"] = answer_dict

        answer = self.clean_list(answer_dict["answer"])

        line_ranges = ast.literal_eval(answer)
        if len(line_ranges) <= 0:
            return obs["axtree_txt"], extra_info

        line_numbers = []
        for line_range in line_ranges:
            if isinstance(line_range, tuple):
                start, end = line_range
                line_numbers.extend(range(start, end + 1))

        # Process pruned tree
        if self.keep_structure:
            pruned_tree = FocusUtils.remove_lines_keep_structure(
                tree=obs["axtree_txt"],
                line_numbers=line_numbers,
                strategy=self.strategy,
            )
        else:
            pruned_tree = FocusUtils.remove_lines(
                tree=obs["axtree_txt"],
                line_numbers=line_numbers,
            )

        # If sanitize_attacks is enabled and attack_lines are provided, sanitize them and add to pruned tree
        if self.sanitize_attacks and "attack_lines" in answer_dict:
            try:
                attack_answer = self.clean_list(answer_dict["attack_lines"])
                attack_line_ranges = ast.literal_eval(attack_answer)

                if len(attack_line_ranges) > 0:
                    # Get sanitized attack lines
                    sanitized_lines = self.sanitize_attack_lines(
                        tree=add_line_numbers_to_tree(obs["axtree_txt"]),
                        attack_line_ranges=attack_line_ranges,
                    )

                    # Add sanitized lines to the pruned tree
                    if sanitized_lines:
                        pruned_tree += "\n\n# Sanitized Attack Elements:\n" + "\n".join(
                            sanitized_lines
                        )
                        extra_info["sanitized_attack_lines"] = sanitized_lines
            except Exception as e:
                logging.warning(f"Failed to process attack_lines: {e}")

        return pruned_tree, extra_info

    def get_action(self, obs):
        obs_history_copy = copy(self.obs_history)
        obs_history_copy.append(obs)
        history = dp.History(
            history_obs=obs_history_copy,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            flags=self.flags.obs,
        )
        obs["history"] = history.prompt
        new_obs, extra_info = self.get_new_obs(obs)
        if get_nb_tokens(new_obs) < get_nb_tokens(obs["axtree_txt"]):
            obs["axtree_txt"] = new_obs

        action, info = super().get_action(obs)
        info.extra_info.update(extra_info)
        info.extra_info["retriever_prompt"] = self.focus_prompt.prompt
        info.extra_info["chunked_tree"] = obs["axtree_txt"]
        info.html_page = self.format_html_page(
            agent_info=info,
            obs=obs,
        )
        return action, info

    def format_html_page(self, agent_info: AgentInfo, obs: dict) -> str:
        html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agent Info</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1 {{
                color: #333;
            }}
            h2 {{
                color: #555;
            }}
            pre {{
                background-color: #333; /* Dark grey background */
                color: #f4f4f4; /* Light grey text */
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow-x: auto;
            }}
            code {{
                font-family: monospace;
            }}
            .image-container {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            details {{
                margin-bottom: 10px;
            }}
            details pre {{
                max-height: 1200px; /* Set a max height for scrollability */
                overflow-y: auto; /* Enable vertical scrolling */
            }}
        </style>
    </head>
    <body>
       <div class="image-container">
            <figure>
                <img src="screenshot_pre_action_placeholder" alt="Pre-action Screenshot">
                <figcaption>Pre-action Screenshot</figcaption>
            </figure>
            <figure>
                <img src="screenshot_post_action_placeholder" alt="Post-action Screenshot">
                <figcaption>Post-action Screenshot</figcaption>
            </figure>
        </div>
        <h1>Agent Info</h1>
    """
        sections = {}
        focus_agent_prompt = agent_info.get("retriever_prompt", "")
        focus_agent_prompt_text = (
            focus_agent_prompt[1].content if focus_agent_prompt else ""
        )
        sections["Focus Agent"] = {
            "Prompt": focus_agent_prompt_text,
        }
        sections["Focus Answer"] = {
            "Think": agent_info.extra_info.get("retriever_think", ""),
            "Answer": agent_info.extra_info.get("retriever_answer", ""),
        }
        sections["Chunked AxTree"] = {
            "Chunked Tree": agent_info.get("chunked_tree", ""),
        }
        for section_title, subsections in sections.items():
            html_template += f"""
            <h2>{section_title}</h2>
            """
            for subsection_title, content in subsections.items():
                if not content:
                    continue
                # wrap the prompt is a collapsible (default collapsed) and scrollable div
                if subsection_title in {"Prompt", "AxTree"}:
                    html_template += f"""
                    <h3>{subsection_title}</h3>
                    <details>
                        <summary>Expand Content</summary>
                        <pre><code>{content}</code></pre>
                    </details>
                    """
                else:
                    html_template += f"""
                    <h3>{subsection_title}</h3>
                    <pre><code>{content} </code></pre>
                    """
        html_template += """
        </body>
        </html>
        """
        return html_template
