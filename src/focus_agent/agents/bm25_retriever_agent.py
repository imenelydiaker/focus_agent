from dataclasses import dataclass
from copy import copy

from browsergym.experiments import Agent
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.llm.chat_api import ChatModelArgs

import agentlab.agents.dynamic_prompting as dp

from focus_agent.retriever.bm25_retriever import BM25SRetriever, BM25RetrieverArgs
from .utils import add_line_numbers_to_tree


@dataclass
class BM25RetrieverAgentFlags:
    use_history: bool = False


@dataclass
class BM25RetrieverAgentArgs(GenericAgentArgs):
    flags: GenericPromptFlags = None
    chat_model_args: ChatModelArgs = None
    retriever_args: BM25RetrieverArgs = None
    retriever_flags: BM25RetrieverAgentFlags = None
    max_retry: int = 4
    agent_name: str = None

    def __post_init__(self):
        if self.agent_name is None:
            self.agent_name = f"BM25RetrieverAgent-{self.chat_model_args.model_name}".replace(
                "/", "_"
            )

    def make_agent(self) -> Agent:
        return BM25RetrieverAgent(
            self.chat_model_args,
            self.flags,
            self.retriever_args,
            self.retriever_flags,
            self.max_retry,
        )


class BM25RetrieverAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: ChatModelArgs,
        flags,
        retriever_args: BM25RetrieverArgs,
        retriever_flags: BM25RetrieverAgentFlags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        self.retriever_args = retriever_args
        self.retriever_flags = retriever_flags

    def get_new_obs(self, obs: dict) -> str:
        query = (
            obs["goal"] + "\n" + obs["history"] if self.retriever_flags.use_history else obs["goal"]
        )
        axtree_txt: str = obs["axtree_txt"]
        # Initialize BM25 retriever with the current AXTree
        retriever = BM25SRetriever(
            axtree_txt,
            bm25_args=self.retriever_args,
        )
        # Retrieve the most relevant chunks
        relevant_chunks = retriever.retrieve(query)
        new_tree = ""
        for i, chunk in enumerate(relevant_chunks):
            new_tree += f"\n\nChunk {i}:\n{chunk}"
        return new_tree

    def get_action(self, obs: dict):
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
        obs["axtree_txt"] = self.get_new_obs(obs)
        action, info = super().get_action(obs)
        info.extra_info["chunked_tree"] = obs["axtree_txt"]
        info.extra_info["retriever_query"] = obs["goal"] + "\n" + obs["history"]
        return action, info
