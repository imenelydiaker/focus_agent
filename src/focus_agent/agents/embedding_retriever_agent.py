from dataclasses import dataclass
from copy import copy
import logging
import re
from functools import partial

import bgym
from browsergym.experiments import Agent
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.llm.chat_api import ChatModelArgs
from agentlab.llm.llm_utils import ParseError, parse_html_tags_raise, retry
import agentlab.agents.dynamic_prompting as dp

from focus_agent.retriever.embedding_retriever import (
    OpenAIRetriever,
    OpenAIRetrieverArgs,
    TextEmbeddingRetriever,
    TextEmbeddingRetrieverArgs,
)
from focus_agent.llm_configs import MODEL_CONFIGS_DICT
from focus_agent.retriever.utils import get_chunks_from_tokenizer
from .utils import add_line_numbers_to_tree


@dataclass
class EmbeddingRetrieverAgentArgs(GenericAgentArgs):
    flags: GenericPromptFlags = None
    chat_model_args: ChatModelArgs = None
    retriever_args: OpenAIRetrieverArgs = None
    max_retry: int = 4
    agent_name: str = None

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            if (
                self.agent_name == None
            ):  # some attributes might be temporarily args.CrossProd for hyperparameter generation
                self.agent_name = (
                    f"EmbeddingRetrieverAgent-{self.chat_model_args.model_name}".replace("/", "_")
                )
        except AttributeError:
            pass

    def make_agent(self) -> Agent:
        return EmbeddingRetrieverAgent(
            self.chat_model_args,
            self.flags,
            self.retriever_args,
            self.max_retry,
        )


class EmbeddingRetrieverAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: ChatModelArgs,
        flags: GenericPromptFlags,
        retriever_args: OpenAIRetrieverArgs | TextEmbeddingRetrieverArgs,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)

        if isinstance(retriever_args, OpenAIRetrieverArgs):
            self.retriever = OpenAIRetriever(retriever_args)
        elif isinstance(retriever_args, TextEmbeddingRetrieverArgs):
            self.retriever = TextEmbeddingRetriever(args=retriever_args)
        else:
            raise ValueError(
                "retriever_args must be either OpenAIRetrieverArgs or TextEmbeddingRetrieverArgs"
            )

    def get_new_obs(self, obs: dict) -> dict:
        query = obs["goal"] + "\n" + obs["history"]
        axtree_txt = obs["axtree_txt"]
        axtree_chunks = []
        if self.retriever.args.use_recursive_text_splitter:
            from langchain_text_splitters.character import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter()
            axtree_chunks = text_splitter.split_text(axtree_txt)
        else:
            axtree_chunks = get_chunks_from_tokenizer(
                axtree_txt, self.retriever.args.chunk_size, self.retriever.args.overlap
            )

        scores, indices = self.retriever.retrieve(query, axtree_chunks)

        new_tree = ""
        top_indices = indices.tolist()
        if isinstance(top_indices, list) and top_indices and isinstance(top_indices[0], list):
            top_indices = top_indices[0]

        for i, index in enumerate(top_indices):
            if not isinstance(index, int):
                continue
            if index < 0 or index >= len(axtree_chunks):
                continue
            new_tree += f"\n\nChunk {i}:\n{axtree_chunks[index]}"

        return new_tree

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
        obs["axtree_txt"] = self.get_new_obs(obs)
        action, info = super().get_action(obs)
        info.extra_info["chunked_tree"] = obs["axtree_txt"]
        info.extra_info["retriever_query"] = obs["goal"] + "\n" + obs["history"]
        return action, info
