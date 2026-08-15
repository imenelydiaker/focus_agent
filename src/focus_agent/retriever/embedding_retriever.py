from dataclasses import dataclass
from typing import Union, Literal
import logging

import os

import torch

import tiktoken
from openai import OpenAI, AzureOpenAI

from focus_agent.retriever.text_embedding_client import TextEmbeddingClient


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


@dataclass
class OpenAIRetrieverArgs:
    client: str = "openai"  # or "azure", "openrouter"
    model_name: str = None
    top_k: int = 5
    chunk_size: int = 100
    overlap: int = 10
    measure: Literal["cosine", "dot"] = "cosine"
    normalize_embeddings: bool = True
    use_recursive_text_splitter: bool = False


class OpenAIRetriever:
    def __init__(self, args: OpenAIRetrieverArgs):
        self.args = args
        self.model_name = args.model_name

        if args.client == "openai":
            self.client = OpenAI()
        elif args.client == "azure":
            self.client = AzureOpenAI()
        elif args.client == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )

    def encode(self, text: Union[str, list[str]]):
        return self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding

    def retrieve(self, query: str, chunks: Union[str, list[str]]):
        logging.info(f"Encoding {len(chunks)} chunks...")
        chunks_embedddings = [torch.tensor(self.encode(chunk)) for chunk in chunks]
        logging.info(f"Encoding query: {query}")
        query_embeddings = self.encode(query)
        chunks_embedddings = torch.stack(chunks_embedddings)
        query_embeddings = torch.tensor(query_embeddings)

        if self.args.normalize_embeddings:
            query_embeddings = normalize_embeddings(query_embeddings)
            chunks_embedddings = normalize_embeddings(chunks_embedddings)

        similarity_scores = torch.nn.functional.cosine_similarity(
            query_embeddings, chunks_embedddings
        )
        scores, indices = torch.topk(similarity_scores, k=min(self.args.top_k, len(chunks)))
        return scores, indices


@dataclass
class TextEmbeddingRetrieverArgs:
    model_name: str
    top_k: int
    chunk_size: int
    overlap: int
    measure: Literal["cosine", "dot"] = "dot"
    normalize_embeddings: bool = True
    use_recursive_text_splitter: bool = False


class TextEmbeddingRetriever:
    def __init__(self, args: TextEmbeddingRetrieverArgs):
        self.args = args
        self.client = TextEmbeddingClient()
        self.task = (
            "Given a task goal, retrieve relevant chunks to interact with to answer the query"
        )

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}\n"

    def encode(self, text: Union[str, list[str]]):
        if isinstance(text, str):
            text = [text]
        return self.client.embed(text)

    def retrieve(self, query: str, chunks: Union[str, list[str]]):
        query = self.get_detailed_instruct(self.task, query)

        query_embedding = torch.tensor(self.encode(query))
        chunks_embeddings = torch.tensor(self.encode(chunks))

        if self.args.normalize_embeddings:
            query_embedding = normalize_embeddings(query_embedding)
            chunks_embeddings = normalize_embeddings(chunks_embeddings)

        if self.args.measure == "cosine":
            similarity_scores = torch.nn.functional.cosine_similarity(
                query_embedding, chunks_embeddings
            )
        elif self.args.measure == "dot":
            similarity_scores = query_embedding @ chunks_embeddings.T

        scores, indices = torch.topk(similarity_scores, k=min(self.args.top_k, len(chunks)))
        return scores, indices
