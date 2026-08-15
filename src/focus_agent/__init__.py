"""FocusAgent: retrieval-based observation pruning for web agents.

The three agents are available directly from the package root::

    from focus_agent import FocusAgent, EmbeddingRetrieverAgent, BM25RetrieverAgent

Each has a matching ``*Args`` class used to configure it, and ``FocusAgent``
additionally takes ``FocusPromptFlags`` to control its retriever prompt.
"""

from .agents import (
    BM25RetrieverAgent,
    BM25RetrieverAgentArgs,
    BM25RetrieverAgentFlags,
    EmbeddingRetrieverAgent,
    EmbeddingRetrieverAgentArgs,
    FocusAgent,
    FocusAgentArgs,
)
from .retriever import FocusPrompt, FocusPromptFlags

__version__ = "0.0.1"

__all__ = [
    "BM25RetrieverAgent",
    "BM25RetrieverAgentArgs",
    "BM25RetrieverAgentFlags",
    "EmbeddingRetrieverAgent",
    "EmbeddingRetrieverAgentArgs",
    "FocusAgent",
    "FocusAgentArgs",
    "FocusPrompt",
    "FocusPromptFlags",
    "__version__",
]
