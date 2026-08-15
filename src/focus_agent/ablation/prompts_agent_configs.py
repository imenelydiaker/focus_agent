from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o

from focus_agent.retriever import (
    FocusPromptFlags,
)
from focus_agent.llm_configs import MODEL_CONFIGS_DICT
from ..agents.focus_agent import FocusAgentArgs


FLAGS_GPT_4o = FLAGS_GPT_4o.copy()
FLAGS_GPT_4o.obs.use_think_history = True

# Restrictive prompt agent config
FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_RESTRICTIVE = FocusAgentArgs(
    agent_name="FocusAgent-4.1-Retriever-4.1-Mini-Restrictive",
    chat_model_args=MODEL_CONFIGS_DICT["gpt-4.1"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=MODEL_CONFIGS_DICT["gpt-4.1-mini"],
    retriever_prompt_flags=FocusPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    retriever_type="restrictive",
    max_retry=4,
)

# Neutral prompt agent config
FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_NEUTRAL = FocusAgentArgs(
    agent_name="FocusAgent-4.1-Retriever-4.1-Mini-Neutral",
    chat_model_args=MODEL_CONFIGS_DICT["gpt-4.1"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=MODEL_CONFIGS_DICT["gpt-4.1-mini"],
    retriever_prompt_flags=FocusPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    retriever_type="neutral",
    max_retry=4,
)
