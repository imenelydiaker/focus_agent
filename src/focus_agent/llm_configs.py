"""File for creating LLM configs that are not available in agentlab."""

from agentlab.llm.chat_api import (
    AzureModelArgs,
    OpenRouterModelArgs,
    SelfHostedModelArgs,
    OpenAIModelArgs,
)

default_oss_llms_args = {
    "n_retry_server": 2,
    "temperature": 0.01,
}


MODEL_CONFIGS_DICT = {
    "openrouter/qwen/qwen3.5-9b": OpenRouterModelArgs(
        model_name="qwen/qwen3.5-9b",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=2_000,
        temperature=0.6,
    ),
    "openrouter/qwen/qwen3-235b-a22b-2507": OpenRouterModelArgs(
        model_name="qwen/qwen3-235b-a22b-2507",
        max_total_tokens=40_000,
        max_input_tokens=40_000,
        max_new_tokens=2_000,
        temperature=0.0, # 0.6
    ),
    "openrouter/qwen/qwen-2.5-72b-instruct": OpenRouterModelArgs(
        model_name="qwen/qwen-2.5-72b-instruct",
        max_total_tokens=40_000,
        max_input_tokens=40_000,
        max_new_tokens=2_000,
        temperature=0.0, # 0.6
    ),
    "openrouter/qwen/qwen3-max": OpenRouterModelArgs(
        model_name="qwen/qwen3-max",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=2_000,
        temperature=0.6,
    ),
    "openrouter/qwen/qwen3-8b": OpenRouterModelArgs(
        model_name="qwen/qwen3-8b",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=2_000,
        temperature=0.6,
    ),
    "openrouter/qwen/qwen3-32b": OpenRouterModelArgs(
        model_name="qwen/qwen3-32b",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=2_000,
        temperature=0.6,
    ),
    "openai/gpt-5-nano-2025-08-07": OpenAIModelArgs(
        model_name="gpt-5-nano-2025-08-07",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=4_000,
        temperature=1,  # temperature param not supported by gpt-5
        vision_support=True,
    ),
    "openai/gpt-5-mini-2025-08-07": OpenAIModelArgs(
        model_name="gpt-5-mini-2025-08-07",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=4_000,
        temperature=1,  # temperature param not supported by gpt-5
        vision_support=True,
    ),
    "openai/gpt-5-2025-08-07": OpenAIModelArgs(
        model_name="gpt-5-2025-08-07",
        max_total_tokens=40_000,
        max_input_tokens=40_000,
        max_new_tokens=4_000,
        temperature=1,  # temperature param not supported by gpt-5
        vision_support=True,
    ),
    "openai/gpt-4.1-2025-04-14": OpenAIModelArgs(
        model_name="gpt-4.1-2025-04-14",
        max_total_tokens=40_000,
        max_input_tokens=40_000,
        max_new_tokens=16_384,
        vision_support=True,
        temperature=0.0,
    ),
    "openai/gpt-4.1-mini-2025-04-14": OpenAIModelArgs(
        model_name="gpt-4.1-mini-2025-04-14",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "gpt-5-mini": AzureModelArgs(
        deployment_name="gpt-5-mini-2025-08-07",
        model_name="gpt-5-mini-2025-08-07",
        max_new_tokens=16_384,
        max_input_tokens=128_000,
        max_total_tokens=128_000,
        vision_support=True,
        # temperature=0.1,
    ),
    "gpt-4.1-nano": AzureModelArgs(
        deployment_name="gpt-4.1-nano-2025-04-14",
        model_name="gpt-4.1-nano-2025-04-14",
        max_new_tokens=16_384,
        max_input_tokens=128_000,
        max_total_tokens=128_000,
        vision_support=True,
        temperature=0.0,
    ),
    "gpt-4.1-mini": AzureModelArgs(
        model_name="gpt-4.1-mini",
        max_new_tokens=16_384,
        max_input_tokens=128_000,
        max_total_tokens=128_000,
        vision_support=True,
        temperature=0.0,
    ),
    "gpt-4.1": AzureModelArgs(
        deployment_name="gpt-4.1-2025-04-14",
        model_name="gpt-4.1-2025-04-14",
        max_new_tokens=16_384,
        max_input_tokens=128_000,
        max_total_tokens=128_000,
        vision_support=True,
        temperature=0.0,
    ),
    "gpt-4o-mini": AzureModelArgs(
        deployment_name="gpt-4o-mini-2024-07-18",
        model_name="gpt-4o-mini-2024-07-18",
        max_new_tokens=16_384,
        max_input_tokens=128_000,
        max_total_tokens=128_000,
        vision_support=True,
        temperature=0.0,
    ),
    "gpt-4o": AzureModelArgs(
        deployment_name="gpt-4o-2024-11-20",
        model_name="gpt-4o-2024-11-20",
        max_new_tokens=16_384,
        max_input_tokens=128_000,
        max_total_tokens=128_000,
        vision_support=True,
        temperature=0.0,
    ),
    "vllm/qwen-3-30b-a3b-thinking-2507": SelfHostedModelArgs(
        model_name="Qwen/Qwen3-30B-A3B-Thinking-2507",
        backend="vllm",
        max_input_tokens=128_000,
        max_total_tokens=128_000,
        **default_oss_llms_args,
    ),
    "vllm/qwen3-4b-instruct": SelfHostedModelArgs(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        backend="vllm",
        max_input_tokens=40_000,
        max_total_tokens=40_000,
        **default_oss_llms_args,
    ),
    "vllm/qwen3-4b-thinking": SelfHostedModelArgs(
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        backend="vllm",
        max_input_tokens=40_000,
        max_total_tokens=40_000,
        max_new_tokens=4_000,
        **default_oss_llms_args,
    ),
    "vllm/qwen3-8b": SelfHostedModelArgs(
        model_name="Qwen/Qwen3-8B",
        backend="vllm",
        max_input_tokens=128_000,
        max_total_tokens=128_000,
        **default_oss_llms_args,
    ),
    "vllm/qwen3.5-27b": SelfHostedModelArgs(
        model_name="Qwen/Qwen3.5-27B",
        backend="vllm",
        max_input_tokens=128_000,
        max_total_tokens=128_000,
        **default_oss_llms_args,
    ),
    "azure/meta-llama-3.1-70b-instruct": AzureModelArgs(
        model_name="Meta-Llama-3.1-70B-Instruct",
        deployment_name="Meta-Llama-3.1-70B-Instruct",
    ),
    "azure/meta-llama-3.1-8b-instruct": AzureModelArgs(
        model_name="Meta-Llama-3.1-8B-Instruct",
        deployment_name="Meta-Llama-3.1-8B-Instruct",
    ),
    # Add other models here
}
