FILTERS = {
    "4o": "gpt-4o",
    "o1-mini": "openai/o1-mini-2024-09-12",
    "4o-mini": "gpt-4o-mini",
    "claude-3.5": "anthropic/claude-3.5-sonnet:beta",
    "llama-70b": "meta-llama/llama-3.1-70b-instruct",
}

OBS_PROMPT_AXTREE_PREFIX = """\
\n## AXTree:
Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.

Note: You can only interact with visible elements. If the "visible" tag is not
present, the element is not visible on the page.\n
"""

OBS_PROMPT_HTML_PREFIX = """\
\n## HTML:\n
"""
