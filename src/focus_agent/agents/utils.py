import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")


def get_nb_tokens(text):
    return len(encoder.encode(text))


def add_line_numbers_to_tree(axtree_txt: str) -> str:
    """
    Adds line numbers to the tree text.
    """
    lines = axtree_txt.strip().splitlines()
    numbered_lines = [f"{i+1:>4} {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def remove_no_bid_lines(axtree_txt: str) -> str:
    """
    Cleans the AXTree text by removing elements with no bid
    """
    lines = axtree_txt.splitlines()
    if not lines:
        return axtree_txt

    # The root line carries no bid but anchors the tree, so it is always kept.
    cleaned_lines = [lines[0]] + [line for line in lines[1:] if "[" in line and "]" in line]
    return "\n".join(cleaned_lines)
