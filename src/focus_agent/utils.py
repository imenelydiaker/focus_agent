from browsergym.experiments.loop import ExpResult, get_exp_result, yield_all_exp_results
import gzip
import pandas as pd
import pickle
import os, sys


def get_all_axtrees(savedir_base):
    exps = yield_all_exp_results(savedir_base)
    all_axtrees = []
    # counter = 0
    for exp in exps:
        exp_dir = exp.exp_args.exp_dir
        num_files = len(os.listdir(exp_dir))
        traces_axtrees = []
        for step in range(num_files):
            if os.path.exists(f"{exp_dir}/step_{step}.pkl.gz"):
                with gzip.open(f"{exp_dir}/step_{step}.pkl.gz", "rb") as f:
                    step_info = pickle.load(f)
                    if step_info.obs is not None:
                        traces_axtrees.append(step_info.obs["axtree_txt"])
            else:
                break
        all_axtrees.extend(traces_axtrees)
        # counter += 1
        # if counter % 10 == 0:
        #     break
    return all_axtrees


def reformat_summary(text):
    to_strip = [
        "Here is a one-sentence summary of the content:",
        "Here is a summary of the content in one sentence:",
    ]

    lines = text.splitlines()

    i = 0
    while i < len(lines):
        for strip in to_strip:
            if strip in lines[i]:
                # Capture the exact indentation from the summary line
                indent = lines[i][: len(lines[i].rstrip())]

                # Find the next non-empty line and replace the summary line
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        lines[i] = indent + lines[j].strip()
                        lines[i] = lines[i].replace(strip, "").rstrip()
                        del lines[j]
                        break
        i += 1

    for line in lines:
        # remove empty lines with only \n
        if not line.strip():
            lines.remove(line)

    return "\n".join(lines)


def concat_user_messages(messages: list):
    texts = []
    for message in messages:
        if message["role"] == "user":
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "text":
                        texts.append(content["text"])
            elif isinstance(message["content"], str):
                texts.append(message["content"])
    return "\n".join(texts)


def clean_chat_messages(chat_messages: list):
    return [
        chat_message
        for chat_message in chat_messages
        if chat_message["role"] != "assistant"
        and not (
            chat_message["role"] == "user"
            and any(
                error in chat_message["content"]
                for error in [
                    "Missing the key <action> in the answer.",
                    "Missing the key <think> in the answer",
                ]
            )
        )
    ]
