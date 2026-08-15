import os
import re
from typing import Tuple
import logging
import json

from openai import OpenAI

from browsergym.core.task import AbstractBrowserTask
from datasets import load_dataset

logger = logging.getLogger(__name__)
_DATASET = None


# ---------------------------------------------------------------------------
# LLM-as-Judge: faithful reconstruction of DeepResearch/evaluation/prompt.py
# (QA judge) + evaluate_deepsearch_official.py (judge call logic)
# ---------------------------------------------------------------------------

# From DeepResearch evaluation/prompt.py  — the QA correctness judge prompt
QA_JUDGE_PROMPT = """\
You have been provided with, a ground-truth answer, and a predicted answer below.
Given the question, you need to score the predicted answer by comparing it to the ground-truth answer.

Here are the grading criteria to follow:
1. Review the predicted answer carefully to ensure the keywords and numerical values are exactly the same as the ground truth.
2. Ensure the predicted answer does not contain any conflicting statements.
3. It is OK if the predicted answer contains more information than the ground truth, as long as it is factually accurate relative to the ground truth.

Respond with ONLY a JSON object in the following format, with no extra text:
{{"score": <0 or 1>, "reason": "<brief explanation>"}}

- score 1 means the predicted answer is correct based on the ground truth.
- score 0 means the predicted answer is incorrect.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {prediction}
"""

WEBARENA_FUZZY_MATCH = """
Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.

question: {question}
reference answer: {reference}

all the string 'N/A' that you see is a special sequence that means 'not achievable'
student answer: {prediction}

Conclude the judgement by correct/incorrect/partially correct.
"""

# WEBARENA_FUZZY_MATCH = """
# Help check if an agent message contains the expected information for a question given in a reference answer.

# question: {question}
# reference answer: {reference}

# all the string 'N/A' that you see is a special sequence that means 'not achievable'
# student answer: {pred}

# Conclude the judgement by correct/incorrect/partially correct.
# """


def _call_llm_judge(
    question: str, ground_truth: str, prediction: str, prompt_type: str = "qa_judge"
) -> Tuple[float, str]:
    """
    Mirrors the judge call in DeepResearch evaluate_deepsearch_official.py.
    Returns (score: 0.0 or 1.0, reason: str).
    Falls back to F1 heuristic if the LLM call fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("JUDGE_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", None)  # optional custom endpoint

    if not api_key:
        logger.warning("No OPENAI_API_KEY set — falling back to F1 heuristic for judge.")
        return _f1_fallback(ground_truth, prediction)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        if prompt_type == "webarena_fuzzy_match":
            prompt = WEBARENA_FUZZY_MATCH.format(
                question=question, reference=ground_truth, prediction=prediction
            )
        elif prompt_type == "qa_judge":
            prompt = QA_JUDGE_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                prediction=prediction,
            )
        response = client.chat.completions.create(
            model=os.environ.get("JUDGE_MODEL", "gpt-5"),
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=256,
        )
        raw = response.choices[0].message.content.strip()

        if prompt_type == "webarena_fuzzy_match":
            if "partially correct" in response or "incorrect" in response:
                return 0.0, ""
            else:
                assert "correct" in response
                return 1.0, ""

        elif prompt_type == "qa_judge":
            # Strip accidental markdown fences
            raw = re.sub(r"^```json\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            result = json.loads(raw)
            score = float(result.get("score", 0))
            reason = result.get("reason", "")
            return score, reason

    except Exception as e:
        logger.warning(f"LLM judge call failed ({e}), falling back to F1 heuristic.")
        return _f1_fallback(ground_truth, prediction)


def _f1_fallback(ground_truth: str, prediction: str) -> Tuple[float, str]:
    """Token-level F1 fallback (mirrors original WebWalkerQA evaluate.py)."""
    pred_tokens = set(re.findall(r"\w+", prediction.lower()))
    gold_tokens = set(re.findall(r"\w+", ground_truth.lower()))
    if not gold_tokens:
        return 0.0, "Empty ground truth"
    overlap = pred_tokens & gold_tokens
    if not overlap:
        return 0.0, "No token overlap"
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
    recall = len(overlap) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    score = 1.0 if f1 >= 0.8 else 0.0
    return score, f"F1 fallback: {f1:.2f}"


def _pick(item: dict, *keys: str, default=None):
    for key in keys:
        if key in item:
            return item[key]
    return default


def get_dataset():
    global _DATASET
    if _DATASET is None:
        dataset = load_dataset("callanwu/WebWalkerQA", split="main")
        # filter out chinese tasks
        dataset = [item for item in dataset if _pick(item, "info", default="")["lang"] == "en"]
        print(f"Registering {len(dataset)} WebWalker tasks for split 'main'")
        _DATASET = list(dataset)
    return _DATASET


def _message_to_text(message) -> str:
    if isinstance(message, list):
        text_parts = []
        for part in message:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(part))
        return "\n".join(text_parts).strip()
    return str(message or "").strip()


class WebWalkerTask(AbstractBrowserTask):
    def __init__(self, seed: int, task_id: int = 0):
        super().__init__(seed)
        dataset = get_dataset()
        self.item = dataset[task_id]
        self.question = _pick(self.item, "question", default="")
        self.answer = _pick(self.item, "answer", default="")
        self.root_url = _pick(self.item, "root_url", default="")

    @classmethod
    def get_task_id(cls):
        return "webwalkerqa"

    def setup(self, page) -> tuple[str, dict]:
        page.goto(self.root_url, wait_until="domcontentloaded")
        goal = (
            "Starting from this webpage, find the answer to the following question "
            "by navigating the website's subpages.\n\n"
            f"Question: {self.question}\n\n"
            "When you have found the answer, send it as a message in the chat."
        )
        info = {
            "question": self.question,
            "root_url": self.root_url,
            "hop": _pick(_pick(self.item, "info", "Info", default={}), "hop", "Hop"),
            "difficulty": _pick(
                _pick(self.item, "info", "Info", default={}),
                "difficulty_level",
                "Difficulty_Level",
            ),
        }
        return goal, info

    def validate(self, page, chat_messages: list) -> Tuple[float, bool, str, dict]:
        if not chat_messages:
            return 0.0, False, "", {}

        latest = chat_messages[-1]

        if isinstance(latest, dict):
            if latest.get("role") != "assistant":
                return 0.0, False, "", {}
            last_msg = _message_to_text(latest.get("message", "")).lower()
        else:
            last_msg = str(latest).lower().strip()

        if not last_msg:
            return 0.0, False, "", {}

        expected = self.answer.lower().strip()

        # 1. Exact match short-circuit (free, no API call)
        if expected in last_msg:
            return 1.0, True, "Correct!", {"match": "exact"}

        # 2. LLM-as-judge (DeepResearch QA judge prompt)
        score, reason = _call_llm_judge(
            question=self.question,
            ground_truth=self.answer,
            prediction=last_msg,
            prompt_type="qa_judge",
        )
        if score >= 1.0:
            return 1.0, True, f"Correct! ({reason})", {"match": "llm_judge", "reason": reason}

        return 0.0, True, f"Incorrect. ({reason})", {"match": "incorrect", "reason": reason}

    def teardown(self) -> None:
        pass
