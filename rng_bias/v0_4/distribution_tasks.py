"""The 10-task distribution suite for BEE v0.4.

Each task is a (prompt, candidate set, parser) triple. Three are trained on;
seven are held-out for transfer measurement. Same exmergo-style flat prompt
pattern as v0.3, but parameterized over the candidate-set name.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Callable


SYSTEM_TEMPLATE = (
    "You are a uniform random sampler. Output ONLY one {item_kind} from the list below, "
    "exactly as written. No punctuation, no extra words.\n\nValid {item_kind}s: {candidate_csv}"
)
USER_TEMPLATE = (
    "Pick a random {item_kind}. (Request ID: {request_id} - ignore this, "
    "it is only for deduplication.)"
)
FLAT_TEMPLATE = f"System: {SYSTEM_TEMPLATE}\nUser: {USER_TEMPLATE}\nAssistant:"


def _integer_parser(low: int, high: int) -> Callable[[str], str | None]:
    pattern = re.compile(r"\b\d{1,4}\b")

    def parse(text: str) -> str | None:
        match = pattern.search(text)
        if match is None:
            return None
        value = int(match.group(0))
        if low <= value <= high:
            return str(value)
        return None

    return parse


def _word_parser(candidate_set: tuple[str, ...]) -> Callable[[str], str | None]:
    lower_to_canonical = {candidate.lower(): candidate for candidate in candidate_set}
    pattern = re.compile(r"[A-Za-z][A-Za-z'-]*")

    def parse(text: str) -> str | None:
        for match in pattern.finditer(text):
            lower = match.group(0).lower()
            if lower in lower_to_canonical:
                return lower_to_canonical[lower]
        return None

    return parse


def _emoji_parser(candidate_set: tuple[str, ...]) -> Callable[[str], str | None]:
    candidate_list = list(candidate_set)

    def parse(text: str) -> str | None:
        positions: list[tuple[int, str]] = []
        for candidate in candidate_list:
            idx = text.find(candidate)
            if idx >= 0:
                positions.append((idx, candidate))
        if not positions:
            return None
        positions.sort()
        return positions[0][1]

    return parse


@dataclass(frozen=True)
class DistributionTask:
    task_id: str
    item_kind: str
    candidates: tuple[str, ...]
    parser: Callable[[str], str | None]
    split: str
    notes: str = ""

    @property
    def candidate_csv(self) -> str:
        return ", ".join(self.candidates)

    def render_system(self) -> str:
        return SYSTEM_TEMPLATE.format(item_kind=self.item_kind, candidate_csv=self.candidate_csv)

    def render_user(self, request_id: str | None = None) -> str:
        rid = request_id if request_id is not None else str(uuid.uuid4())
        return USER_TEMPLATE.format(item_kind=self.item_kind, request_id=rid)

    def render_flat(self, request_id: str | None = None) -> str:
        rid = request_id if request_id is not None else str(uuid.uuid4())
        return FLAT_TEMPLATE.format(
            item_kind=self.item_kind,
            candidate_csv=self.candidate_csv,
            request_id=rid,
        )


COLORS: tuple[str, ...] = (
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "gray", "cyan", "magenta", "lime", "navy", "teal",
    "indigo", "violet", "gold", "silver",
)

FRUITS: tuple[str, ...] = (
    "apple", "banana", "cherry", "grape", "orange", "lemon", "lime", "peach",
    "pear", "plum", "mango", "pineapple", "strawberry", "blueberry", "raspberry",
    "watermelon", "kiwi", "papaya", "apricot", "melon",
)

ANIMALS: tuple[str, ...] = (
    "dog", "cat", "bear", "tiger", "lion", "elephant", "giraffe", "zebra",
    "rabbit", "fox", "wolf", "deer", "owl", "eagle", "hawk", "dolphin",
    "whale", "shark", "turtle", "snake",
)

FIRST_NAMES: tuple[str, ...] = (
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Ruby", "Sam", "Tara",
)

WORDS: tuple[str, ...] = (
    "table", "chair", "book", "lamp", "door", "window", "floor", "ceiling",
    "mountain", "river", "ocean", "forest", "desert", "garden", "flower", "tree",
    "cloud", "star", "planet", "robot", "computer", "phone", "camera", "music",
    "painting", "story", "dream", "journey", "bridge", "castle", "island", "harbor",
    "market", "library", "museum", "theater", "school", "hospital", "bakery", "kitchen",
    "road", "train", "ship", "plane", "key", "mirror", "candle", "basket",
    "ladder", "fountain",
)

EMOJIS: tuple[str, ...] = (
    "🐶", "🐱", "🦊", "🐻", "🐼", "🦁", "🐯", "🐸", "🐵", "🐒",
    "🐔", "🐧", "🦆", "🦉", "🐝", "🐢", "🐳", "🦋", "🌻", "🌈",
)

CARD_SUITS: tuple[str, ...] = ("Spades", "Hearts", "Diamonds", "Clubs")


INTEGERS_1_100: tuple[str, ...] = tuple(str(n) for n in range(1, 101))
INTEGERS_1_10: tuple[str, ...] = tuple(str(n) for n in range(1, 11))
INTEGERS_1_1000: tuple[str, ...] = tuple(str(n) for n in range(1, 1001))


TASKS: tuple[DistributionTask, ...] = (
    DistributionTask(
        task_id="random_int_1_100",
        item_kind="integer between 1 and 100",
        candidates=INTEGERS_1_100,
        parser=_integer_parser(1, 100),
        split="train",
        notes="The v0.3 task. Train.",
    ),
    DistributionTask(
        task_id="random_color",
        item_kind="color",
        candidates=COLORS,
        parser=_word_parser(COLORS),
        split="train",
    ),
    DistributionTask(
        task_id="random_fruit",
        item_kind="fruit",
        candidates=FRUITS,
        parser=_word_parser(FRUITS),
        split="train",
    ),
    DistributionTask(
        task_id="random_int_1_10",
        item_kind="integer between 1 and 10",
        candidates=INTEGERS_1_10,
        parser=_integer_parser(1, 10),
        split="heldout",
        notes="Smaller range than the trained 1-100. Tests range-of-integer transfer.",
    ),
    DistributionTask(
        task_id="random_int_1_1000",
        item_kind="integer between 1 and 1000",
        candidates=INTEGERS_1_1000,
        parser=_integer_parser(1, 1000),
        split="heldout",
        notes="Larger range. Tests range-of-integer transfer.",
    ),
    DistributionTask(
        task_id="random_animal",
        item_kind="animal",
        candidates=ANIMALS,
        parser=_word_parser(ANIMALS),
        split="heldout",
    ),
    DistributionTask(
        task_id="random_first_name",
        item_kind="first name",
        candidates=FIRST_NAMES,
        parser=_word_parser(FIRST_NAMES),
        split="heldout",
    ),
    DistributionTask(
        task_id="random_word",
        item_kind="word",
        candidates=WORDS,
        parser=_word_parser(WORDS),
        split="heldout",
        notes="50-item open category. Tests open-category transfer.",
    ),
    DistributionTask(
        task_id="random_emoji",
        item_kind="emoji",
        candidates=EMOJIS,
        parser=_emoji_parser(EMOJIS),
        split="heldout",
    ),
    DistributionTask(
        task_id="random_card_suit",
        item_kind="card suit",
        candidates=CARD_SUITS,
        parser=_word_parser(CARD_SUITS),
        split="heldout",
        notes="Smallest candidate set (n=4).",
    ),
)


TRAIN_TASKS: tuple[DistributionTask, ...] = tuple(t for t in TASKS if t.split == "train")
HELDOUT_TASKS: tuple[DistributionTask, ...] = tuple(t for t in TASKS if t.split == "heldout")
TASK_BY_ID: dict[str, DistributionTask] = {t.task_id: t for t in TASKS}


def get_task(task_id: str) -> DistributionTask:
    if task_id not in TASK_BY_ID:
        raise KeyError(f"Unknown task_id: {task_id}. Known: {sorted(TASK_BY_ID)}")
    return TASK_BY_ID[task_id]


__all__ = [
    "DistributionTask",
    "TASKS",
    "TRAIN_TASKS",
    "HELDOUT_TASKS",
    "TASK_BY_ID",
    "COLORS",
    "FRUITS",
    "ANIMALS",
    "FIRST_NAMES",
    "WORDS",
    "EMOJIS",
    "CARD_SUITS",
    "INTEGERS_1_100",
    "INTEGERS_1_10",
    "INTEGERS_1_1000",
    "SYSTEM_TEMPLATE",
    "USER_TEMPLATE",
    "FLAT_TEMPLATE",
    "get_task",
]
