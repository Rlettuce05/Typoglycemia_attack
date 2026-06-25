import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence


WORD_TOKEN_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")


@dataclass(frozen=True)
class WordToken:
    index: int
    token: str
    word: str
    prefix: str
    suffix: str


@dataclass(frozen=True)
class AttackEdit:
    token_index: int
    original: str
    replacement: str
    attack_name: str


@dataclass(frozen=True)
class AttackResult:
    original_text: str
    poisoned_text: str
    edits: tuple[AttackEdit, ...]

    @property
    def changed_words(self):
        return len(self.edits)


class PoisoningBaseline:
    """Shared framework for lightweight poisoning baselines."""

    attack_name = "baseline"

    def __init__(self, seed=0, min_word_length=4):
        if min_word_length <= 0:
            raise ValueError("min_word_length must be greater than 0")
        self.min_word_length = min_word_length
        self.random = random.Random(seed)

    def poison_text(self, text, max_changed_words=2):
        if max_changed_words <= 0:
            raise ValueError("max_changed_words must be greater than 0")

        original_text = str(text)
        tokens = original_text.split()
        edits = []
        changed_indices = set()

        for word_token in self._rank_tokens(self._word_tokens(tokens), original_text):
            if len(edits) >= max_changed_words:
                break
            if word_token.index in changed_indices:
                continue

            candidates = self._replacement_candidates(word_token, original_text, tuple(edits))
            replacement = self._select_candidate(word_token.word, candidates)
            if replacement is None:
                continue

            poisoned_token = self._replace_token_word(word_token.token, replacement)
            tokens[word_token.index] = poisoned_token
            edits.append(
                AttackEdit(
                    token_index=word_token.index,
                    original=word_token.token,
                    replacement=poisoned_token,
                    attack_name=self.attack_name,
                )
            )
            changed_indices.add(word_token.index)

        return AttackResult(
            original_text=original_text,
            poisoned_text=" ".join(tokens),
            edits=tuple(edits),
        )

    def poison_dataframe(
        self,
        data_frame,
        text_column="text",
        image_column=None,
        output_column="poisoned_text",
        max_changed_words=2,
        include_edits=False,
    ):
        if text_column not in data_frame.columns:
            raise ValueError(f"text_column '{text_column}' not found in data_frame columns")
        if image_column is not None and image_column not in data_frame.columns:
            raise ValueError(f"image_column '{image_column}' not found in data_frame columns")

        columns = [text_column]
        if image_column is not None:
            columns.insert(0, image_column)
        poisoned_df = data_frame[columns].copy()

        results = [
            self.poison_text(text, max_changed_words=max_changed_words)
            for text in data_frame[text_column]
        ]
        poisoned_df[output_column] = [result.poisoned_text for result in results]
        poisoned_df["changed_words"] = [result.changed_words for result in results]
        if include_edits:
            poisoned_df["edits"] = [
                [edit.__dict__.copy() for edit in result.edits]
                for result in results
            ]
        return poisoned_df

    def _word_tokens(self, tokens):
        word_tokens = []
        for index, token in enumerate(tokens):
            match = WORD_TOKEN_RE.match(token)
            if not match:
                continue
            prefix, word, suffix = match.groups()
            normalized_word = word.lower()
            if normalized_word.isalpha() and len(normalized_word) >= self.min_word_length:
                word_tokens.append(
                    WordToken(
                        index=index,
                        token=token,
                        word=normalized_word,
                        prefix=prefix,
                        suffix=suffix,
                    )
                )
        return word_tokens

    def _rank_tokens(self, word_tokens, text):
        return word_tokens

    def _replacement_candidates(self, word_token, text, edits):
        raise NotImplementedError

    def _select_candidate(self, original_word, candidates):
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            normalized = str(candidate).lower()
            if normalized == original_word or normalized in seen:
                continue
            if normalized.isalpha():
                unique_candidates.append(normalized)
                seen.add(normalized)
        if not unique_candidates:
            return None
        return self.random.choice(unique_candidates)

    def _replace_token_word(self, token, replacement):
        match = WORD_TOKEN_RE.match(token)
        if not match:
            return token

        prefix, original_word, suffix = match.groups()
        if original_word.isupper():
            cased_replacement = replacement.upper()
        elif original_word[0].isupper():
            cased_replacement = replacement.capitalize()
        else:
            cased_replacement = replacement.lower()
        return prefix + cased_replacement + suffix


class CharmerBaseline(PoisoningBaseline):
    """Character-level perturbation baseline inspired by Charmer."""

    attack_name = "charmer"

    KEYBOARD_NEIGHBORS = {
        "a": "s",
        "b": "v",
        "c": "x",
        "d": "s",
        "e": "r",
        "f": "d",
        "g": "f",
        "h": "g",
        "i": "o",
        "j": "h",
        "k": "j",
        "l": "k",
        "m": "n",
        "n": "b",
        "o": "p",
        "p": "o",
        "q": "w",
        "r": "e",
        "s": "a",
        "t": "r",
        "u": "i",
        "v": "c",
        "w": "q",
        "x": "z",
        "y": "u",
        "z": "x",
    }

    def __init__(
        self,
        seed=0,
        min_word_length=4,
        operations=("swap", "delete", "insert", "replace"),
        preserve_edges=True,
    ):
        super().__init__(seed=seed, min_word_length=min_word_length)
        self.operations = tuple(operations)
        self.preserve_edges = preserve_edges

    def _rank_tokens(self, word_tokens, text):
        return sorted(word_tokens, key=lambda token: (-len(token.word), token.index))

    def _replacement_candidates(self, word_token, text, edits):
        word = word_token.word
        positions = self._editable_positions(word)
        candidates = []

        if "swap" in self.operations:
            for position in positions:
                if position + 1 >= len(word):
                    continue
                if self.preserve_edges and position + 1 == len(word) - 1:
                    continue
                letters = list(word)
                letters[position], letters[position + 1] = letters[position + 1], letters[position]
                candidates.append("".join(letters))

        if "delete" in self.operations and len(word) > self.min_word_length:
            for position in positions:
                candidates.append(word[:position] + word[position + 1:])

        if "insert" in self.operations:
            for position in positions:
                candidates.append(word[:position + 1] + word[position] + word[position + 1:])

        if "replace" in self.operations:
            for position in positions:
                replacement_char = self.KEYBOARD_NEIGHBORS.get(word[position])
                if replacement_char is None:
                    continue
                candidates.append(word[:position] + replacement_char + word[position + 1:])

        return candidates

    def _editable_positions(self, word):
        if self.preserve_edges and len(word) > 2:
            return range(1, len(word) - 1)
        return range(len(word))


class TextFoolerBaseline(PoisoningBaseline):
    """Word-substitution baseline with TextFooler-style importance ranking."""

    attack_name = "textfooler"

    DEFAULT_SYNONYMS = {
        "airplane": ("aircraft",),
        "artist": ("painter",),
        "bicycle": ("bike",),
        "bike": ("bicycle",),
        "building": ("structure",),
        "car": ("automobile",),
        "child": ("kid",),
        "children": ("kids",),
        "large": ("big",),
        "little": ("small",),
        "man": ("person",),
        "people": ("persons", "individuals"),
        "person": ("individual",),
        "quick": ("fast",),
        "road": ("street",),
        "small": ("little",),
        "street": ("road",),
        "woman": ("person",),
    }

    def __init__(
        self,
        seed=0,
        min_word_length=4,
        synonym_provider: Optional[
            Mapping[str, Sequence[str]] | Callable[[str], Iterable[str]]
        ] = None,
        importance_scorer: Optional[Callable[[str, str], float]] = None,
    ):
        super().__init__(seed=seed, min_word_length=min_word_length)
        self.synonym_provider = synonym_provider or self.DEFAULT_SYNONYMS
        self.importance_scorer = importance_scorer or self._default_importance_scorer

    def _rank_tokens(self, word_tokens, text):
        return sorted(
            word_tokens,
            key=lambda token: (-self.importance_scorer(text, token.word), token.index),
        )

    def _replacement_candidates(self, word_token, text, edits):
        if callable(self.synonym_provider):
            return self.synonym_provider(word_token.word)
        return self.synonym_provider.get(word_token.word, ())

    def _default_importance_scorer(self, text, word):
        return float(len(word))