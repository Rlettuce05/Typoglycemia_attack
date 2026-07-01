"""Aggregate DF-Impact token alignments into word-level scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
DF_WORD_COLUMNS = ("word", "clean_word", "token", "term")
DF_SCORE_COLUMNS = ("DF", "df_score", "document_frequency_score", "score")
OPERATION_ORDER = {"match": 0, "substitute": 1, "insert": 2, "delete": 3}


@dataclass(frozen=True)
class AlignmentStep:
    sample_id: str
    operation: str
    clean_token: str
    poisoned_token: str
    clean_prompt: str
    poisoned_prompt: str
    clean_start_char: int | None
    clean_end_char: int | None
    poisoned_start_char: int | None
    poisoned_end_char: int | None
    l2_distance: float | None
    cosine_distance: float | None
    step_cost: float


@dataclass(frozen=True)
class WordSpan:
    word_index: int
    word: str
    start_char: int | None
    end_char: int | None
    source: str
    fallback: bool = False


@dataclass(frozen=True)
class WordImpactRow:
    sample_id: str
    word_index: int
    word: str
    clean_word: str
    poisoned_word: str
    operation: str
    token_count: int
    changed_token_count: int
    mean_l2_distance: float | None
    max_l2_distance: float | None
    mean_cosine_distance: float | None
    max_cosine_distance: float | None
    total_step_cost: float
    impact_score: float
    document_frequency: int
    df_score: float
    df_impact_score: float


@dataclass(frozen=True)
class WordImpactSummary:
    word: str
    occurrence_count: int
    sample_count: int
    changed_occurrences: int
    mean_impact_score: float
    max_impact_score: float
    df_score: float
    mean_df_impact_score: float
    max_df_impact_score: float
    representative_sample_id: str
    representative_operation: str


@dataclass
class WordAccumulator:
    sample_id: str
    word_index: int
    word: str
    clean_word: str = ""
    poisoned_word: str = ""
    operations: set[str] = field(default_factory=set)
    token_count: int = 0
    changed_token_count: int = 0
    l2_values: list[float] = field(default_factory=list)
    cosine_values: list[float] = field(default_factory=list)
    total_step_cost: float = 0.0

    def add(self, step, clean_word, poisoned_word):
        self.operations.add(step.operation)
        self.token_count += 1
        self.total_step_cost += step.step_cost
        if step.operation != "match":
            self.changed_token_count += 1
        if step.l2_distance is not None:
            self.l2_values.append(step.l2_distance)
        if step.cosine_distance is not None:
            self.cosine_values.append(step.cosine_distance)
        if clean_word and not self.clean_word:
            self.clean_word = clean_word
        if poisoned_word and not self.poisoned_word:
            self.poisoned_word = poisoned_word

    def to_row(self, document_frequency, df_score):
        impact_score = self.impact_score()
        return WordImpactRow(
            sample_id=self.sample_id,
            word_index=self.word_index,
            word=self.word,
            clean_word=self.clean_word,
            poisoned_word=self.poisoned_word,
            operation=summarize_operations(self.operations),
            token_count=self.token_count,
            changed_token_count=self.changed_token_count,
            mean_l2_distance=mean(self.l2_values) if self.l2_values else None,
            max_l2_distance=max(self.l2_values) if self.l2_values else None,
            mean_cosine_distance=mean(self.cosine_values) if self.cosine_values else None,
            max_cosine_distance=max(self.cosine_values) if self.cosine_values else None,
            total_step_cost=self.total_step_cost,
            impact_score=impact_score,
            document_frequency=document_frequency,
            df_score=df_score,
            df_impact_score=impact_score * df_score,
        )

    def impact_score(self):
        if self.l2_values:
            return mean(self.l2_values)
        if self.changed_token_count:
            return self.total_step_cost
        return 0.0


def load_alignment_steps(path):
    rows = read_rows(Path(path))
    if not rows:
        raise ValueError("alignment file has no records")
    return [parse_alignment_row(row, row_number) for row_number, row in enumerate(rows, start=1)]


def build_word_impact_rows(steps, df_scores=None):
    """Build one word-level impact row per sample and word occurrence."""

    df_scores = df_scores or {}
    accumulators = []
    for sample_id, sample_steps in group_steps_by_sample(steps).items():
        fallback_index = 0
        sample_accumulators = {}
        for step in sample_steps:
            span, clean_word, poisoned_word = resolve_step_words(step, fallback_index)
            if span.fallback:
                fallback_index += 1
            key = (span.source, span.word_index, span.start_char, span.end_char, span.word)
            if key not in sample_accumulators:
                sample_accumulators[key] = WordAccumulator(
                    sample_id=sample_id,
                    word_index=span.word_index,
                    word=span.word,
                )
            sample_accumulators[key].add(step, clean_word, poisoned_word)
        accumulators.extend(sample_accumulators.values())

    document_frequency = compute_document_frequency(accumulators)
    computed_df_scores = {
        word: math.log1p(count)
        for word, count in document_frequency.items()
    }
    computed_df_scores.update({normalize_word(word): score for word, score in df_scores.items()})

    rows = []
    for accumulator in accumulators:
        normalized_word = normalize_word(accumulator.word)
        rows.append(
            accumulator.to_row(
                document_frequency=document_frequency.get(normalized_word, 0),
                df_score=computed_df_scores.get(normalized_word, 0.0),
            )
        )
    return sorted(rows, key=lambda row: (row.sample_id, row.word_index, row.word))


def summarize_word_impacts(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[normalize_word(row.word)].append(row)

    summaries = []
    for word in sorted(grouped):
        word_rows = grouped[word]
        representative = max(word_rows, key=lambda row: (row.df_impact_score, row.impact_score, row.sample_id))
        samples = {row.sample_id for row in word_rows}
        summaries.append(
            WordImpactSummary(
                word=word,
                occurrence_count=len(word_rows),
                sample_count=len(samples),
                changed_occurrences=sum(1 for row in word_rows if row.changed_token_count > 0),
                mean_impact_score=mean(row.impact_score for row in word_rows),
                max_impact_score=max(row.impact_score for row in word_rows),
                df_score=representative.df_score,
                mean_df_impact_score=mean(row.df_impact_score for row in word_rows),
                max_df_impact_score=representative.df_impact_score,
                representative_sample_id=representative.sample_id,
                representative_operation=representative.operation,
            )
        )
    return sorted(summaries, key=lambda summary: (-summary.max_df_impact_score, summary.word))


def load_df_scores(path, word_column=None, score_column=None):
    rows = read_rows(Path(path))
    if not rows:
        raise ValueError("DF table has no records")
    fieldnames = sorted({key for row in rows for key in row})
    word_column = resolve_column(fieldnames, word_column, DF_WORD_COLUMNS, "DF word")
    score_column = resolve_column(fieldnames, score_column, DF_SCORE_COLUMNS, "DF score")

    scores = {}
    for row_number, row in enumerate(rows, start=2):
        word = normalize_word(required_text(row, word_column, row_number))
        scores[word] = required_float(row, score_column, row_number)
    return scores


def write_word_rows_csv(rows, path):
    path = Path(path)
    ensure_parent(path)
    fieldnames = [
        "sample_id",
        "word_index",
        "word",
        "clean_word",
        "poisoned_word",
        "operation",
        "token_count",
        "changed_token_count",
        "mean_l2_distance",
        "max_l2_distance",
        "mean_cosine_distance",
        "max_cosine_distance",
        "total_step_cost",
        "impact_score",
        "document_frequency",
        "df_score",
        "df_impact_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(word_row_to_dict(row))


def write_summary_csv(summaries, path):
    path = Path(path)
    ensure_parent(path)
    fieldnames = [
        "word",
        "occurrence_count",
        "sample_count",
        "changed_occurrences",
        "mean_impact_score",
        "max_impact_score",
        "df_score",
        "mean_df_impact_score",
        "max_df_impact_score",
        "representative_sample_id",
        "representative_operation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary_to_dict(summary))


def write_markdown_report(summaries, path, top_k=20):
    path = Path(path)
    ensure_parent(path)
    path.write_text(render_markdown_report(summaries, top_k=top_k), encoding="utf-8")


def render_markdown_report(summaries, top_k=20):
    lines = [
        "# DF-Impact Word Summary",
        "",
        "## Top DF-Integrated Words",
        "",
        (
            "| Rank | Word | Samples | Occurrences | Mean impact | Max impact | "
            "DF score | Max DF-Impact | Representative sample |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, summary in enumerate(summaries[:top_k], start=1):
        lines.append(
            "| {rank} | {word} | {samples} | {occurrences} | {mean_impact} | {max_impact} | "
            "{df_score} | {df_impact} | {sample} |".format(
                rank=rank,
                word=summary.word,
                samples=summary.sample_count,
                occurrences=summary.occurrence_count,
                mean_impact=format_float(summary.mean_impact_score),
                max_impact=format_float(summary.max_impact_score),
                df_score=format_float(summary.df_score),
                df_impact=format_float(summary.max_df_impact_score),
                sample=summary.representative_sample_id,
            )
        )
    lines.extend(
        [
            "",
            "## Scoring Notes",
            "",
            "- `impact_score` is the mean token L2 hidden-state distance for a word.",
            "- Insert/delete-only words fall back to their accumulated DP step cost.",
            "- `df_impact_score = impact_score * df_score`.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_alignment_row(row, row_number):
    clean = row.get("clean") if isinstance(row.get("clean"), dict) else {}
    poisoned = row.get("poisoned") if isinstance(row.get("poisoned"), dict) else {}
    sample_id = row.get("sample_id") or clean.get("sample_id") or poisoned.get("sample_id")
    operation = row.get("operation")
    if not sample_id:
        raise ValueError(f"alignment row {row_number} has an empty 'sample_id'")
    if not operation:
        raise ValueError(f"alignment row {row_number} has an empty 'operation'")
    return AlignmentStep(
        sample_id=str(sample_id),
        operation=str(operation),
        clean_token=str(clean.get("token", row.get("clean_token", "")) or ""),
        poisoned_token=str(poisoned.get("token", row.get("poisoned_token", "")) or ""),
        clean_prompt=str(clean.get("prompt", row.get("clean_prompt", "")) or ""),
        poisoned_prompt=str(poisoned.get("prompt", row.get("poisoned_prompt", "")) or ""),
        clean_start_char=parse_optional_int(clean.get("start_char", row.get("clean_start_char")), row_number),
        clean_end_char=parse_optional_int(clean.get("end_char", row.get("clean_end_char")), row_number),
        poisoned_start_char=parse_optional_int(poisoned.get("start_char", row.get("poisoned_start_char")), row_number),
        poisoned_end_char=parse_optional_int(poisoned.get("end_char", row.get("poisoned_end_char")), row_number),
        l2_distance=parse_optional_float(row.get("l2_distance"), row_number),
        cosine_distance=parse_optional_float(row.get("cosine_distance"), row_number),
        step_cost=parse_optional_float(row.get("step_cost"), row_number) or 0.0,
    )


def resolve_step_words(step, fallback_index):
    clean_span = span_from_prompt(
        step.clean_prompt,
        step.clean_start_char,
        step.clean_end_char,
        source="clean",
    )
    poisoned_span = span_from_prompt(
        step.poisoned_prompt,
        step.poisoned_start_char,
        step.poisoned_end_char,
        source="poisoned",
    )
    clean_word = clean_span.word if clean_span else normalize_token_text(step.clean_token)
    poisoned_word = poisoned_span.word if poisoned_span else normalize_token_text(step.poisoned_token)
    primary_span = clean_span or poisoned_span
    if primary_span:
        return primary_span, clean_word, poisoned_word

    fallback_word = clean_word or poisoned_word or step.operation
    span = WordSpan(
        word_index=fallback_index,
        word=normalize_word(fallback_word),
        start_char=None,
        end_char=None,
        source="fallback",
        fallback=True,
    )
    return span, clean_word, poisoned_word


def span_from_prompt(prompt, start_char, end_char, source):
    if not prompt or start_char is None or end_char is None:
        return None
    for word_index, match in enumerate(WORD_RE.finditer(prompt)):
        if start_char < match.end() and end_char > match.start():
            return WordSpan(
                word_index=word_index,
                word=normalize_word(match.group(0)),
                start_char=match.start(),
                end_char=match.end(),
                source=source,
            )
    return None


def group_steps_by_sample(steps):
    grouped = defaultdict(list)
    for step in steps:
        grouped[step.sample_id].append(step)
    return grouped


def compute_document_frequency(accumulators):
    sample_ids_by_word = defaultdict(set)
    for accumulator in accumulators:
        sample_ids_by_word[normalize_word(accumulator.word)].add(accumulator.sample_id)
    return {word: len(sample_ids) for word, sample_ids in sample_ids_by_word.items()}


def read_rows(path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of row objects")
        return data

    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def resolve_column(fieldnames, explicit_name, candidates, role):
    if explicit_name:
        if explicit_name not in fieldnames:
            raise ValueError(f"{role} column {explicit_name!r} was not found")
        return explicit_name
    field_by_normalized_name = {normalize_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        match = field_by_normalized_name.get(normalize_name(candidate))
        if match:
            return match
    raise ValueError(f"could not infer {role} column from available columns: {', '.join(fieldnames)}")


def required_text(row, column, row_number):
    value = row.get(column)
    if value is None or str(value).strip() == "":
        raise ValueError(f"row {row_number} has an empty value for column {column!r}")
    return str(value).strip()


def required_float(row, column, row_number):
    value = required_text(row, column, row_number)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number} has a non-numeric value for {column!r}: {value!r}") from exc


def parse_optional_int(value, row_number):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alignment row {row_number} has invalid integer value: {value!r}") from exc


def parse_optional_float(value, row_number):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alignment row {row_number} has invalid float value: {value!r}") from exc


def normalize_word(value):
    return "".join(character for character in str(value).casefold() if character.isalnum())


def normalize_name(value):
    return normalize_word(value)


def normalize_token_text(token):
    value = str(token).strip()
    if not value:
        return ""
    for prefix in ("##", chr(0x0120)):
        if value.startswith(prefix):
            value = value[len(prefix):]
    if value.endswith("</w>"):
        value = value[:-4]
    return normalize_word(value)


def summarize_operations(operations):
    return "+".join(sorted(operations, key=lambda operation: (OPERATION_ORDER.get(operation, 99), operation)))


def word_row_to_dict(row):
    return {
        "sample_id": row.sample_id,
        "word_index": row.word_index,
        "word": row.word,
        "clean_word": row.clean_word,
        "poisoned_word": row.poisoned_word,
        "operation": row.operation,
        "token_count": row.token_count,
        "changed_token_count": row.changed_token_count,
        "mean_l2_distance": format_optional_float(row.mean_l2_distance),
        "max_l2_distance": format_optional_float(row.max_l2_distance),
        "mean_cosine_distance": format_optional_float(row.mean_cosine_distance),
        "max_cosine_distance": format_optional_float(row.max_cosine_distance),
        "total_step_cost": format_float(row.total_step_cost),
        "impact_score": format_float(row.impact_score),
        "document_frequency": row.document_frequency,
        "df_score": format_float(row.df_score),
        "df_impact_score": format_float(row.df_impact_score),
    }


def summary_to_dict(summary):
    return {
        "word": summary.word,
        "occurrence_count": summary.occurrence_count,
        "sample_count": summary.sample_count,
        "changed_occurrences": summary.changed_occurrences,
        "mean_impact_score": format_float(summary.mean_impact_score),
        "max_impact_score": format_float(summary.max_impact_score),
        "df_score": format_float(summary.df_score),
        "mean_df_impact_score": format_float(summary.mean_df_impact_score),
        "max_df_impact_score": format_float(summary.max_df_impact_score),
        "representative_sample_id": summary.representative_sample_id,
        "representative_operation": summary.representative_operation,
    }


def format_optional_float(value):
    return "" if value is None else format_float(value)


def format_float(value):
    return f"{value:.6f}"


def ensure_parent(path):
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def output_paths(input_path, output_prefix):
    prefix = Path(output_prefix) if output_prefix else input_path.with_suffix("")
    prefix_text = str(prefix)
    return (
        Path(f"{prefix_text}_word_impact_rows.csv"),
        Path(f"{prefix_text}_word_impact_summary.csv"),
        Path(f"{prefix_text}_df_impact_report.md"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate DF-Impact token alignments into word-level DF-integrated scores."
    )
    parser.add_argument("alignment_path", help="Alignment CSV/TSV/JSON/JSONL from df_impact_matching.py.")
    parser.add_argument("--output-prefix", "-o", help="Output prefix. Defaults to alignment path without suffix.")
    parser.add_argument("--df-table", help="Optional Typoglycemia DF table with word and DF columns.")
    parser.add_argument("--df-word-column", help="Word column in --df-table.")
    parser.add_argument("--df-score-column", help="DF score column in --df-table.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of words to include in the Markdown report.")
    args = parser.parse_args(argv)

    input_path = Path(args.alignment_path)
    df_scores = None
    if args.df_table:
        df_scores = load_df_scores(
            args.df_table,
            word_column=args.df_word_column,
            score_column=args.df_score_column,
        )
    steps = load_alignment_steps(input_path)
    rows = build_word_impact_rows(steps, df_scores=df_scores)
    summaries = summarize_word_impacts(rows)
    row_path, summary_path, report_path = output_paths(input_path, args.output_prefix)
    write_word_rows_csv(rows, row_path)
    write_summary_csv(summaries, summary_path)
    write_markdown_report(summaries, report_path, top_k=args.top_k)
    print(f"Wrote {row_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
