"""Summarize CLIPScore deltas by poisoning rate and changed-word count."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pvariance

from clip_score_summary import DEFAULT_ORIGINAL_LABELS, _format_float, _normalize_name


SAMPLE_ID_COLUMNS = ("sample_id", "image_id", "id", "file_path", "image_path", "File Path")
METHOD_COLUMNS = ("method", "attack", "variant", "perturbation", "condition")
PROMPT_COLUMNS = ("prompt", "caption", "text", "Caption")
SCORE_COLUMNS = ("clip_score", "clipscore", "score", "CLIPScore")
CHANGED_WORDS_COLUMNS = ("changed_words", "changed_word_count", "num_changed_words", "edit_count")
POISON_RATE_COLUMNS = ("poison_rate", "poisoning_rate", "pollution_rate", "contamination_rate", "mix_rate")
WORD_COUNT_COLUMNS = ("original_word_count", "word_count", "prompt_word_count", "caption_word_count")
SPLIT_COLUMNS = ("split", "dataset_split", "data_split")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class ScoredPromptRow:
    row_number: int
    sample_id: str
    method: str
    prompt: str
    clip_score: float
    split: str
    changed_words: int | None
    poison_rate: float | None
    word_count: int | None


@dataclass(frozen=True)
class PoisonRateDeltaRow:
    split: str
    sample_id: str
    method: str
    poison_rate: float
    changed_words: int
    original_word_count: int
    original_clip_score: float
    perturbed_clip_score: float
    clip_score_delta: float
    original_prompt: str
    perturbed_prompt: str


@dataclass(frozen=True)
class PoisonRateSummary:
    split: str
    method: str
    poison_rate: float
    count: int
    mean_clip_score_delta: float
    variance_clip_score_delta: float
    mean_absolute_clip_score_delta: float
    min_clip_score_delta: float
    max_clip_score_delta: float
    mean_changed_words: float
    representative_sample_id: str
    representative_delta: float


@dataclass(frozen=True)
class ChangedWordsSummary:
    split: str
    method: str
    changed_words: int
    count: int
    mean_poison_rate: float
    mean_clip_score_delta: float
    variance_clip_score_delta: float
    mean_absolute_clip_score_delta: float
    representative_sample_id: str
    representative_delta: float


def load_scored_prompt_rows(
    path,
    sample_id_column=None,
    method_column=None,
    prompt_column=None,
    score_column=None,
    changed_words_column=None,
    poison_rate_column=None,
    word_count_column=None,
    split_column=None,
):
    rows = read_rows(Path(path))
    if not rows:
        raise ValueError("input file has no records")

    fieldnames = sorted({key for row in rows for key in row})
    sample_id_column = resolve_column(fieldnames, sample_id_column, SAMPLE_ID_COLUMNS, "sample id")
    method_column = resolve_column(fieldnames, method_column, METHOD_COLUMNS, "method")
    prompt_column = resolve_column(fieldnames, prompt_column, PROMPT_COLUMNS, "prompt")
    score_column = resolve_column(fieldnames, score_column, SCORE_COLUMNS, "CLIPScore")
    changed_words_column = resolve_optional_column(
        fieldnames,
        changed_words_column,
        CHANGED_WORDS_COLUMNS,
        "changed words",
    )
    poison_rate_column = resolve_optional_column(fieldnames, poison_rate_column, POISON_RATE_COLUMNS, "poison rate")
    word_count_column = resolve_optional_column(fieldnames, word_count_column, WORD_COUNT_COLUMNS, "word count")
    split_column = resolve_optional_column(fieldnames, split_column, SPLIT_COLUMNS, "split")

    scored_rows = []
    for row_number, row in enumerate(rows, start=2):
        scored_rows.append(
            ScoredPromptRow(
                row_number=row_number,
                sample_id=required_text(row, sample_id_column, row_number),
                method=required_text(row, method_column, row_number),
                prompt=required_text(row, prompt_column, row_number),
                clip_score=required_float(row, score_column, row_number),
                split=optional_text(row, split_column) if split_column else "unspecified",
                changed_words=parse_optional_int(row.get(changed_words_column), changed_words_column, row_number)
                if changed_words_column
                else None,
                poison_rate=parse_optional_rate(row.get(poison_rate_column), poison_rate_column, row_number)
                if poison_rate_column
                else None,
                word_count=parse_optional_int(row.get(word_count_column), word_count_column, row_number)
                if word_count_column
                else None,
            )
        )
    return scored_rows


def build_poison_rate_delta_rows(
    rows,
    original_labels=DEFAULT_ORIGINAL_LABELS,
    rate_precision=4,
):
    if rate_precision < 0:
        raise ValueError("rate_precision must be 0 or greater")

    originals = {}
    variants = []
    for row in rows:
        if matches_original_label(row.method, original_labels):
            key = (row.split, row.sample_id)
            if key in originals:
                raise ValueError(f"multiple original rows found for sample id {row.sample_id!r} in split {row.split!r}")
            originals[key] = row
        else:
            variants.append(row)

    if not originals:
        labels = ", ".join(original_labels)
        raise ValueError(f"no original rows found; expected one of these method labels: {labels}")
    if not variants:
        raise ValueError("no perturbed rows found")

    delta_rows = []
    for variant in variants:
        original = originals.get((variant.split, variant.sample_id))
        if original is None:
            raise ValueError(
                f"missing original row for sample id {variant.sample_id!r} in split {variant.split!r}"
            )
        if variant.changed_words is None:
            raise ValueError(
                f"row {variant.row_number} has no changed-word count; provide one of: "
                f"{', '.join(CHANGED_WORDS_COLUMNS)}"
            )
        original_word_count = (
            original.word_count
            if original.word_count is not None
            else variant.word_count
            if variant.word_count is not None
            else count_prompt_words(original.prompt)
        )
        if original_word_count <= 0:
            raise ValueError(
                f"row {variant.row_number} cannot compute poison rate because original word count is 0"
            )
        poison_rate = variant.poison_rate
        if poison_rate is None:
            poison_rate = variant.changed_words / original_word_count
        if poison_rate < 0 or poison_rate > 1:
            raise ValueError(
                f"row {variant.row_number} computed poison rate outside [0, 1]: {poison_rate!r}"
            )
        poison_rate = round(poison_rate, rate_precision)
        delta_rows.append(
            PoisonRateDeltaRow(
                split=variant.split,
                sample_id=variant.sample_id,
                method=variant.method,
                poison_rate=poison_rate,
                changed_words=variant.changed_words,
                original_word_count=original_word_count,
                original_clip_score=original.clip_score,
                perturbed_clip_score=variant.clip_score,
                clip_score_delta=variant.clip_score - original.clip_score,
                original_prompt=original.prompt,
                perturbed_prompt=variant.prompt,
            )
        )
    return sorted(delta_rows, key=lambda row: (row.split, row.method, row.poison_rate, row.sample_id))


def summarize_by_poison_rate(delta_rows):
    grouped = defaultdict(list)
    for row in delta_rows:
        grouped[(row.split, row.method, row.poison_rate)].append(row)

    summaries = []
    for (split, method, poison_rate), rows in grouped.items():
        deltas = [row.clip_score_delta for row in rows]
        representative = max(rows, key=lambda row: abs(row.clip_score_delta))
        summaries.append(
            PoisonRateSummary(
                split=split,
                method=method,
                poison_rate=poison_rate,
                count=len(rows),
                mean_clip_score_delta=mean(deltas),
                variance_clip_score_delta=pvariance(deltas) if len(deltas) > 1 else 0.0,
                mean_absolute_clip_score_delta=mean(abs(delta) for delta in deltas),
                min_clip_score_delta=min(deltas),
                max_clip_score_delta=max(deltas),
                mean_changed_words=mean(row.changed_words for row in rows),
                representative_sample_id=representative.sample_id,
                representative_delta=representative.clip_score_delta,
            )
        )
    return sorted(summaries, key=lambda summary: (summary.split, summary.method, summary.poison_rate))


def summarize_by_changed_words(delta_rows):
    grouped = defaultdict(list)
    for row in delta_rows:
        grouped[(row.split, row.method, row.changed_words)].append(row)

    summaries = []
    for (split, method, changed_words), rows in grouped.items():
        deltas = [row.clip_score_delta for row in rows]
        representative = max(rows, key=lambda row: abs(row.clip_score_delta))
        summaries.append(
            ChangedWordsSummary(
                split=split,
                method=method,
                changed_words=changed_words,
                count=len(rows),
                mean_poison_rate=mean(row.poison_rate for row in rows),
                mean_clip_score_delta=mean(deltas),
                variance_clip_score_delta=pvariance(deltas) if len(deltas) > 1 else 0.0,
                mean_absolute_clip_score_delta=mean(abs(delta) for delta in deltas),
                representative_sample_id=representative.sample_id,
                representative_delta=representative.clip_score_delta,
            )
        )
    return sorted(summaries, key=lambda summary: (summary.split, summary.method, summary.changed_words))


def write_delta_rows_csv(rows, path):
    path = Path(path)
    ensure_parent(path)
    fieldnames = [
        "split",
        "sample_id",
        "method",
        "poison_rate",
        "changed_words",
        "original_word_count",
        "original_clip_score",
        "perturbed_clip_score",
        "clip_score_delta",
        "original_prompt",
        "perturbed_prompt",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(delta_row_to_dict(row))


def write_poison_rate_summary_csv(summaries, path):
    path = Path(path)
    ensure_parent(path)
    fieldnames = [
        "split",
        "method",
        "poison_rate",
        "count",
        "mean_clip_score_delta",
        "variance_clip_score_delta",
        "mean_absolute_clip_score_delta",
        "min_clip_score_delta",
        "max_clip_score_delta",
        "mean_changed_words",
        "representative_sample_id",
        "representative_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(poison_rate_summary_to_dict(summary))


def write_changed_words_summary_csv(summaries, path):
    path = Path(path)
    ensure_parent(path)
    fieldnames = [
        "split",
        "method",
        "changed_words",
        "count",
        "mean_poison_rate",
        "mean_clip_score_delta",
        "variance_clip_score_delta",
        "mean_absolute_clip_score_delta",
        "representative_sample_id",
        "representative_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(changed_words_summary_to_dict(summary))


def write_markdown_report(rate_summaries, changed_summaries, metadata, path, top_k=20):
    path = Path(path)
    ensure_parent(path)
    path.write_text(
        render_markdown_report(rate_summaries, changed_summaries, metadata, top_k=top_k),
        encoding="utf-8",
    )


def render_markdown_report(rate_summaries, changed_summaries, metadata, top_k=20):
    lines = [
        "# Poison Rate CLIPScore Summary",
        "",
        "## Metadata",
        "",
        f"- Dataset: {metadata['dataset']}",
        f"- Split(s): {', '.join(metadata['splits'])}",
        f"- Attack setting: {metadata['attack_setting']}",
        f"- Model: {metadata['model']}",
        f"- Seed: {metadata['seed']}",
        f"- Git commit: {metadata['git_commit']}",
        f"- Created at: {metadata['created_at']}",
        "",
        "## By Poison Rate",
        "",
        (
            "| Split | Method | Poison rate | n | Mean CLIPScore delta | Variance | "
            "Mean abs. delta | Mean changed words | Representative sample |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in rate_summaries[:top_k]:
        lines.append(
            "| {split} | {method} | {rate} | {count} | {mean_delta} | {variance_delta} | "
            "{mean_abs_delta} | {changed_words} | {sample} |".format(
                split=summary.split,
                method=summary.method,
                rate=_format_float(summary.poison_rate),
                count=summary.count,
                mean_delta=_format_float(summary.mean_clip_score_delta),
                variance_delta=_format_float(summary.variance_clip_score_delta),
                mean_abs_delta=_format_float(summary.mean_absolute_clip_score_delta),
                changed_words=_format_float(summary.mean_changed_words),
                sample=summary.representative_sample_id,
            )
        )

    lines.extend(
        [
            "",
            "## By Changed Words",
            "",
            (
                "| Split | Method | Changed words | n | Mean poison rate | Mean CLIPScore delta | "
                "Mean abs. delta | Representative sample |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for summary in changed_summaries[:top_k]:
        lines.append(
            "| {split} | {method} | {changed_words} | {count} | {rate} | {mean_delta} | "
            "{mean_abs_delta} | {sample} |".format(
                split=summary.split,
                method=summary.method,
                changed_words=summary.changed_words,
                count=summary.count,
                rate=_format_float(summary.mean_poison_rate),
                mean_delta=_format_float(summary.mean_clip_score_delta),
                mean_abs_delta=_format_float(summary.mean_absolute_clip_score_delta),
                sample=summary.representative_sample_id,
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Perturbed rows are paired with original rows by `sample_id` and `split`.",
            "- If no poison-rate column is provided, `poison_rate = changed_words / original_word_count`.",
            "- If no split column is provided, all rows are reported under `unspecified`; do not mix dataset splits.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_metadata_json(metadata, path):
    path = Path(path)
    ensure_parent(path)
    path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_metadata(args, input_path, delta_rows):
    return {
        "attack_setting": args.attack_setting or "unspecified",
        "config": args.config or "unspecified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset or "unspecified",
        "git_commit": get_git_commit_hash(),
        "input_path": str(input_path),
        "model": args.model or "unspecified",
        "output_prefix": str(args.output_prefix or input_path.with_suffix("")),
        "poison_rate_definition": (
            "explicit poison_rate column when present; otherwise changed_words/original_word_count"
        ),
        "rate_precision": args.rate_precision,
        "seed": str(args.seed) if args.seed is not None else "unspecified",
        "splits": sorted({row.split for row in delta_rows}) or ["unspecified"],
    }


def delta_row_to_dict(row):
    return {
        "split": row.split,
        "sample_id": row.sample_id,
        "method": row.method,
        "poison_rate": _format_float(row.poison_rate),
        "changed_words": row.changed_words,
        "original_word_count": row.original_word_count,
        "original_clip_score": _format_float(row.original_clip_score),
        "perturbed_clip_score": _format_float(row.perturbed_clip_score),
        "clip_score_delta": _format_float(row.clip_score_delta),
        "original_prompt": row.original_prompt,
        "perturbed_prompt": row.perturbed_prompt,
    }


def poison_rate_summary_to_dict(summary):
    return {
        "split": summary.split,
        "method": summary.method,
        "poison_rate": _format_float(summary.poison_rate),
        "count": summary.count,
        "mean_clip_score_delta": _format_float(summary.mean_clip_score_delta),
        "variance_clip_score_delta": _format_float(summary.variance_clip_score_delta),
        "mean_absolute_clip_score_delta": _format_float(summary.mean_absolute_clip_score_delta),
        "min_clip_score_delta": _format_float(summary.min_clip_score_delta),
        "max_clip_score_delta": _format_float(summary.max_clip_score_delta),
        "mean_changed_words": _format_float(summary.mean_changed_words),
        "representative_sample_id": summary.representative_sample_id,
        "representative_delta": _format_float(summary.representative_delta),
    }


def changed_words_summary_to_dict(summary):
    return {
        "split": summary.split,
        "method": summary.method,
        "changed_words": summary.changed_words,
        "count": summary.count,
        "mean_poison_rate": _format_float(summary.mean_poison_rate),
        "mean_clip_score_delta": _format_float(summary.mean_clip_score_delta),
        "variance_clip_score_delta": _format_float(summary.variance_clip_score_delta),
        "mean_absolute_clip_score_delta": _format_float(summary.mean_absolute_clip_score_delta),
        "representative_sample_id": summary.representative_sample_id,
        "representative_delta": _format_float(summary.representative_delta),
    }


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

    field_by_normalized_name = {_normalize_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        match = field_by_normalized_name.get(_normalize_name(candidate))
        if match:
            return match
    raise ValueError(f"could not infer {role} column from available columns: {', '.join(fieldnames)}")


def resolve_optional_column(fieldnames, explicit_name, candidates, role):
    if explicit_name:
        return resolve_column(fieldnames, explicit_name, candidates, role)

    field_by_normalized_name = {_normalize_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        match = field_by_normalized_name.get(_normalize_name(candidate))
        if match:
            return match
    return None


def required_text(row, column, row_number):
    value = row.get(column)
    if value is None or str(value).strip() == "":
        raise ValueError(f"row {row_number} has an empty value for column {column!r}")
    return str(value).strip()


def optional_text(row, column):
    value = row.get(column)
    if value is None or str(value).strip() == "":
        return "unspecified"
    return str(value).strip()


def required_float(row, column, row_number):
    value = required_text(row, column, row_number)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number} has a non-numeric value for {column!r}: {value!r}") from exc


def parse_optional_int(value, column, row_number):
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number} has a non-numeric value for {column!r}: {value!r}") from exc
    if not number.is_integer() or number < 0:
        raise ValueError(f"row {row_number} has an invalid non-negative integer for {column!r}: {value!r}")
    return int(number)


def parse_optional_rate(value, column, row_number):
    if value is None or str(value).strip() == "":
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number} has a non-numeric value for {column!r}: {value!r}") from exc
    if rate < 0 or rate > 1:
        raise ValueError(f"row {row_number} has poison rate outside [0, 1] for {column!r}: {value!r}")
    return rate


def matches_original_label(method, original_labels):
    normalized_method = _normalize_name(method)
    return any(normalized_method == _normalize_name(label) for label in original_labels)


def count_prompt_words(prompt):
    return len(WORD_RE.findall(str(prompt)))


def get_git_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def ensure_parent(path):
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def output_paths(input_path, output_prefix):
    prefix = Path(output_prefix) if output_prefix else input_path.with_suffix("")
    prefix_text = str(prefix)
    return (
        Path(f"{prefix_text}_poison_rate_delta_rows.csv"),
        Path(f"{prefix_text}_poison_rate_summary.csv"),
        Path(f"{prefix_text}_changed_words_summary.csv"),
        Path(f"{prefix_text}_poison_rate_report.md"),
        Path(f"{prefix_text}_poison_rate_metadata.json"),
    )


def ensure_outputs_available(paths, overwrite=False):
    if overwrite:
        return
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing output files; pass --overwrite to replace: "
            + ", ".join(existing)
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate paired poisoning CLIPScore rows by poison rate and changed-word count."
    )
    parser.add_argument("input_path", help="CSV, TSV, JSON, or JSONL with original and perturbed CLIPScore rows.")
    parser.add_argument("--output-prefix", "-o", help="Output prefix. Defaults to input path without suffix.")
    parser.add_argument("--sample-id-column", help="Column identifying paired samples.")
    parser.add_argument("--method-column", help="Column containing method or attack labels.")
    parser.add_argument("--prompt-column", help="Column containing prompts or captions.")
    parser.add_argument("--score-column", help="Column containing CLIPScore values.")
    parser.add_argument("--changed-words-column", help="Column containing transformed word counts.")
    parser.add_argument("--poison-rate-column", help="Optional column containing explicit poison rates in [0, 1].")
    parser.add_argument("--word-count-column", help="Optional original prompt word-count denominator column.")
    parser.add_argument("--split-column", help="Optional dataset split column. If omitted, split is `unspecified`.")
    parser.add_argument(
        "--original-label",
        action="append",
        dest="original_labels",
        help="Original/clean method label.",
    )
    parser.add_argument("--dataset", help="Dataset name to record in metadata.")
    parser.add_argument("--attack-setting", help="Attack setting to record in metadata.")
    parser.add_argument("--model", help="CLIP or scoring model name to record in metadata.")
    parser.add_argument("--seed", type=int, help="Seed to record in metadata, if applicable.")
    parser.add_argument("--config", help="Config path or short config identifier to record in metadata.")
    parser.add_argument("--rate-precision", type=int, default=4, help="Decimal places used for poison-rate grouping.")
    parser.add_argument("--top-k", type=int, default=20, help="Rows per table in the Markdown report.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    paths = output_paths(input_path, args.output_prefix)
    ensure_outputs_available(paths, overwrite=args.overwrite)

    rows = load_scored_prompt_rows(
        input_path,
        sample_id_column=args.sample_id_column,
        method_column=args.method_column,
        prompt_column=args.prompt_column,
        score_column=args.score_column,
        changed_words_column=args.changed_words_column,
        poison_rate_column=args.poison_rate_column,
        word_count_column=args.word_count_column,
        split_column=args.split_column,
    )
    original_labels = tuple(args.original_labels or DEFAULT_ORIGINAL_LABELS)
    delta_rows = build_poison_rate_delta_rows(
        rows,
        original_labels=original_labels,
        rate_precision=args.rate_precision,
    )
    rate_summaries = summarize_by_poison_rate(delta_rows)
    changed_summaries = summarize_by_changed_words(delta_rows)
    metadata = build_metadata(args, input_path, delta_rows)

    delta_path, rate_summary_path, changed_summary_path, report_path, metadata_path = paths
    write_delta_rows_csv(delta_rows, delta_path)
    write_poison_rate_summary_csv(rate_summaries, rate_summary_path)
    write_changed_words_summary_csv(changed_summaries, changed_summary_path)
    write_markdown_report(rate_summaries, changed_summaries, metadata, report_path, top_k=args.top_k)
    write_metadata_json(metadata, metadata_path)

    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
