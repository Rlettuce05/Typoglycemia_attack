"""Summarize paired CLIPScore results for poisoning experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pvariance


DEFAULT_ORIGINAL_LABELS = ("original", "clean", "baseline", "normal")
SAMPLE_ID_COLUMNS = ("sample_id", "image_id", "id", "file_path", "image_path", "File Path")
METHOD_COLUMNS = ("method", "attack", "variant", "perturbation", "condition")
PROMPT_COLUMNS = ("prompt", "caption", "text", "Caption")
SCORE_COLUMNS = ("clip_score", "clipscore", "score", "CLIPScore")


@dataclass(frozen=True)
class ClipScoreRecord:
    sample_id: str
    method: str
    prompt: str
    clip_score: float


@dataclass(frozen=True)
class ClipScoreDeltaRow:
    sample_id: str
    method: str
    original_prompt: str
    perturbed_prompt: str
    original_clip_score: float
    perturbed_clip_score: float
    clip_score_delta: float


@dataclass(frozen=True)
class MethodSummary:
    method: str
    count: int
    mean_delta: float
    variance_delta: float
    mean_absolute_delta: float
    min_delta: float
    max_delta: float
    representative_sample_id: str
    representative_delta: float


def load_clip_score_records(
    path,
    sample_id_column=None,
    method_column=None,
    prompt_column=None,
    score_column=None,
):
    rows = _read_rows(Path(path))
    if not rows:
        raise ValueError("input file has no records")

    fieldnames = sorted({key for row in rows for key in row})
    sample_id_column = _resolve_column(fieldnames, sample_id_column, SAMPLE_ID_COLUMNS, "sample id")
    method_column = _resolve_column(fieldnames, method_column, METHOD_COLUMNS, "method")
    prompt_column = _resolve_column(fieldnames, prompt_column, PROMPT_COLUMNS, "prompt")
    score_column = _resolve_column(fieldnames, score_column, SCORE_COLUMNS, "CLIPScore")

    records = []
    for row_number, row in enumerate(rows, start=2):
        sample_id = _required_text(row, sample_id_column, row_number)
        method = _required_text(row, method_column, row_number)
        prompt = _required_text(row, prompt_column, row_number)
        clip_score = _required_float(row, score_column, row_number)
        records.append(ClipScoreRecord(sample_id, method, prompt, clip_score))
    return records


def build_clip_delta_rows(records, original_labels=DEFAULT_ORIGINAL_LABELS):
    original_by_sample = {}
    variant_records = []

    for record in records:
        if _matches_original_label(record.method, original_labels):
            if record.sample_id in original_by_sample:
                raise ValueError(f"multiple original rows found for sample id {record.sample_id!r}")
            original_by_sample[record.sample_id] = record
        else:
            variant_records.append(record)

    if not original_by_sample:
        labels = ", ".join(original_labels)
        raise ValueError(f"no original rows found; expected one of these method labels: {labels}")
    if not variant_records:
        raise ValueError("no perturbed rows found")

    missing_originals = sorted(
        {record.sample_id for record in variant_records if record.sample_id not in original_by_sample}
    )
    if missing_originals:
        preview = ", ".join(missing_originals[:5])
        suffix = "" if len(missing_originals) <= 5 else f", ... ({len(missing_originals)} total)"
        raise ValueError(f"missing original rows for sample ids: {preview}{suffix}")

    delta_rows = []
    for record in variant_records:
        original = original_by_sample[record.sample_id]
        delta_rows.append(
            ClipScoreDeltaRow(
                sample_id=record.sample_id,
                method=record.method,
                original_prompt=original.prompt,
                perturbed_prompt=record.prompt,
                original_clip_score=original.clip_score,
                perturbed_clip_score=record.clip_score,
                clip_score_delta=record.clip_score - original.clip_score,
            )
        )
    return delta_rows


def summarize_clip_delta_rows(delta_rows):
    grouped = defaultdict(list)
    for row in delta_rows:
        grouped[row.method].append(row)

    summaries = []
    for method in sorted(grouped):
        rows = grouped[method]
        deltas = [row.clip_score_delta for row in rows]
        representative = max(rows, key=lambda row: abs(row.clip_score_delta))
        summaries.append(
            MethodSummary(
                method=method,
                count=len(rows),
                mean_delta=mean(deltas),
                variance_delta=pvariance(deltas) if len(deltas) > 1 else 0.0,
                mean_absolute_delta=mean(abs(delta) for delta in deltas),
                min_delta=min(deltas),
                max_delta=max(deltas),
                representative_sample_id=representative.sample_id,
                representative_delta=representative.clip_score_delta,
            )
        )
    return summaries


def write_delta_rows_csv(delta_rows, path):
    path = Path(path)
    _ensure_parent(path)
    fieldnames = [
        "sample_id",
        "method",
        "original_prompt",
        "perturbed_prompt",
        "original_clip_score",
        "perturbed_clip_score",
        "clip_score_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in delta_rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "method": row.method,
                    "original_prompt": row.original_prompt,
                    "perturbed_prompt": row.perturbed_prompt,
                    "original_clip_score": _format_float(row.original_clip_score),
                    "perturbed_clip_score": _format_float(row.perturbed_clip_score),
                    "clip_score_delta": _format_float(row.clip_score_delta),
                }
            )


def write_summary_csv(summaries, path):
    path = Path(path)
    _ensure_parent(path)
    fieldnames = [
        "method",
        "count",
        "mean_delta",
        "variance_delta",
        "mean_absolute_delta",
        "min_delta",
        "max_delta",
        "representative_sample_id",
        "representative_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "method": summary.method,
                    "count": summary.count,
                    "mean_delta": _format_float(summary.mean_delta),
                    "variance_delta": _format_float(summary.variance_delta),
                    "mean_absolute_delta": _format_float(summary.mean_absolute_delta),
                    "min_delta": _format_float(summary.min_delta),
                    "max_delta": _format_float(summary.max_delta),
                    "representative_sample_id": summary.representative_sample_id,
                    "representative_delta": _format_float(summary.representative_delta),
                }
            )


def render_markdown_report(delta_rows, summaries):
    summary_by_method = {summary.method: summary for summary in summaries}
    representative_by_method = {
        method: max((row for row in delta_rows if row.method == method), key=lambda row: abs(row.clip_score_delta))
        for method in summary_by_method
    }

    lines = [
        "# CLIPScore Difference Summary",
        "",
        "## Paper Table Candidate",
        "",
        (
            "| Method | n | Mean delta | Variance | Mean absolute delta | "
            "Largest drop | Largest gain | Representative sample |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| {method} | {count} | {mean_delta} | {variance_delta} | {mean_absolute_delta} | "
            "{min_delta} | {max_delta} | {sample_id} |".format(
                method=summary.method,
                count=summary.count,
                mean_delta=_format_float(summary.mean_delta),
                variance_delta=_format_float(summary.variance_delta),
                mean_absolute_delta=_format_float(summary.mean_absolute_delta),
                min_delta=_format_float(summary.min_delta),
                max_delta=_format_float(summary.max_delta),
                sample_id=summary.representative_sample_id,
            )
        )

    lines.extend(["", "## Slide Summary", ""])
    if summaries:
        largest_average_drop = min(summaries, key=lambda summary: summary.mean_delta)
        most_variable = max(summaries, key=lambda summary: summary.variance_delta)
        strongest_example = max(delta_rows, key=lambda row: abs(row.clip_score_delta))
        lines.extend(
            [
                (
                    "- Largest average CLIPScore decrease: "
                    f"{largest_average_drop.method} ({_format_float(largest_average_drop.mean_delta)})."
                ),
                (
                    "- Most variable perturbation: "
                    f"{most_variable.method} (variance {_format_float(most_variable.variance_delta)})."
                ),
                (
                    "- Strongest representative example: "
                    f"{strongest_example.method} on {strongest_example.sample_id} "
                    f"({_format_float(strongest_example.clip_score_delta)})."
                ),
            ]
        )

    lines.extend(["", "## Representative Examples", ""])
    for method in sorted(representative_by_method):
        row = representative_by_method[method]
        lines.extend(
            [
                f"### {method}",
                "",
                f"- Sample: {row.sample_id}",
                f"- CLIPScore delta: {_format_float(row.clip_score_delta)}",
                f"- Original prompt: {_truncate(row.original_prompt)}",
                f"- Perturbed prompt: {_truncate(row.perturbed_prompt)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(delta_rows, summaries, path):
    path = Path(path)
    _ensure_parent(path)
    path.write_text(render_markdown_report(delta_rows, summaries), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create CLIPScore delta tables and a Markdown summary from paired poisoning results."
    )
    parser.add_argument("input_path", help="CSV, TSV, JSON, or JSONL file with original and perturbed CLIPScore rows.")
    parser.add_argument("--output-prefix", "-o", help="Output prefix. Defaults to the input path without suffix.")
    parser.add_argument("--sample-id-column", help="Column identifying paired samples.")
    parser.add_argument("--method-column", help="Column containing method or attack labels.")
    parser.add_argument("--prompt-column", help="Column containing prompts or captions.")
    parser.add_argument("--score-column", help="Column containing CLIPScore values.")
    parser.add_argument(
        "--original-label",
        action="append",
        dest="original_labels",
        help="Method label for original/clean rows. Can be provided more than once.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    original_labels = tuple(args.original_labels or DEFAULT_ORIGINAL_LABELS)
    records = load_clip_score_records(
        input_path,
        sample_id_column=args.sample_id_column,
        method_column=args.method_column,
        prompt_column=args.prompt_column,
        score_column=args.score_column,
    )
    delta_rows = build_clip_delta_rows(records, original_labels=original_labels)
    summaries = summarize_clip_delta_rows(delta_rows)
    delta_path, summary_path, report_path = _output_paths(input_path, args.output_prefix)

    write_delta_rows_csv(delta_rows, delta_path)
    write_summary_csv(summaries, summary_path)
    write_markdown_report(delta_rows, summaries, report_path)

    print(f"Wrote {delta_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


def _read_rows(path):
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


def _resolve_column(fieldnames, explicit_name, candidates, role):
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


def _required_text(row, column, row_number):
    value = row.get(column)
    if value is None or str(value).strip() == "":
        raise ValueError(f"row {row_number} has an empty value for column {column!r}")
    return str(value).strip()


def _required_float(row, column, row_number):
    value = _required_text(row, column, row_number)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number} has a non-numeric CLIPScore value: {value!r}") from exc


def _matches_original_label(method, original_labels):
    normalized_method = _normalize_name(method)
    return any(normalized_method == _normalize_name(label) for label in original_labels)


def _normalize_name(value):
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _format_float(value):
    return f"{value:.6f}"


def _truncate(value, max_length=140):
    text = " ".join(str(value).split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _ensure_parent(path):
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def _output_paths(input_path, output_prefix):
    prefix = Path(output_prefix) if output_prefix else input_path.with_suffix("")
    prefix_text = str(prefix)
    return (
        Path(f"{prefix_text}_clip_delta_rows.csv"),
        Path(f"{prefix_text}_clip_summary.csv"),
        Path(f"{prefix_text}_clip_report.md"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
