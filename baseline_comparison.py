"""Generate and summarize comparable poisoning baseline outputs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from statistics import mean

import pandas as pd

from clip_score_summary import (
    DEFAULT_ORIGINAL_LABELS,
    ClipScoreRecord,
    _format_float,
    _normalize_name,
    _output_paths,
    _required_float,
    _required_text,
    build_clip_delta_rows,
    summarize_clip_delta_rows,
    write_delta_rows_csv,
)
from typoglycemia_pos_filter import Typoglycemia
from poisoning_baselines import CharmerBaseline, TextFoolerBaseline


TEXT_COLUMNS = ("Caption", "caption", "prompt", "text")
IMAGE_COLUMNS = ("File Path", "image_path", "file_path", "image", "path")
SAMPLE_ID_COLUMNS = ("sample_id", "image_id", "id", "File Path", "image_path", "file_path")
CHANGED_WORDS_COLUMN = "changed_words"


@dataclass(frozen=True)
class AttackPromptRow:
    sample_id: str
    image_path: str
    method: str
    prompt: str
    changed_words: int


@dataclass(frozen=True)
class BaselineComparisonSummary:
    method: str
    count: int
    mean_clip_score_delta: float
    variance_clip_score_delta: float
    mean_absolute_clip_score_delta: float
    mean_changed_words: float
    representative_sample_id: str
    representative_delta: float
    representative_original_prompt: str
    representative_perturbed_prompt: str


def load_input_dataframe(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            return pd.DataFrame(json.loads(line) for line in handle if line.strip())
    if suffix == ".json":
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of row objects")
        return pd.DataFrame(data)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    return pd.read_csv(path, encoding="utf-8-sig")


def generate_attack_prompt_rows(
    data_frame,
    text_column=None,
    image_column=None,
    sample_id_column=None,
    max_changed_words=2,
    typoglycemia_factory=None,
    attacks=None,
):
    if max_changed_words <= 0:
        raise ValueError("max_changed_words must be greater than 0")

    text_column = _resolve_dataframe_column(data_frame, text_column, TEXT_COLUMNS, "text")
    image_column = _resolve_dataframe_column(data_frame, image_column, IMAGE_COLUMNS, "image")
    sample_id_column = _resolve_dataframe_column(
        data_frame,
        sample_id_column,
        SAMPLE_ID_COLUMNS,
        "sample id",
    )

    rows = []
    original_by_index = []
    for row_index, row in data_frame.iterrows():
        sample_id = str(row[sample_id_column])
        image_path = str(row[image_column])
        prompt = str(row[text_column])
        original_by_index.append((sample_id, image_path, prompt))
        rows.append(
            AttackPromptRow(
                sample_id=sample_id,
                image_path=image_path,
                method="original",
                prompt=prompt,
                changed_words=0,
            )
        )

    typoglycemia = typoglycemia_factory() if typoglycemia_factory else Typoglycemia(seed=42)
    typoglycemia.load_data_frame(data_frame)
    typoglycemia.count_words_in_text(text_column=text_column)
    typoglycemia.calculate_DF_scores()
    typoglycemia.gen_shuffled_word()
    typoglycemia_frame = typoglycemia.gen_poisoned_text(
        max_changed_words=max_changed_words,
        text_column=text_column,
        image_column=image_column,
    )
    for row_index, (_, _, original_prompt) in enumerate(original_by_index):
        poisoned_prompt = str(typoglycemia_frame.iloc[row_index][text_column])
        rows.append(
            AttackPromptRow(
                sample_id=original_by_index[row_index][0],
                image_path=original_by_index[row_index][1],
                method="typoglycemia",
                prompt=poisoned_prompt,
                changed_words=count_changed_word_tokens(original_prompt, poisoned_prompt),
            )
        )

    attacks = attacks or (
        CharmerBaseline(seed=42),
        TextFoolerBaseline(seed=42),
    )
    for attack in attacks:
        for sample_id, image_path, original_prompt in original_by_index:
            result = attack.poison_text(original_prompt, max_changed_words=max_changed_words)
            rows.append(
                AttackPromptRow(
                    sample_id=sample_id,
                    image_path=image_path,
                    method=result.edits[0].attack_name if result.edits else attack.attack_name,
                    prompt=result.poisoned_text,
                    changed_words=result.changed_words,
                )
            )
    return rows


def write_attack_prompt_rows_csv(rows, path):
    path = Path(path)
    _ensure_parent(path)
    fieldnames = ["sample_id", "image_path", "method", "prompt", "changed_words"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "image_path": row.image_path,
                    "method": row.method,
                    "prompt": row.prompt,
                    "changed_words": row.changed_words,
                }
            )


def load_scored_comparison_rows(path):
    rows = _read_rows(Path(path))
    if not rows:
        raise ValueError("scored comparison file has no records")

    fieldnames = sorted({key for row in rows for key in row})
    sample_id_column = _resolve_fieldname(fieldnames, SAMPLE_ID_COLUMNS, "sample id")
    method_column = _resolve_fieldname(fieldnames, ("method", "attack", "variant"), "method")
    prompt_column = _resolve_fieldname(fieldnames, ("prompt", "caption", "text", "Caption"), "prompt")
    score_column = _resolve_fieldname(fieldnames, ("clip_score", "clipscore", "score", "CLIPScore"), "CLIPScore")
    changed_words_column = _resolve_fieldname(
        fieldnames,
        (CHANGED_WORDS_COLUMN, "changed_word_count", "num_changed_words"),
        "changed words",
    )

    records = []
    changed_words_by_key = {}
    for row_number, row in enumerate(rows, start=2):
        sample_id = _required_text(row, sample_id_column, row_number)
        method = _required_text(row, method_column, row_number)
        prompt = _required_text(row, prompt_column, row_number)
        clip_score = _required_float(row, score_column, row_number)
        changed_words = int(_required_float(row, changed_words_column, row_number))
        records.append(ClipScoreRecord(sample_id, method, prompt, clip_score))
        changed_words_by_key[(sample_id, _normalize_name(method))] = changed_words
    return records, changed_words_by_key


def summarize_scored_comparison_rows(
    records,
    changed_words_by_key,
    original_labels=DEFAULT_ORIGINAL_LABELS,
):
    delta_rows = build_clip_delta_rows(records, original_labels=original_labels)
    clip_summaries = summarize_clip_delta_rows(delta_rows)
    rows_by_method = {}
    for row in delta_rows:
        rows_by_method.setdefault(row.method, []).append(row)

    summaries = []
    for clip_summary in clip_summaries:
        rows = rows_by_method[clip_summary.method]
        changed_word_values = [
            changed_words_by_key.get((row.sample_id, _normalize_name(row.method)), 0)
            for row in rows
        ]
        representative = max(rows, key=lambda row: abs(row.clip_score_delta))
        summaries.append(
            BaselineComparisonSummary(
                method=clip_summary.method,
                count=clip_summary.count,
                mean_clip_score_delta=clip_summary.mean_delta,
                variance_clip_score_delta=clip_summary.variance_delta,
                mean_absolute_clip_score_delta=clip_summary.mean_absolute_delta,
                mean_changed_words=mean(changed_word_values) if changed_word_values else 0.0,
                representative_sample_id=representative.sample_id,
                representative_delta=representative.clip_score_delta,
                representative_original_prompt=representative.original_prompt,
                representative_perturbed_prompt=representative.perturbed_prompt,
            )
        )
    return delta_rows, summaries


def write_comparison_summary_csv(summaries, path):
    path = Path(path)
    _ensure_parent(path)
    fieldnames = [
        "method",
        "count",
        "mean_clip_score_delta",
        "variance_clip_score_delta",
        "mean_absolute_clip_score_delta",
        "mean_changed_words",
        "representative_sample_id",
        "representative_delta",
        "representative_original_prompt",
        "representative_perturbed_prompt",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "method": summary.method,
                    "count": summary.count,
                    "mean_clip_score_delta": _format_float(summary.mean_clip_score_delta),
                    "variance_clip_score_delta": _format_float(summary.variance_clip_score_delta),
                    "mean_absolute_clip_score_delta": _format_float(summary.mean_absolute_clip_score_delta),
                    "mean_changed_words": _format_float(summary.mean_changed_words),
                    "representative_sample_id": summary.representative_sample_id,
                    "representative_delta": _format_float(summary.representative_delta),
                    "representative_original_prompt": summary.representative_original_prompt,
                    "representative_perturbed_prompt": summary.representative_perturbed_prompt,
                }
            )


def render_comparison_markdown_report(summaries):
    lines = [
        "# Poisoning Baseline Comparison",
        "",
        "## Paper Table Candidate",
        "",
        (
            "| Method | n | Mean CLIPScore delta | Variance | "
            "Mean abs. delta | Mean changed words | Representative sample |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| {method} | {count} | {mean_delta} | {variance_delta} | {mean_abs_delta} | "
            "{changed_words} | {sample_id} |".format(
                method=summary.method,
                count=summary.count,
                mean_delta=_format_float(summary.mean_clip_score_delta),
                variance_delta=_format_float(summary.variance_clip_score_delta),
                mean_abs_delta=_format_float(summary.mean_absolute_clip_score_delta),
                changed_words=_format_float(summary.mean_changed_words),
                sample_id=summary.representative_sample_id,
            )
        )

    lines.extend(["", "## Slide Summary", ""])
    if summaries:
        largest_drop = min(summaries, key=lambda summary: summary.mean_clip_score_delta)
        most_changed = max(summaries, key=lambda summary: summary.mean_changed_words)
        strongest_example = max(summaries, key=lambda summary: abs(summary.representative_delta))
        lines.extend(
            [
                (
                    "- Largest average CLIPScore decrease: "
                    f"{largest_drop.method} ({_format_float(largest_drop.mean_clip_score_delta)})."
                ),
                (
                    "- Highest edit budget usage: "
                    f"{most_changed.method} ({_format_float(most_changed.mean_changed_words)} words)."
                ),
                (
                    "- Strongest representative example: "
                    f"{strongest_example.method} on {strongest_example.representative_sample_id} "
                    f"({_format_float(strongest_example.representative_delta)})."
                ),
            ]
        )

    lines.extend(["", "## Representative Examples", ""])
    for summary in summaries:
        lines.extend(
            [
                f"### {summary.method}",
                "",
                f"- Sample: {summary.representative_sample_id}",
                f"- CLIPScore delta: {_format_float(summary.representative_delta)}",
                f"- Original prompt: {_truncate(summary.representative_original_prompt)}",
                f"- Perturbed prompt: {_truncate(summary.representative_perturbed_prompt)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_comparison_markdown_report(summaries, path):
    path = Path(path)
    _ensure_parent(path)
    path.write_text(render_comparison_markdown_report(summaries), encoding="utf-8")


def count_changed_word_tokens(original_prompt, perturbed_prompt):
    return sum(
        1
        for original, perturbed in zip_longest(
            str(original_prompt).split(),
            str(perturbed_prompt).split(),
        )
        if original != perturbed
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate Typoglycemia, Charmer, and TextFooler comparison prompts, "
            "then optionally summarize CLIPScore results for those prompts."
        )
    )
    parser.add_argument("input_path", help="CSV, TSV, JSON, or JSONL file containing source prompts.")
    parser.add_argument("--output-prefix", "-o", help="Output prefix. Defaults to input path without suffix.")
    parser.add_argument("--text-column", help="Column containing source prompts or captions.")
    parser.add_argument("--image-column", help="Column containing image paths.")
    parser.add_argument("--sample-id-column", help="Column identifying paired samples.")
    parser.add_argument("--max-changed-words", type=int, default=2)
    parser.add_argument(
        "--use-heuristic-pos-tagger",
        action="store_true",
        help="Use Typoglycemia's built-in conservative POS tagger instead of NLTK.",
    )
    parser.add_argument(
        "--scored-results",
        help=(
            "Optional CSV/TSV/JSON/JSONL produced after CLIPScore evaluation. "
            "It must contain sample_id, method, prompt, changed_words, and CLIPScore columns."
        ),
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    output_prefix = Path(args.output_prefix) if args.output_prefix else input_path.with_suffix("")
    attack_rows_path = Path(f"{output_prefix}_attack_prompts.csv")

    data_frame = load_input_dataframe(input_path)
    typoglycemia_factory = _create_heuristic_typoglycemia if args.use_heuristic_pos_tagger else None
    attack_rows = generate_attack_prompt_rows(
        data_frame,
        text_column=args.text_column,
        image_column=args.image_column,
        sample_id_column=args.sample_id_column,
        max_changed_words=args.max_changed_words,
        typoglycemia_factory=typoglycemia_factory,
    )
    write_attack_prompt_rows_csv(attack_rows, attack_rows_path)
    print(f"Wrote {attack_rows_path}")

    if args.scored_results:
        records, changed_words_by_key = load_scored_comparison_rows(args.scored_results)
        delta_rows, summaries = summarize_scored_comparison_rows(records, changed_words_by_key)
        delta_path, summary_path, report_path = _output_paths(input_path, str(output_prefix))
        write_delta_rows_csv(delta_rows, delta_path)
        write_comparison_summary_csv(summaries, summary_path)
        write_comparison_markdown_report(summaries, report_path)
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


def _resolve_dataframe_column(data_frame, explicit_name, candidates, role):
    if explicit_name:
        if explicit_name not in data_frame.columns:
            raise ValueError(f"{role} column {explicit_name!r} was not found")
        return explicit_name
    for candidate in candidates:
        if candidate in data_frame.columns:
            return candidate
    raise ValueError(
        f"could not infer {role} column from available columns: {', '.join(data_frame.columns)}"
    )


def _resolve_fieldname(fieldnames, candidates, role):
    field_by_normalized_name = {_normalize_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        match = field_by_normalized_name.get(_normalize_name(candidate))
        if match:
            return match
    raise ValueError(f"could not infer {role} column from available columns: {', '.join(fieldnames)}")


def _truncate(value, max_length=140):
    text = " ".join(str(value).split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _create_heuristic_typoglycemia():
    typoglycemia = Typoglycemia(seed=42, tokenizer=_simple_tokenizer)
    typoglycemia.pos_tagger = typoglycemia._heuristic_pos_tag
    return typoglycemia


def _simple_tokenizer(text):
    return str(text).replace(".", " .").split()


def _ensure_parent(path):
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
