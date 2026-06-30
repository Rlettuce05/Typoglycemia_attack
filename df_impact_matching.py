"""Align clean and poisoned CLIP token features for DF-Impact analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GAP_COST = 1.0
DEFAULT_MISMATCH_COST = 1.0
DEFAULT_HIDDEN_WEIGHT = 0.0
FEATURE_COLUMNS = (
    "sample_id",
    "prompt",
    "token_index",
    "token_id",
    "token",
    "start_char",
    "end_char",
    "hidden_state",
)
ALIGNMENT_COLUMNS = (
    "sample_id",
    "operation",
    "clean_token_index",
    "poisoned_token_index",
    "clean_token_id",
    "poisoned_token_id",
    "clean_token",
    "poisoned_token",
    "clean_start_char",
    "clean_end_char",
    "poisoned_start_char",
    "poisoned_end_char",
    "l2_distance",
    "cosine_distance",
    "step_cost",
)


@dataclass(frozen=True)
class TokenFeature:
    sample_id: str
    prompt: str
    token_index: int
    token_id: int
    token: str
    start_char: int | None
    end_char: int | None
    hidden_state: tuple[float, ...]


@dataclass(frozen=True)
class AlignmentCosts:
    gap: float = DEFAULT_GAP_COST
    mismatch: float = DEFAULT_MISMATCH_COST
    hidden_weight: float = DEFAULT_HIDDEN_WEIGHT


@dataclass(frozen=True)
class TokenAlignment:
    sample_id: str
    operation: str
    clean: TokenFeature | None
    poisoned: TokenFeature | None
    l2_distance: float | None
    cosine_distance: float | None
    step_cost: float


def align_feature_maps(clean_by_sample, poisoned_by_sample, costs=None, allow_unpaired=False):
    """Align token feature sequences for each sample id shared by two maps."""

    costs = costs or AlignmentCosts()
    clean_ids = set(clean_by_sample)
    poisoned_ids = set(poisoned_by_sample)
    missing_poisoned = clean_ids - poisoned_ids
    missing_clean = poisoned_ids - clean_ids
    if (missing_clean or missing_poisoned) and not allow_unpaired:
        missing_parts = []
        if missing_poisoned:
            missing_parts.append(f"missing poisoned features for: {format_sample_ids(missing_poisoned)}")
        if missing_clean:
            missing_parts.append(f"missing clean features for: {format_sample_ids(missing_clean)}")
        raise ValueError("; ".join(missing_parts))

    alignments = []
    for sample_id in sorted(clean_ids & poisoned_ids):
        clean_tokens = sorted(clean_by_sample[sample_id], key=lambda record: record.token_index)
        poisoned_tokens = sorted(poisoned_by_sample[sample_id], key=lambda record: record.token_index)
        alignments.extend(align_token_sequences(sample_id, clean_tokens, poisoned_tokens, costs))
    return alignments


def align_token_sequences(sample_id, clean_tokens, poisoned_tokens, costs=None):
    """Needleman-Wunsch style dynamic-programming alignment for token features."""

    costs = costs or AlignmentCosts()
    row_count = len(clean_tokens) + 1
    column_count = len(poisoned_tokens) + 1
    scores = [[0.0 for _ in range(column_count)] for _ in range(row_count)]
    backpointers = [[None for _ in range(column_count)] for _ in range(row_count)]

    for row in range(1, row_count):
        scores[row][0] = scores[row - 1][0] + costs.gap
        backpointers[row][0] = "delete"
    for column in range(1, column_count):
        scores[0][column] = scores[0][column - 1] + costs.gap
        backpointers[0][column] = "insert"

    for row in range(1, row_count):
        clean_token = clean_tokens[row - 1]
        for column in range(1, column_count):
            poisoned_token = poisoned_tokens[column - 1]
            pair_cost = replacement_cost(clean_token, poisoned_token, costs)
            candidates = [
                (scores[row - 1][column - 1] + pair_cost, "pair"),
                (scores[row - 1][column] + costs.gap, "delete"),
                (scores[row][column - 1] + costs.gap, "insert"),
            ]
            best_score, best_operation = min(candidates, key=lambda item: (item[0], operation_rank(item[1])))
            scores[row][column] = best_score
            backpointers[row][column] = best_operation

    return traceback_alignments(sample_id, clean_tokens, poisoned_tokens, scores, backpointers, costs)


def traceback_alignments(sample_id, clean_tokens, poisoned_tokens, scores, backpointers, costs):
    alignments = []
    row = len(clean_tokens)
    column = len(poisoned_tokens)
    while row > 0 or column > 0:
        operation = backpointers[row][column]
        if operation == "pair":
            clean = clean_tokens[row - 1]
            poisoned = poisoned_tokens[column - 1]
            label = "match" if normalized_token(clean.token) == normalized_token(poisoned.token) else "substitute"
            alignments.append(make_alignment(sample_id, label, clean, poisoned, replacement_cost(clean, poisoned, costs)))
            row -= 1
            column -= 1
        elif operation == "delete":
            alignments.append(make_alignment(sample_id, "delete", clean_tokens[row - 1], None, costs.gap))
            row -= 1
        elif operation == "insert":
            alignments.append(make_alignment(sample_id, "insert", None, poisoned_tokens[column - 1], costs.gap))
            column -= 1
        else:
            raise ValueError(f"alignment traceback stopped at row={row}, column={column}, score={scores[row][column]}")

    alignments.reverse()
    return alignments


def replacement_cost(clean, poisoned, costs):
    token_cost = 0.0 if normalized_token(clean.token) == normalized_token(poisoned.token) else costs.mismatch
    hidden_distance = l2_distance(clean.hidden_state, poisoned.hidden_state)
    if hidden_distance is None:
        return token_cost
    return token_cost + costs.hidden_weight * hidden_distance


def make_alignment(sample_id, operation, clean, poisoned, step_cost):
    if clean is not None and poisoned is not None:
        l2_value = l2_distance(clean.hidden_state, poisoned.hidden_state)
        cosine_value = cosine_distance(clean.hidden_state, poisoned.hidden_state)
    else:
        l2_value = None
        cosine_value = None
    return TokenAlignment(
        sample_id=sample_id,
        operation=operation,
        clean=clean,
        poisoned=poisoned,
        l2_distance=l2_value,
        cosine_distance=cosine_value,
        step_cost=step_cost,
    )


def read_feature_map(path):
    rows = read_rows(Path(path))
    feature_map = {}
    for row_number, row in enumerate(rows, start=1):
        feature = parse_feature_row(row, row_number)
        feature_map.setdefault(feature.sample_id, []).append(feature)
    return feature_map


def read_rows(path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON feature input must be a list of row objects")
        return data

    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def parse_feature_row(row, row_number):
    missing = [column for column in FEATURE_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"feature row {row_number} is missing columns: {', '.join(missing)}")
    return TokenFeature(
        sample_id=required_text(row, "sample_id", row_number),
        prompt=str(row.get("prompt", "")),
        token_index=parse_int(row["token_index"], "token_index", row_number),
        token_id=parse_int(row["token_id"], "token_id", row_number),
        token=required_text(row, "token", row_number),
        start_char=parse_optional_int(row.get("start_char"), "start_char", row_number),
        end_char=parse_optional_int(row.get("end_char"), "end_char", row_number),
        hidden_state=parse_hidden_state(row["hidden_state"], row_number),
    )


def write_alignment_jsonl(alignments, path):
    path = Path(path)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for alignment in alignments:
            handle.write(json.dumps(alignment_to_dict(alignment), ensure_ascii=False) + "\n")


def write_alignment_csv(alignments, path):
    path = Path(path)
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALIGNMENT_COLUMNS)
        writer.writeheader()
        for alignment in alignments:
            writer.writerow(alignment_to_flat_dict(alignment))


def alignment_to_dict(alignment):
    return {
        "sample_id": alignment.sample_id,
        "operation": alignment.operation,
        "clean": feature_to_dict(alignment.clean),
        "poisoned": feature_to_dict(alignment.poisoned),
        "l2_distance": alignment.l2_distance,
        "cosine_distance": alignment.cosine_distance,
        "step_cost": alignment.step_cost,
    }


def alignment_to_flat_dict(alignment):
    clean = alignment.clean
    poisoned = alignment.poisoned
    return {
        "sample_id": alignment.sample_id,
        "operation": alignment.operation,
        "clean_token_index": value_or_empty(clean.token_index if clean else None),
        "poisoned_token_index": value_or_empty(poisoned.token_index if poisoned else None),
        "clean_token_id": value_or_empty(clean.token_id if clean else None),
        "poisoned_token_id": value_or_empty(poisoned.token_id if poisoned else None),
        "clean_token": clean.token if clean else "",
        "poisoned_token": poisoned.token if poisoned else "",
        "clean_start_char": value_or_empty(clean.start_char if clean else None),
        "clean_end_char": value_or_empty(clean.end_char if clean else None),
        "poisoned_start_char": value_or_empty(poisoned.start_char if poisoned else None),
        "poisoned_end_char": value_or_empty(poisoned.end_char if poisoned else None),
        "l2_distance": value_or_empty(alignment.l2_distance),
        "cosine_distance": value_or_empty(alignment.cosine_distance),
        "step_cost": alignment.step_cost,
    }


def feature_to_dict(feature):
    if feature is None:
        return None
    return {
        "sample_id": feature.sample_id,
        "prompt": feature.prompt,
        "token_index": feature.token_index,
        "token_id": feature.token_id,
        "token": feature.token,
        "start_char": feature.start_char,
        "end_char": feature.end_char,
        "hidden_state": list(feature.hidden_state),
    }


def l2_distance(left, right):
    if len(left) != len(right):
        return None
    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def cosine_distance(left, right):
    if len(left) != len(right):
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    similarity = sum(left_value * right_value for left_value, right_value in zip(left, right)) / (left_norm * right_norm)
    return 1.0 - max(-1.0, min(1.0, similarity))


def normalized_token(token):
    value = str(token).strip().casefold()
    for prefix in ("##", chr(0x0120)):
        if value.startswith(prefix):
            value = value[len(prefix):]
    if value.endswith("</w>"):
        value = value[:-4]
    return value


def operation_rank(operation):
    return {"pair": 0, "delete": 1, "insert": 2}[operation]


def parse_hidden_state(value, row_number):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return tuple()
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"feature row {row_number} has invalid hidden_state JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"feature row {row_number} hidden_state must be a JSON list")
    return tuple(float(item) for item in value)


def required_text(row, column, row_number):
    value = row.get(column)
    if value is None or str(value).strip() == "":
        raise ValueError(f"feature row {row_number} has an empty {column!r}")
    return str(value).strip()


def parse_int(value, column, row_number):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature row {row_number} has invalid integer {column!r}: {value!r}") from exc


def parse_optional_int(value, column, row_number):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature row {row_number} has invalid integer {column!r}: {value!r}") from exc


def value_or_empty(value):
    return "" if value is None else value


def format_sample_ids(sample_ids):
    return ", ".join(str(sample_id) for sample_id in sorted(sample_ids))


def ensure_parent(path):
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DP-align clean and poisoned CLIP token features for DF-Impact analysis."
    )
    parser.add_argument("clean_features", help="Clean prompt token features from df_impact_features.py.")
    parser.add_argument("poisoned_features", help="Poisoned prompt token features from df_impact_features.py.")
    parser.add_argument("--output-csv", required=True, help="Token alignment CSV output path.")
    parser.add_argument("--output-jsonl", help="Optional token alignment JSONL output path.")
    parser.add_argument("--gap-cost", type=float, default=DEFAULT_GAP_COST, help="Insertion/deletion DP cost.")
    parser.add_argument("--mismatch-cost", type=float, default=DEFAULT_MISMATCH_COST, help="Token substitution DP cost.")
    parser.add_argument(
        "--hidden-weight",
        type=float,
        default=DEFAULT_HIDDEN_WEIGHT,
        help="Optional hidden-state L2 contribution to pair costs. Defaults to token-only matching.",
    )
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Skip sample ids that are present in only one input file.",
    )
    args = parser.parse_args(argv)

    costs = AlignmentCosts(gap=args.gap_cost, mismatch=args.mismatch_cost, hidden_weight=args.hidden_weight)
    clean_by_sample = read_feature_map(args.clean_features)
    poisoned_by_sample = read_feature_map(args.poisoned_features)
    alignments = align_feature_maps(
        clean_by_sample,
        poisoned_by_sample,
        costs=costs,
        allow_unpaired=args.allow_unpaired,
    )
    write_alignment_csv(alignments, args.output_csv)
    print(f"Wrote {args.output_csv}")
    if args.output_jsonl:
        write_alignment_jsonl(alignments, args.output_jsonl)
        print(f"Wrote {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
