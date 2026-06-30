"""Extract CLIP text-token hidden states for DF-Impact analysis."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_COLUMNS = ("Caption", "caption", "prompt", "text")
SAMPLE_ID_COLUMNS = ("sample_id", "image_id", "id", "File Path", "image_path", "file_path")
MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "position_ids")


@dataclass(frozen=True)
class ClipTokenHiddenState:
    sample_id: str
    prompt: str
    token_index: int
    token_id: int
    token: str
    start_char: int | None
    end_char: int | None
    attention_mask: int
    is_special_token: bool
    hidden_state: tuple[float, ...]


class ClipTextHiddenStateExtractor:
    """Run a CLIP text encoder and return one row per token."""

    def __init__(self, tokenizer, model, device=None, max_length=77):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length
        if device is not None and hasattr(model, "to"):
            self.model = model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_pretrained(cls, model_name=DEFAULT_MODEL_NAME, device=None, max_length=77):
        try:
            from transformers import AutoTokenizer, CLIPTextModel
        except ImportError as exc:
            raise ImportError(
                "Install transformers and torch to load a CLIP text model "
                "(for example: pip install transformers torch)."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = CLIPTextModel.from_pretrained(model_name)
        return cls(tokenizer=tokenizer, model=model, device=device, max_length=max_length)

    def encode(
        self,
        prompts,
        sample_ids=None,
        hidden_layer=-1,
        include_special_tokens=False,
    ):
        prompts = [str(prompt) for prompt in prompts]
        if sample_ids is None:
            sample_ids = [str(index) for index in range(len(prompts))]
        else:
            sample_ids = [str(sample_id) for sample_id in sample_ids]
        if len(sample_ids) != len(prompts):
            raise ValueError("sample_ids must have the same length as prompts")

        tokenized = tokenize_with_offsets(self.tokenizer, prompts, max_length=self.max_length)
        model_inputs = {
            key: value
            for key, value in tokenized.items()
            if key in MODEL_INPUT_KEYS
        }
        model_inputs = move_batch_to_device(model_inputs, self.device)
        outputs = run_text_model(self.model, model_inputs)
        selected_hidden_states = select_hidden_layer(outputs, hidden_layer)

        input_ids = to_python(tokenized["input_ids"])
        attention_masks = to_python(tokenized.get("attention_mask", default_attention_mask(input_ids)))
        offsets = to_python(tokenized.get("offset_mapping", default_offsets(input_ids)))
        special_masks = to_python(tokenized.get("special_tokens_mask", self._special_tokens_mask(input_ids)))
        hidden_rows = to_python(selected_hidden_states)

        records = []
        for prompt_index, prompt in enumerate(prompts):
            tokens = self._convert_ids_to_tokens(input_ids[prompt_index])
            for token_index, token_id in enumerate(input_ids[prompt_index]):
                attention_mask = int(attention_masks[prompt_index][token_index])
                is_special = bool(special_masks[prompt_index][token_index])
                if attention_mask == 0:
                    continue
                if is_special and not include_special_tokens:
                    continue

                start_char, end_char = normalize_offset(offsets[prompt_index][token_index])
                records.append(
                    ClipTokenHiddenState(
                        sample_id=sample_ids[prompt_index],
                        prompt=prompt,
                        token_index=token_index,
                        token_id=int(token_id),
                        token=str(tokens[token_index]),
                        start_char=start_char,
                        end_char=end_char,
                        attention_mask=attention_mask,
                        is_special_token=is_special,
                        hidden_state=tuple(float(value) for value in hidden_rows[prompt_index][token_index]),
                    )
                )
        return records

    def _convert_ids_to_tokens(self, token_ids):
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            return self.tokenizer.convert_ids_to_tokens(token_ids)
        return [str(token_id) for token_id in token_ids]

    def _special_tokens_mask(self, input_ids):
        masks = []
        for row in input_ids:
            if hasattr(self.tokenizer, "get_special_tokens_mask"):
                masks.append(self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True))
            else:
                masks.append([0 for _ in row])
        return masks


def tokenize_with_offsets(tokenizer, prompts, max_length=77):
    kwargs = {
        "padding": True,
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
        "return_special_tokens_mask": True,
    }
    try:
        return tokenizer(prompts, return_offsets_mapping=True, **kwargs)
    except (NotImplementedError, TypeError, ValueError):
        return tokenizer(prompts, **kwargs)


def run_text_model(model, model_inputs):
    try:
        import torch
    except ImportError:
        return model(**model_inputs, output_hidden_states=True, return_dict=True)

    with torch.inference_mode():
        return model(**model_inputs, output_hidden_states=True, return_dict=True)


def select_hidden_layer(outputs, hidden_layer):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    if hidden_states is None:
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None and isinstance(outputs, dict):
            last_hidden_state = outputs.get("last_hidden_state")
        if last_hidden_state is None:
            raise ValueError("model output did not include hidden states")
        return last_hidden_state
    return hidden_states[hidden_layer]


def load_prompt_rows(path, text_column=None, sample_id_column=None):
    rows = read_rows(Path(path))
    if not rows:
        raise ValueError("input file has no records")

    fieldnames = sorted({key for row in rows for key in row})
    text_column = resolve_column(fieldnames, text_column, TEXT_COLUMNS, "text")
    sample_id_column = resolve_column(
        fieldnames,
        sample_id_column,
        SAMPLE_ID_COLUMNS,
        "sample id",
        required=False,
    )

    prompts = []
    sample_ids = []
    for row_number, row in enumerate(rows, start=2):
        prompt = required_text(row, text_column, row_number)
        prompts.append(prompt)
        if sample_id_column is None:
            sample_ids.append(str(row_number - 2))
        else:
            sample_ids.append(required_text(row, sample_id_column, row_number))
    return sample_ids, prompts


def write_jsonl(records, path):
    path = Path(path)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_dict(record), ensure_ascii=False) + "\n")


def write_csv(records, path):
    path = Path(path)
    ensure_parent(path)
    fieldnames = [
        "sample_id",
        "prompt",
        "token_index",
        "token_id",
        "token",
        "start_char",
        "end_char",
        "attention_mask",
        "is_special_token",
        "hidden_state",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record_to_dict(record)
            row["hidden_state"] = json.dumps(row["hidden_state"])
            writer.writerow(row)


def record_to_dict(record):
    return {
        "sample_id": record.sample_id,
        "prompt": record.prompt,
        "token_index": record.token_index,
        "token_id": record.token_id,
        "token": record.token,
        "start_char": record.start_char,
        "end_char": record.end_char,
        "attention_mask": record.attention_mask,
        "is_special_token": record.is_special_token,
        "hidden_state": list(record.hidden_state),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract CLIP text-token IDs and hidden states for DF-Impact analysis."
    )
    parser.add_argument("input_path", help="CSV, TSV, JSON, or JSONL file containing prompts or captions.")
    parser.add_argument("--output-jsonl", help="Token-level JSONL output path.")
    parser.add_argument("--output-csv", help="Optional token-level CSV output path.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face CLIP text model name.")
    parser.add_argument("--text-column", help="Prompt/caption column name.")
    parser.add_argument("--sample-id-column", help="Sample id column name.")
    parser.add_argument("--device", help="Optional torch device, such as cuda or cpu.")
    parser.add_argument("--max-length", type=int, default=77, help="Maximum CLIP text token length.")
    parser.add_argument("--hidden-layer", type=int, default=-1, help="Hidden-state layer index to export.")
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Include BOS/EOS/pad-special token rows. Padding is still skipped.",
    )
    args = parser.parse_args(argv)

    sample_ids, prompts = load_prompt_rows(
        args.input_path,
        text_column=args.text_column,
        sample_id_column=args.sample_id_column,
    )
    extractor = ClipTextHiddenStateExtractor.from_pretrained(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
    )
    records = extractor.encode(
        prompts,
        sample_ids=sample_ids,
        hidden_layer=args.hidden_layer,
        include_special_tokens=args.include_special_tokens,
    )

    input_path = Path(args.input_path)
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else input_path.with_suffix("")
    if not args.output_jsonl:
        output_jsonl = Path(f"{output_jsonl}_clip_hidden_states.jsonl")
    write_jsonl(records, output_jsonl)
    print(f"Wrote {output_jsonl}")

    if args.output_csv:
        write_csv(records, args.output_csv)
        print(f"Wrote {args.output_csv}")
    return 0


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


def resolve_column(fieldnames, explicit_name, candidates, role, required=True):
    if explicit_name:
        if explicit_name not in fieldnames:
            raise ValueError(f"{role} column {explicit_name!r} was not found")
        return explicit_name

    field_by_normalized_name = {normalize_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        match = field_by_normalized_name.get(normalize_name(candidate))
        if match:
            return match

    if required:
        raise ValueError(f"could not infer {role} column from available columns: {', '.join(fieldnames)}")
    return None


def required_text(row, column, row_number):
    value = row.get(column)
    if value is None or str(value).strip() == "":
        raise ValueError(f"row {row_number} has an empty value for column {column!r}")
    return str(value).strip()


def normalize_name(value):
    return "".join(character for character in str(value).casefold() if character.isalnum())


def to_python(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def move_batch_to_device(batch, device):
    if device is None:
        return batch
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def default_attention_mask(input_ids):
    return [[1 for _ in row] for row in input_ids]


def default_offsets(input_ids):
    return [[None for _ in row] for row in input_ids]


def normalize_offset(offset):
    if offset is None:
        return None, None
    if len(offset) != 2:
        return None, None
    start_char, end_char = offset
    if start_char is None or end_char is None:
        return None, None
    if int(start_char) == 0 and int(end_char) == 0:
        return None, None
    return int(start_char), int(end_char)


def ensure_parent(path):
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
