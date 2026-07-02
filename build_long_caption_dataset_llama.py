"""Build a long-caption JSONL dataset with a Llama Vision model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_CLIP_TOKENIZER = "openai/clip-vit-base-patch32"
IMAGE_COLUMNS = ("image_path", "File Path", "file_path", "image", "path")
CAPTION_COLUMNS = ("caption", "Caption", "original_caption", "prompt", "text")
SAMPLE_ID_COLUMNS = ("sample_id", "image_id", "id", "image_path", "File Path", "file_path")
SPLIT_COLUMNS = ("split", "dataset_split", "data_split")
WORD_RE = re.compile(r"[A-Za-z]+")

PROMPT_TEMPLATE = """Given the image and the original caption, write one detailed English caption.

Rules:
- Describe only visible content in the image.
- Preserve the meaning of the original caption.
- Add concrete visible details such as objects, attributes, actions, colors, spatial relations, and scene context.
- Do not infer invisible information such as names, identities, emotions, exact location, date, or background story.
- Do not add objects that are not visible.
- Do not use bullet points.
- Do not intentionally misspell words.
- Write a single natural sentence.
- The caption should be between {min_words} and {max_words} words.

Original caption:
{original_caption}
"""


@dataclass(frozen=True)
class SourceCaptionRow:
    row_number: int
    sample_id: str
    image_path: str
    original_caption: str
    split: str


@dataclass(frozen=True)
class BuilderContext:
    dataset: str
    attack_setting: str
    model: str
    seed: str
    config: str
    prompt_template: str
    min_words: int
    max_words: int
    clip_token_limit: int
    created_at: str
    git_commit: str


class DryRunCaptionGenerator:
    """Deterministic plumbing check that does not produce research data."""

    model_id = "dry-run-template"

    def generate(self, image_path, original_caption):
        caption = str(original_caption).strip().rstrip(".")
        return (
            f"{caption} with visible objects, colors, actions, spatial relations, "
            "and surrounding scene details described in one sentence."
        )


class LlamaVisionCaptionGenerator:
    def __init__(
        self,
        model_id=DEFAULT_MODEL_ID,
        device="auto",
        torch_dtype="auto",
        max_new_tokens=96,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        image_root=None,
        prompt_template=PROMPT_TEMPLATE,
        min_words=40,
        max_words=60,
    ):
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor
            try:
                from transformers import MllamaForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as ModelClass
        except ImportError as exc:
            raise RuntimeError(
                "Llama Vision generation requires pillow, torch, and transformers. "
                "Install them or run with --dry-run for a plumbing check."
            ) from exc

        self.Image = Image
        self.torch = torch
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.image_root = Path(image_root) if image_root else None
        self.prompt_template = prompt_template
        self.min_words = min_words
        self.max_words = max_words
        self.processor = AutoProcessor.from_pretrained(model_id)

        model_kwargs = {}
        dtype = self._resolve_torch_dtype(torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = ModelClass.from_pretrained(model_id, **model_kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()

    def generate(self, image_path, original_caption):
        prompt = render_prompt(
            original_caption,
            min_words=self.min_words,
            max_words=self.max_words,
            prompt_template=self.prompt_template,
        )
        image = self.Image.open(self._resolve_image_path(image_path)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            input_text = prompt

        inputs = self.processor(images=image, text=input_text, return_tensors="pt")
        if self.device != "auto":
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation_kwargs.update({"temperature": self.temperature, "top_p": self.top_p})

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_length:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _resolve_image_path(self, image_path):
        path = Path(str(image_path))
        if not path.is_absolute() and self.image_root:
            path = self.image_root / path
        return path

    def _resolve_torch_dtype(self, torch_dtype):
        if torch_dtype in (None, "", "auto"):
            return None
        dtype = getattr(self.torch, torch_dtype, None)
        if dtype is None:
            raise ValueError(f"unknown torch dtype: {torch_dtype}")
        return dtype


class ClipTokenCounter:
    def __init__(self, tokenizer, token_limit=77):
        self.tokenizer = tokenizer
        self.token_limit = token_limit

    @classmethod
    def from_pretrained(cls, tokenizer_name=DEFAULT_CLIP_TOKENIZER, token_limit=77):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "CLIP token counting requires transformers. Pass --skip-clip-tokenizer "
                "to record null token lengths."
            ) from exc
        return cls(AutoTokenizer.from_pretrained(tokenizer_name), token_limit=token_limit)

    def count(self, text):
        encoded = self.tokenizer(str(text), add_special_tokens=True, truncation=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)


def render_prompt(original_caption, min_words=40, max_words=60, prompt_template=PROMPT_TEMPLATE):
    return prompt_template.format(
        original_caption=str(original_caption).strip(),
        min_words=min_words,
        max_words=max_words,
    )


def read_input_rows(path):
    path = Path(path)
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


def load_source_rows(
    input_path,
    image_column=None,
    caption_column=None,
    sample_id_column=None,
    split_column=None,
    split=None,
    limit=None,
):
    rows = read_input_rows(input_path)
    if not rows:
        raise ValueError("input file has no records")

    fieldnames = sorted({key for row in rows for key in row})
    image_column = resolve_column(fieldnames, image_column, IMAGE_COLUMNS, "image")
    caption_column = resolve_column(fieldnames, caption_column, CAPTION_COLUMNS, "caption")
    sample_id_column = resolve_optional_column(fieldnames, sample_id_column, SAMPLE_ID_COLUMNS, "sample id")
    split_column = resolve_optional_column(fieldnames, split_column, SPLIT_COLUMNS, "split")

    source_rows = []
    for row_number, row in enumerate(rows, start=2):
        if limit is not None and len(source_rows) >= limit:
            break
        image_path = required_text(row, image_column, row_number)
        original_caption = required_text(row, caption_column, row_number)
        sample_id = required_text(row, sample_id_column, row_number) if sample_id_column else image_path
        row_split = split or (optional_text(row, split_column) if split_column else "unspecified")
        source_rows.append(
            SourceCaptionRow(
                row_number=row_number,
                sample_id=sample_id,
                image_path=image_path,
                original_caption=original_caption,
                split=row_split,
            )
        )
    return source_rows


def build_output_row(
    source_row,
    long_caption,
    context,
    clip_counter=None,
    status="ok",
    error=None,
):
    clean_caption = clean_generated_caption(long_caption)
    original_word_count = count_words(source_row.original_caption)
    long_word_count = count_words(clean_caption)
    original_eligible_count = count_typoglycemia_eligible_words(source_row.original_caption)
    long_eligible_count = count_typoglycemia_eligible_words(clean_caption)
    clip_token_length = clip_counter.count(clean_caption) if clip_counter and clean_caption else None
    clip_overflow = clip_token_length > context.clip_token_limit if clip_token_length is not None else None

    return {
        "sample_id": source_row.sample_id,
        "image_path": source_row.image_path,
        "original_caption": source_row.original_caption,
        "long_caption": clean_caption,
        "original_word_count": original_word_count,
        "original_eligible_word_count": original_eligible_count,
        "long_word_count": long_word_count,
        "long_eligible_word_count": long_eligible_count,
        "eligible_word_gain": long_eligible_count - original_eligible_count,
        "clip_token_length": clip_token_length,
        "clip_overflow": clip_overflow,
        "status": status,
        "error": error,
        "dataset": context.dataset,
        "split": source_row.split,
        "attack_setting": context.attack_setting,
        "model": context.model,
        "seed": context.seed,
        "config": context.config,
        "created_at": context.created_at,
        "git_commit": context.git_commit,
        "source_row_number": source_row.row_number,
        "prompt_template": context.prompt_template,
        "quality_flags": {
            "eligible_word_count_increased": long_eligible_count > original_eligible_count,
            "word_count_in_range": context.min_words <= long_word_count <= context.max_words,
            "clip_within_limit": not clip_overflow if clip_overflow is not None else None,
        },
    }


def run_builder(
    source_rows,
    output_jsonl,
    generator,
    context,
    clip_counter=None,
    resume=False,
):
    output_jsonl = Path(output_jsonl)
    ensure_parent(output_jsonl)
    processed_keys = load_processed_keys(output_jsonl) if resume and output_jsonl.exists() else set()
    mode = "a" if resume and output_jsonl.exists() else "w"
    summary = {"written": 0, "skipped": 0, "errors": 0}
    new_rows = []

    with output_jsonl.open(mode, encoding="utf-8", newline="\n") as handle:
        for source_row in source_rows:
            key = processed_key(source_row)
            if key in processed_keys:
                summary["skipped"] += 1
                continue
            try:
                generated_caption = generator.generate(source_row.image_path, source_row.original_caption)
                status = "dry_run" if isinstance(generator, DryRunCaptionGenerator) else "ok"
                row = build_output_row(
                    source_row,
                    generated_caption,
                    context,
                    clip_counter=clip_counter,
                    status=status,
                )
                if not row["long_caption"]:
                    raise ValueError("generated caption is empty after cleanup")
            except Exception as exc:  # noqa: BLE001 - record row-level errors and continue the dataset pass.
                summary["errors"] += 1
                row = build_output_row(
                    source_row,
                    "",
                    context,
                    clip_counter=clip_counter,
                    status="error",
                    error=str(exc),
                )

            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            new_rows.append(row)
            processed_keys.add(key)
            summary["written"] += 1
    return summary, new_rows


def write_review_samples(rows, output_path, sample_size, seed=42):
    if sample_size <= 0 or not output_path:
        return []
    sample_size = min(sample_size, len(rows))
    sampled = random.Random(seed).sample(rows, sample_size) if sample_size else []
    output_path = Path(output_path)
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return sampled


def ensure_outputs_available(paths, overwrite=False):
    if overwrite:
        return
    existing = [str(path) for path in paths if path and Path(path).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing output files; pass --overwrite or --resume: "
            + ", ".join(existing)
        )


def load_processed_keys(output_jsonl):
    keys = set()
    with Path(output_jsonl).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(
                (
                    str(row.get("sample_id", "")),
                    str(row.get("image_path", "")),
                    str(row.get("original_caption", "")),
                )
            )
    return keys


def processed_key(source_row):
    return (source_row.sample_id, source_row.image_path, source_row.original_caption)


def clean_generated_caption(text):
    text = str(text or "").strip()
    for _ in range(2):
        text = strip_surrounding_quotes(text)
        text = re.sub(r"^[\-*\d.)\s]+", "", text).strip()
        text = re.sub(r"^(caption|long caption|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return strip_surrounding_quotes(text)


def strip_surrounding_quotes(text):
    text = str(text).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        text = text[1:-1].strip()
    return text


def count_words(text):
    return len(WORD_RE.findall(str(text)))


def count_typoglycemia_eligible_words(text):
    return sum(1 for word in WORD_RE.findall(str(text)) if is_typoglycemia_eligible_word(word))


def is_typoglycemia_eligible_word(word):
    word = str(word)
    if not word.isalpha() or len(word) <= 3:
        return False
    middle = word[1:-1].lower()
    return len(set(middle)) > 1


def resolve_column(fieldnames, explicit_name, candidates, role):
    if explicit_name:
        if explicit_name not in fieldnames:
            raise ValueError(f"{role} column {explicit_name!r} was not found")
        return explicit_name
    match = resolve_optional_column(fieldnames, None, candidates, role)
    if match:
        return match
    raise ValueError(f"could not infer {role} column from available columns: {', '.join(fieldnames)}")


def resolve_optional_column(fieldnames, explicit_name, candidates, role):
    if explicit_name:
        return resolve_column(fieldnames, explicit_name, candidates, role)
    field_by_normalized_name = {normalize_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        match = field_by_normalized_name.get(normalize_name(candidate))
        if match:
            return match
    return None


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


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


def ensure_parent(path):
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


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


def build_context(args):
    prompt_template = load_prompt_template(args.prompt_template_file)
    return BuilderContext(
        dataset=args.dataset or "unspecified",
        attack_setting=args.attack_setting or "long-caption-generation",
        model=DryRunCaptionGenerator.model_id if args.dry_run else args.model_id,
        seed=str(args.seed) if args.seed is not None else "unspecified",
        config=args.config or "unspecified",
        prompt_template=prompt_template,
        min_words=args.min_words,
        max_words=args.max_words,
        clip_token_limit=args.clip_token_limit,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit_hash(),
    )


def build_generator(args):
    prompt_template = load_prompt_template(args.prompt_template_file)
    if args.dry_run:
        return DryRunCaptionGenerator()
    return LlamaVisionCaptionGenerator(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        image_root=args.image_root,
        prompt_template=prompt_template,
        min_words=args.min_words,
        max_words=args.max_words,
    )


def load_prompt_template(prompt_template_file):
    if not prompt_template_file:
        return PROMPT_TEMPLATE
    return Path(prompt_template_file).read_text(encoding="utf-8")


def build_clip_counter(args):
    if args.skip_clip_tokenizer:
        return None
    return ClipTokenCounter.from_pretrained(args.clip_tokenizer, token_limit=args.clip_token_limit)


def default_review_sample_path(output_jsonl):
    path = Path(output_jsonl)
    return path.with_name(f"{path.stem}_review_samples.jsonl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate long image captions and write reviewable JSONL rows for Typoglycemia experiments."
    )
    parser.add_argument("input_path", help="CSV, TSV, JSON, or JSONL with image paths and original captions.")
    parser.add_argument("--output-jsonl", required=True, help="Destination JSONL path.")
    parser.add_argument("--image-column", help="Column containing image paths.")
    parser.add_argument("--caption-column", help="Column containing original captions.")
    parser.add_argument("--sample-id-column", help="Column identifying samples. Defaults to image path.")
    parser.add_argument("--split-column", help="Optional dataset split column.")
    parser.add_argument("--split", help="Fixed split value to record for every output row.")
    parser.add_argument("--dataset", help="Dataset name recorded in each output row.")
    parser.add_argument("--config", help="Config path or short identifier recorded in each output row.")
    parser.add_argument("--attack-setting", help="Attack setting recorded in each output row.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face Llama Vision model id.")
    parser.add_argument("--image-root", help="Root directory used to resolve relative image paths for generation.")
    parser.add_argument("--device", default="auto", help="Torch device, or auto for device_map='auto'.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype name such as float16, bfloat16, or auto.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument("--prompt-template-file", help="Optional prompt template file with {original_caption}.")
    parser.add_argument("--clip-tokenizer", default=DEFAULT_CLIP_TOKENIZER)
    parser.add_argument("--clip-token-limit", type=int, default=77)
    parser.add_argument("--skip-clip-tokenizer", action="store_true", help="Record null CLIP token lengths.")
    parser.add_argument("--limit", type=int, help="Maximum number of input rows to process.")
    parser.add_argument("--resume", action="store_true", help="Append and skip pairs already present in output JSONL.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--review-samples", type=int, default=0, help="Number of generated rows to sample for review.")
    parser.add_argument("--review-sample-jsonl", help="Destination for sampled review rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a deterministic placeholder captioner for plumbing checks only.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    review_path = args.review_sample_jsonl
    if args.review_samples and not review_path:
        review_path = default_review_sample_path(args.output_jsonl)
    if not args.resume:
        ensure_outputs_available([Path(args.output_jsonl)], overwrite=args.overwrite)
    ensure_outputs_available([Path(review_path) if review_path else None], overwrite=args.overwrite)

    source_rows = load_source_rows(
        args.input_path,
        image_column=args.image_column,
        caption_column=args.caption_column,
        sample_id_column=args.sample_id_column,
        split_column=args.split_column,
        split=args.split,
        limit=args.limit,
    )
    context = build_context(args)
    generator = build_generator(args)
    clip_counter = build_clip_counter(args)
    summary, new_rows = run_builder(
        source_rows,
        args.output_jsonl,
        generator,
        context,
        clip_counter=clip_counter,
        resume=args.resume,
    )
    sampled_rows = write_review_samples(new_rows, review_path, args.review_samples, seed=args.seed)

    print(
        "Wrote {written} rows to {output} ({errors} errors, {skipped} skipped).".format(
            output=args.output_jsonl,
            **summary,
        )
    )
    if sampled_rows:
        print(f"Wrote {len(sampled_rows)} review samples to {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
