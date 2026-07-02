import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_long_caption_dataset_llama import (
    BuilderContext,
    build_output_row,
    clean_generated_caption,
    count_typoglycemia_eligible_words,
    load_source_rows,
    main,
    run_builder,
)


class FakeCaptionGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, image_path, original_caption):
        self.calls.append((image_path, original_caption))
        return (
            "Caption: A bright red bicycle rests beside a crowded city sidewalk "
            "while a person in a blue jacket looks toward nearby storefront windows."
        )


class FakeClipCounter:
    token_limit = 77

    def count(self, text):
        return len(str(text).split()) + 2


class LongCaptionDatasetBuilderTest(unittest.TestCase):
    def test_builds_output_row_with_metadata_counts_and_quality_flags(self):
        source_row = load_source_rows(self._write_input_rows([self._row("img-1.jpg", "A person rides a bike.")]))[0]
        context = self._context()

        row = build_output_row(
            source_row,
            (
                '"Caption: A careful rider moves along a sunny street on a small '
                'bicycle near parked cars and storefront windows."'
            ),
            context,
            clip_counter=FakeClipCounter(),
        )

        self.assertEqual(row["sample_id"], "img-1.jpg")
        self.assertEqual(row["original_caption"], "A person rides a bike.")
        self.assertTrue(row["long_caption"].startswith("A careful rider"))
        self.assertGreater(row["long_eligible_word_count"], row["original_eligible_word_count"])
        self.assertEqual(row["clip_overflow"], False)
        self.assertEqual(row["dataset"], "unit-test")
        self.assertEqual(row["split"], "train")
        self.assertEqual(row["model"], "fake-llama")
        self.assertIn("eligible_word_count_increased", row["quality_flags"])

    def test_resume_skips_existing_processed_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = self._write_input_rows(
                [
                    self._row("img-1.jpg", "A person rides a bicycle."),
                    self._row("img-2.jpg", "A dog sits near a window."),
                ],
                directory=directory,
            )
            output_path = directory / "long.jsonl"
            context = self._context()
            first_row = build_output_row(
                load_source_rows(input_path, limit=1)[0],
                "A person rides a bicycle near a red wall and a narrow street.",
                context,
            )
            output_path.write_text(json.dumps(first_row) + "\n", encoding="utf-8")

            generator = FakeCaptionGenerator()
            summary, rows = run_builder(
                load_source_rows(input_path),
                output_path,
                generator,
                context,
                resume=True,
            )

            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(summary["written"], 1)
            self.assertEqual(generator.calls, [("img-2.jpg", "A dog sits near a window.")])
            self.assertEqual(rows[0]["image_path"], "img-2.jpg")

    def test_cli_dry_run_writes_jsonl_review_sample_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = self._write_input_rows(
                [self._row("img-1.jpg", "A person rides a bicycle.")],
                directory=directory,
            )
            output_path = directory / "long.jsonl"
            review_path = directory / "review.jsonl"

            exit_code = main(
                [
                    str(input_path),
                    "--output-jsonl",
                    str(output_path),
                    "--dataset",
                    "unit-test",
                    "--split-column",
                    "split",
                    "--dry-run",
                    "--skip-clip-tokenizer",
                    "--review-samples",
                    "1",
                    "--review-sample-jsonl",
                    str(review_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            output_rows = self._read_jsonl(output_path)
            review_rows = self._read_jsonl(review_path)
            self.assertEqual(len(output_rows), 1)
            self.assertEqual(output_rows[0]["status"], "dry_run")
            self.assertEqual(output_rows[0]["clip_token_length"], None)
            self.assertEqual(len(review_rows), 1)

            with self.assertRaises(FileExistsError):
                main(
                    [
                        str(input_path),
                        "--output-jsonl",
                        str(output_path),
                        "--dry-run",
                        "--skip-clip-tokenizer",
                    ]
                )

    def test_clean_generated_caption_removes_common_wrappers(self):
        self.assertEqual(
            clean_generated_caption('Caption: "A person walks near a red building."\n'),
            "A person walks near a red building.",
        )
        self.assertEqual(
            clean_generated_caption("- Long caption: A person walks outside."),
            "A person walks outside.",
        )

    def test_counts_only_typoglycemia_transformable_words(self):
        self.assertEqual(count_typoglycemia_eligible_words("person bicycle street moon"), 3)

    def _context(self):
        return BuilderContext(
            dataset="unit-test",
            attack_setting="long-caption-generation",
            model="fake-llama",
            seed="7",
            config="unit",
            prompt_template="template",
            min_words=5,
            max_words=30,
            clip_token_limit=77,
            created_at="2026-07-02T00:00:00+00:00",
            git_commit="abc123",
        )

    def _row(self, image_path, caption):
        return {
            "image_path": image_path,
            "caption": caption,
            "split": "train",
        }

    def _write_input_rows(self, rows, directory=None):
        if directory is None:
            directory = Path(tempfile.mkdtemp())
        path = Path(directory) / "captions.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "caption", "split"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _read_jsonl(self, path):
        with Path(path).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    unittest.main()
