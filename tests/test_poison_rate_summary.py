import csv
import json
import tempfile
import unittest
from pathlib import Path

from poison_rate_summary import (
    build_poison_rate_delta_rows,
    load_scored_prompt_rows,
    main,
    summarize_by_changed_words,
    summarize_by_poison_rate,
)


class PoisonRateSummaryTest(unittest.TestCase):
    def test_computes_poison_rate_from_changed_words_and_keeps_splits_separate(self):
        rows = [
            {
                "sample_id": "image-1",
                "split": "train",
                "method": "original",
                "prompt": "A person rides a small bicycle.",
                "changed_words": "0",
                "clip_score": "0.80",
            },
            {
                "sample_id": "image-1",
                "split": "train",
                "method": "typoglycemia",
                "prompt": "A psreon rides a small bcilyce.",
                "changed_words": "2",
                "clip_score": "0.62",
            },
            {
                "sample_id": "image-1",
                "split": "validation",
                "method": "original",
                "prompt": "A person rides a small bicycle.",
                "changed_words": "0",
                "clip_score": "0.77",
            },
            {
                "sample_id": "image-1",
                "split": "validation",
                "method": "typoglycemia",
                "prompt": "A psreon rides a small bicycle.",
                "changed_words": "1",
                "clip_score": "0.70",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "scores.csv"
            self._write_rows(input_path, rows)

            loaded_rows = load_scored_prompt_rows(input_path)
            delta_rows = build_poison_rate_delta_rows(loaded_rows)
            summaries = summarize_by_poison_rate(delta_rows)

        summary_by_split = {summary.split: summary for summary in summaries}
        self.assertAlmostEqual(summary_by_split["train"].poison_rate, 2 / 6, places=4)
        self.assertAlmostEqual(summary_by_split["train"].mean_clip_score_delta, -0.18)
        self.assertAlmostEqual(summary_by_split["validation"].poison_rate, 1 / 6, places=4)
        self.assertAlmostEqual(summary_by_split["validation"].mean_clip_score_delta, -0.07)

    def test_uses_explicit_poison_rate_for_rate_summary_and_changed_word_summary(self):
        rows = [
            {
                "sample_id": "image-1",
                "method": "original",
                "prompt": "A person rides a bicycle.",
                "changed_words": "0",
                "poison_rate": "",
                "clip_score": "0.50",
            },
            {
                "sample_id": "image-1",
                "method": "typoglycemia",
                "prompt": "A psreon rides a bicycle.",
                "changed_words": "1",
                "poison_rate": "0.2500",
                "clip_score": "0.45",
            },
            {
                "sample_id": "image-2",
                "method": "original",
                "prompt": "An artist paints murals.",
                "changed_words": "0",
                "poison_rate": "",
                "clip_score": "0.70",
            },
            {
                "sample_id": "image-2",
                "method": "typoglycemia",
                "prompt": "An atrist pinats murals.",
                "changed_words": "2",
                "poison_rate": "0.5000",
                "clip_score": "0.52",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "scores.csv"
            self._write_rows(input_path, rows)

            loaded_rows = load_scored_prompt_rows(input_path)
            delta_rows = build_poison_rate_delta_rows(loaded_rows)
            rate_summaries = summarize_by_poison_rate(delta_rows)
            changed_summaries = summarize_by_changed_words(delta_rows)

        self.assertEqual([summary.poison_rate for summary in rate_summaries], [0.25, 0.5])
        changed_by_count = {summary.changed_words: summary for summary in changed_summaries}
        self.assertAlmostEqual(changed_by_count[1].mean_poison_rate, 0.25)
        self.assertAlmostEqual(changed_by_count[2].mean_clip_score_delta, -0.18)

    def test_cli_writes_outputs_and_refuses_to_overwrite_without_flag(self):
        rows = [
            {
                "sample_id": "image-1",
                "split": "test",
                "method": "original",
                "prompt": "A person rides a bicycle.",
                "changed_words": "0",
                "clip_score": "0.50",
            },
            {
                "sample_id": "image-1",
                "split": "test",
                "method": "typoglycemia",
                "prompt": "A psreon rides a bicycle.",
                "changed_words": "1",
                "clip_score": "0.42",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "scores.csv"
            output_prefix = directory / "poison_rate"
            self._write_rows(input_path, rows)

            exit_code = main(
                [
                    str(input_path),
                    "--output-prefix",
                    str(output_prefix),
                    "--dataset",
                    "unit-test",
                    "--split-column",
                    "split",
                    "--attack-setting",
                    "typoglycemia-max-1",
                    "--model",
                    "mock-clip",
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(f"{output_prefix}_poison_rate_delta_rows.csv").exists())
            self.assertTrue(Path(f"{output_prefix}_poison_rate_summary.csv").exists())
            self.assertTrue(Path(f"{output_prefix}_changed_words_summary.csv").exists())
            self.assertTrue(Path(f"{output_prefix}_poison_rate_report.md").exists())
            metadata_path = Path(f"{output_prefix}_poison_rate_metadata.json")
            self.assertTrue(metadata_path.exists())
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["dataset"], "unit-test")
            self.assertEqual(metadata["splits"], ["test"])

            with self.assertRaises(FileExistsError):
                main([str(input_path), "--output-prefix", str(output_prefix)])

    def _write_rows(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
