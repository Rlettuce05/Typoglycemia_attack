import csv
import tempfile
import unittest
from pathlib import Path

from clip_score_summary import (
    build_clip_delta_rows,
    load_clip_score_records,
    main,
    render_markdown_report,
    summarize_clip_delta_rows,
)


class ClipScoreSummaryTest(unittest.TestCase):
    def test_summarizes_paired_clip_score_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "clip_scores.csv"
            self._write_rows(input_path)

            records = load_clip_score_records(input_path)
            delta_rows = build_clip_delta_rows(records)
            summaries = summarize_clip_delta_rows(delta_rows)

            self.assertEqual(len(delta_rows), 4)
            summary_by_method = {summary.method: summary for summary in summaries}
            self.assertEqual(summary_by_method["typoglycemia"].count, 2)
            self.assertAlmostEqual(summary_by_method["typoglycemia"].mean_delta, -0.155)
            self.assertAlmostEqual(summary_by_method["typoglycemia"].variance_delta, 0.003025)
            self.assertEqual(summary_by_method["typoglycemia"].representative_sample_id, "image-1")

            report = render_markdown_report(delta_rows, summaries)

            self.assertIn("Paper Table Candidate", report)
            self.assertIn("typoglycemia", report)
            self.assertIn("Largest average CLIPScore decrease", report)

    def test_cli_writes_delta_summary_and_report_files(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "clip_scores.tsv"
            output_prefix = Path(directory) / "chapter4"
            self._write_rows(input_path, delimiter="\t")

            exit_code = main([str(input_path), "--output-prefix", str(output_prefix)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(f"{output_prefix}_clip_delta_rows.csv").exists())
            self.assertTrue(Path(f"{output_prefix}_clip_summary.csv").exists())
            self.assertTrue(Path(f"{output_prefix}_clip_report.md").exists())

    def test_missing_original_rows_fail_with_clear_message(self):
        records = [
            {
                "sample_id": "image-1",
                "method": "typoglycemia",
                "prompt": "A preson rides a bicycle.",
                "clip_score": "0.12",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "clip_scores.csv"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)

            loaded_records = load_clip_score_records(input_path)

        with self.assertRaisesRegex(ValueError, "no original rows found"):
            build_clip_delta_rows(loaded_records)

    def _write_rows(self, path, delimiter=","):
        rows = [
            {
                "sample_id": "image-1",
                "method": "original",
                "prompt": "A person rides a bicycle.",
                "clip_score": "0.31",
            },
            {
                "sample_id": "image-1",
                "method": "typoglycemia",
                "prompt": "A preson rides a bicycle.",
                "clip_score": "0.10",
            },
            {
                "sample_id": "image-1",
                "method": "char_delete",
                "prompt": "A prson rides a bicycle.",
                "clip_score": "0.20",
            },
            {
                "sample_id": "image-2",
                "method": "original",
                "prompt": "An artist paints murals.",
                "clip_score": "0.50",
            },
            {
                "sample_id": "image-2",
                "method": "typoglycemia",
                "prompt": "An atrist paints murals.",
                "clip_score": "0.40",
            },
            {
                "sample_id": "image-2",
                "method": "char_delete",
                "prompt": "An artist paints muras.",
                "clip_score": "0.55",
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
