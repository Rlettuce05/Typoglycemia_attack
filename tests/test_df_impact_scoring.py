import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from df_impact_scoring import (
    AlignmentStep,
    build_word_impact_rows,
    load_alignment_steps,
    main,
    summarize_word_impacts,
)


def step(sample_id, operation, clean_token, poisoned_token, clean_prompt, poisoned_prompt, start, end, l2):
    return AlignmentStep(
        sample_id=sample_id,
        operation=operation,
        clean_token=clean_token,
        poisoned_token=poisoned_token,
        clean_prompt=clean_prompt,
        poisoned_prompt=poisoned_prompt,
        clean_start_char=start,
        clean_end_char=end,
        poisoned_start_char=start,
        poisoned_end_char=end,
        l2_distance=l2,
        cosine_distance=0.25 if l2 is not None else None,
        step_cost=1.0 if operation != "match" else 0.0,
    )


class DfImpactScoringTest(unittest.TestCase):
    def test_aggregates_alignment_steps_by_word_and_computes_df_impact(self):
        steps = [
            step("image-1", "substitute", "cat</w>", "cta</w>", "A cat sits", "A cta sits", 2, 5, 2.0),
            step("image-1", "match", "sits</w>", "sits</w>", "A cat sits", "A cta sits", 6, 10, 0.1),
            step("image-2", "substitute", "cat</w>", "cta</w>", "A cat jumps", "A cta jumps", 2, 5, 4.0),
        ]

        rows = build_word_impact_rows(steps)
        summaries = summarize_word_impacts(rows)

        cat_rows = [row for row in rows if row.word == "cat"]
        self.assertEqual(len(cat_rows), 2)
        self.assertEqual(cat_rows[0].clean_word, "cat")
        self.assertEqual(cat_rows[0].poisoned_word, "cta")
        self.assertEqual(cat_rows[0].operation, "substitute")
        self.assertAlmostEqual(cat_rows[0].impact_score, 2.0)
        self.assertAlmostEqual(cat_rows[0].df_score, math.log1p(2))

        cat_summary = {summary.word: summary for summary in summaries}["cat"]
        self.assertEqual(cat_summary.sample_count, 2)
        self.assertAlmostEqual(cat_summary.mean_impact_score, 3.0)
        self.assertAlmostEqual(cat_summary.max_df_impact_score, 4.0 * math.log1p(2))

    def test_uses_explicit_df_scores_and_step_cost_for_insertions(self):
        insert = AlignmentStep(
            sample_id="image-1",
            operation="insert",
            clean_token="",
            poisoned_token="small</w>",
            clean_prompt="A cat",
            poisoned_prompt="A small cat",
            clean_start_char=None,
            clean_end_char=None,
            poisoned_start_char=2,
            poisoned_end_char=7,
            l2_distance=None,
            cosine_distance=None,
            step_cost=1.5,
        )

        rows = build_word_impact_rows([insert], df_scores={"small": 2.0})

        self.assertEqual(rows[0].word, "small")
        self.assertEqual(rows[0].changed_token_count, 1)
        self.assertAlmostEqual(rows[0].impact_score, 1.5)
        self.assertAlmostEqual(rows[0].df_impact_score, 3.0)


class DfImpactScoringFileIoTest(unittest.TestCase):
    def test_loads_matching_jsonl_and_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            alignment_path = directory / "alignments.jsonl"
            df_path = directory / "df.csv"
            output_prefix = directory / "df_impact"
            alignment_rows = [
                {
                    "sample_id": "image-1",
                    "operation": "substitute",
                    "clean": {
                        "sample_id": "image-1",
                        "prompt": "A cat sits",
                        "token_index": 1,
                        "token_id": 10,
                        "token": "cat</w>",
                        "start_char": 2,
                        "end_char": 5,
                        "hidden_state": [0.0, 1.0],
                    },
                    "poisoned": {
                        "sample_id": "image-1",
                        "prompt": "A cta sits",
                        "token_index": 1,
                        "token_id": 20,
                        "token": "cta</w>",
                        "start_char": 2,
                        "end_char": 5,
                        "hidden_state": [0.0, 3.0],
                    },
                    "l2_distance": 2.0,
                    "cosine_distance": 0.2,
                    "step_cost": 1.0,
                }
            ]
            with alignment_path.open("w", encoding="utf-8") as handle:
                for row in alignment_rows:
                    handle.write(json.dumps(row) + "\n")
            with df_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["word", "DF"])
                writer.writeheader()
                writer.writerow({"word": "cat", "DF": "3.0"})

            steps = load_alignment_steps(alignment_path)
            exit_code = main(
                [
                    str(alignment_path),
                    "--df-table",
                    str(df_path),
                    "--output-prefix",
                    str(output_prefix),
                    "--top-k",
                    "5",
                ]
            )

            self.assertEqual(len(steps), 1)
            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(f"{output_prefix}_word_impact_rows.csv").exists())
            summary_path = Path(f"{output_prefix}_word_impact_summary.csv")
            self.assertTrue(summary_path.exists())
            self.assertTrue(Path(f"{output_prefix}_df_impact_report.md").exists())
            with summary_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["word"], "cat")
            self.assertEqual(rows[0]["df_score"], "3.000000")
            self.assertEqual(rows[0]["max_df_impact_score"], "6.000000")


if __name__ == "__main__":
    unittest.main()
