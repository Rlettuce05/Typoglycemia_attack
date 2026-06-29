import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from baseline_comparison import (
    generate_attack_prompt_rows,
    load_scored_comparison_rows,
    main,
    render_comparison_markdown_report,
    summarize_scored_comparison_rows,
)
from gen_poison_typoglycemia_mscoco_DF_choice import Typoglycemia
from poisoning_baselines import CharmerBaseline, TextFoolerBaseline


def fixed_tokenizer(text):
    return text.replace(".", " .").split()


def fixed_pos_tagger(words):
    tags = {
        "a": "DT",
        "an": "DT",
        "person": "NN",
        "walks": "VBZ",
        "on": "IN",
        "street": "NN",
        "artist": "NN",
        "paints": "VBZ",
        "murals": "NNS",
        ".": ".",
    }
    return [(word, tags[word.lower()]) for word in words]


class BaselineComparisonTest(unittest.TestCase):
    def test_generates_typoglycemia_and_baseline_rows_on_same_samples(self):
        df = pd.DataFrame(
            {
                "File Path": ["image-1.jpg", "image-2.jpg"],
                "Caption": [
                    "A person walks on a street.",
                    "An artist paints murals.",
                ],
            }
        )

        rows = generate_attack_prompt_rows(
            df,
            max_changed_words=1,
            typoglycemia_factory=self._typoglycemia_factory,
            attacks=(
                CharmerBaseline(seed=1, operations=("delete",)),
                TextFoolerBaseline(
                    seed=1,
                    synonym_provider={
                        "person": ["individual"],
                        "artist": ["painter"],
                    },
                ),
            ),
        )

        methods_by_sample = {}
        for row in rows:
            methods_by_sample.setdefault(row.sample_id, set()).add(row.method)

        self.assertEqual(
            methods_by_sample,
            {
                "image-1.jpg": {"original", "typoglycemia", "charmer", "textfooler"},
                "image-2.jpg": {"original", "typoglycemia", "charmer", "textfooler"},
            },
        )
        self.assertTrue(all(row.changed_words == 0 for row in rows if row.method == "original"))
        self.assertTrue(any(row.changed_words == 1 for row in rows if row.method == "typoglycemia"))

    def test_summarizes_scored_rows_with_changed_word_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            scored_path = Path(directory) / "scored.csv"
            self._write_scored_rows(scored_path)

            records, changed_words_by_key = load_scored_comparison_rows(scored_path)
            _, summaries = summarize_scored_comparison_rows(records, changed_words_by_key)
            summary_by_method = {summary.method: summary for summary in summaries}

            self.assertAlmostEqual(summary_by_method["typoglycemia"].mean_clip_score_delta, -0.15)
            self.assertAlmostEqual(summary_by_method["typoglycemia"].mean_changed_words, 1.5)
            self.assertEqual(summary_by_method["charmer"].representative_sample_id, "image-2")

            report = render_comparison_markdown_report(summaries)
            self.assertIn("Paper Table Candidate", report)
            self.assertIn("Mean changed words", report)
            self.assertIn("typoglycemia", report)

    def test_cli_writes_attack_prompt_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "captions.csv"
            scored_path = Path(directory) / "scored.csv"
            output_prefix = Path(directory) / "comparison"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["File Path", "Caption"])
                writer.writeheader()
                writer.writerow({"File Path": "image-1.jpg", "Caption": "A person walks on a street."})
            self._write_scored_rows(scored_path)

            exit_code = main(
                [
                    str(input_path),
                    "--output-prefix",
                    str(output_prefix),
                    "--max-changed-words",
                    "1",
                    "--use-heuristic-pos-tagger",
                    "--scored-results",
                    str(scored_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(f"{output_prefix}_attack_prompts.csv").exists())
            self.assertTrue(Path(f"{output_prefix}_clip_delta_rows.csv").exists())
            self.assertTrue(Path(f"{output_prefix}_clip_summary.csv").exists())
            self.assertTrue(Path(f"{output_prefix}_clip_report.md").exists())

    def _typoglycemia_factory(self):
        return Typoglycemia(
            seed=1,
            pos_tagger=fixed_pos_tagger,
            tokenizer=fixed_tokenizer,
        )

    def _write_scored_rows(self, path):
        rows = [
            {
                "sample_id": "image-1",
                "method": "original",
                "prompt": "A person walks on a street.",
                "changed_words": "0",
                "clip_score": "0.50",
            },
            {
                "sample_id": "image-1",
                "method": "typoglycemia",
                "prompt": "A psreon walks on a street.",
                "changed_words": "1",
                "clip_score": "0.40",
            },
            {
                "sample_id": "image-1",
                "method": "charmer",
                "prompt": "A prson walks on a street.",
                "changed_words": "1",
                "clip_score": "0.45",
            },
            {
                "sample_id": "image-2",
                "method": "original",
                "prompt": "An artist paints murals.",
                "changed_words": "0",
                "clip_score": "0.70",
            },
            {
                "sample_id": "image-2",
                "method": "typoglycemia",
                "prompt": "An atrist pinats murals.",
                "changed_words": "2",
                "clip_score": "0.50",
            },
            {
                "sample_id": "image-2",
                "method": "charmer",
                "prompt": "An artist paints muals.",
                "changed_words": "1",
                "clip_score": "0.60",
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
