import csv
import json
import tempfile
import unittest
from pathlib import Path

from df_impact_matching import (
    AlignmentCosts,
    TokenFeature,
    align_feature_maps,
    align_token_sequences,
    read_feature_map,
    write_alignment_csv,
    write_alignment_jsonl,
)


def token(sample_id, index, value, hidden_state=None):
    return TokenFeature(
        sample_id=sample_id,
        prompt="prompt",
        token_index=index,
        token_id=100 + index,
        token=value,
        start_char=index,
        end_char=index + 1,
        hidden_state=tuple(hidden_state or [float(index), 0.0]),
    )


class DfImpactMatchingTest(unittest.TestCase):
    def test_aligns_equal_tokens_and_marks_substitutions(self):
        clean_tokens = [
            token("image-1", 0, "a</w>", [1.0, 0.0]),
            token("image-1", 1, "cat</w>", [0.0, 1.0]),
            token("image-1", 2, "sat</w>", [1.0, 1.0]),
        ]
        poisoned_tokens = [
            token("image-1", 0, "a</w>", [1.0, 0.0]),
            token("image-1", 1, "cta</w>", [0.0, 3.0]),
            token("image-1", 2, "sat</w>", [1.0, 2.0]),
        ]

        alignments = align_token_sequences("image-1", clean_tokens, poisoned_tokens)

        self.assertEqual([alignment.operation for alignment in alignments], ["match", "substitute", "match"])
        self.assertEqual(alignments[1].clean.token, "cat</w>")
        self.assertEqual(alignments[1].poisoned.token, "cta</w>")
        self.assertEqual(alignments[1].l2_distance, 2.0)
        self.assertEqual(alignments[1].step_cost, 1.0)

    def test_aligns_inserted_tokens(self):
        clean_tokens = [token("image-1", 0, "a</w>"), token("image-1", 1, "cat</w>")]
        poisoned_tokens = [
            token("image-1", 0, "a</w>"),
            token("image-1", 1, "small</w>"),
            token("image-1", 2, "cat</w>"),
        ]

        alignments = align_token_sequences("image-1", clean_tokens, poisoned_tokens)

        self.assertEqual([alignment.operation for alignment in alignments], ["match", "insert", "match"])
        self.assertIsNone(alignments[1].clean)
        self.assertEqual(alignments[1].poisoned.token, "small</w>")

    def test_rejects_unpaired_sample_ids_by_default(self):
        clean_by_sample = {"clean-only": [token("clean-only", 0, "a</w>")]}
        poisoned_by_sample = {"poison-only": [token("poison-only", 0, "a</w>")]}

        with self.assertRaisesRegex(ValueError, "missing poisoned features"):
            align_feature_maps(clean_by_sample, poisoned_by_sample)

    def test_can_skip_unpaired_sample_ids(self):
        clean_by_sample = {"clean-only": [token("clean-only", 0, "a</w>")]}
        poisoned_by_sample = {"poison-only": [token("poison-only", 0, "a</w>")]}

        self.assertEqual(align_feature_maps(clean_by_sample, poisoned_by_sample, allow_unpaired=True), [])

    def test_hidden_weight_can_affect_pair_cost(self):
        clean_tokens = [token("image-1", 0, "cat</w>", [0.0, 0.0])]
        poisoned_tokens = [token("image-1", 0, "cat</w>", [3.0, 4.0])]
        costs = AlignmentCosts(hidden_weight=0.2)

        alignments = align_token_sequences("image-1", clean_tokens, poisoned_tokens, costs=costs)

        self.assertEqual(alignments[0].operation, "match")
        self.assertEqual(alignments[0].step_cost, 1.0)


class DfImpactMatchingFileIoTest(unittest.TestCase):
    def test_reads_jsonl_and_writes_alignment_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            clean_path = directory / "clean.jsonl"
            poisoned_path = directory / "poisoned.jsonl"
            csv_path = directory / "aligned.csv"
            jsonl_path = directory / "aligned.jsonl"
            self.write_jsonl(
                clean_path,
                [
                    {
                        "sample_id": "image-1",
                        "prompt": "A cat",
                        "token_index": 0,
                        "token_id": 101,
                        "token": "cat</w>",
                        "start_char": 2,
                        "end_char": 5,
                        "hidden_state": [0.0, 1.0],
                    }
                ],
            )
            self.write_jsonl(
                poisoned_path,
                [
                    {
                        "sample_id": "image-1",
                        "prompt": "A cta",
                        "token_index": 0,
                        "token_id": 201,
                        "token": "cta</w>",
                        "start_char": 2,
                        "end_char": 5,
                        "hidden_state": [0.0, 3.0],
                    }
                ],
            )

            clean_by_sample = read_feature_map(clean_path)
            poisoned_by_sample = read_feature_map(poisoned_path)
            alignments = align_feature_maps(clean_by_sample, poisoned_by_sample)
            write_alignment_csv(alignments, csv_path)
            write_alignment_jsonl(alignments, jsonl_path)

            with csv_path.open(encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            json_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(csv_rows[0]["operation"], "substitute")
            self.assertEqual(csv_rows[0]["l2_distance"], "2.0")
            self.assertEqual(json_rows[0]["clean"]["token"], "cat</w>")
            self.assertEqual(json_rows[0]["poisoned"]["token"], "cta</w>")

    def write_jsonl(self, path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    unittest.main()
