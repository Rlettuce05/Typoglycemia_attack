import csv
import json
import tempfile
import unittest
from pathlib import Path

from df_impact_features import (
    ClipTextHiddenStateExtractor,
    load_prompt_rows,
    write_csv,
    write_jsonl,
)


class FakeOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class FakeModel:
    def __init__(self):
        self.called_with_hidden_states = False
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, **kwargs):
        self.called_with_hidden_states = kwargs["output_hidden_states"]
        input_ids = kwargs["input_ids"]
        layer_zero = []
        layer_one = []
        for row_index, row in enumerate(input_ids):
            zero_row = []
            one_row = []
            for token_index, _ in enumerate(row):
                zero_row.append([float(row_index), float(token_index)])
                one_row.append([float(row_index + 10), float(token_index + 20)])
            layer_zero.append(zero_row)
            layer_one.append(one_row)
        return FakeOutputs(hidden_states=[layer_zero, layer_one])


class FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        self.last_kwargs = kwargs
        return {
            "input_ids": [
                [49406, 101, 102, 49407],
                [49406, 201, 49407, 0],
            ],
            "attention_mask": [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
            ],
            "offset_mapping": [
                [(0, 0), (0, 1), (2, 5), (0, 0)],
                [(0, 0), (0, 3), (0, 0), (0, 0)],
            ],
            "special_tokens_mask": [
                [1, 0, 0, 1],
                [1, 0, 1, 1],
            ],
        }

    def convert_ids_to_tokens(self, token_ids):
        token_map = {
            49406: "<|startoftext|>",
            49407: "<|endoftext|>",
            101: "a</w>",
            102: "cat</w>",
            201: "dog</w>",
            0: "<pad>",
        }
        return [token_map[token_id] for token_id in token_ids]


class OffsetlessTokenizer(FakeTokenizer):
    def __call__(self, prompts, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            raise ValueError("offsets require a fast tokenizer")
        rows = super().__call__(prompts, **kwargs)
        rows.pop("offset_mapping")
        return rows


class ClipTextHiddenStateExtractorTest(unittest.TestCase):
    def test_extracts_non_special_token_hidden_states(self):
        model = FakeModel()
        extractor = ClipTextHiddenStateExtractor(
            tokenizer=FakeTokenizer(),
            model=model,
            max_length=8,
        )

        records = extractor.encode(
            ["A cat", "Dog"],
            sample_ids=["image-1", "image-2"],
            hidden_layer=-1,
        )

        self.assertTrue(model.eval_called)
        self.assertTrue(model.called_with_hidden_states)
        self.assertEqual([record.token for record in records], ["a</w>", "cat</w>", "dog</w>"])
        self.assertEqual([record.sample_id for record in records], ["image-1", "image-1", "image-2"])
        self.assertEqual(records[0].start_char, 0)
        self.assertEqual(records[0].end_char, 1)
        self.assertEqual(records[0].hidden_state, (10.0, 21.0))
        self.assertFalse(records[0].is_special_token)

    def test_can_include_special_tokens_but_skips_padding(self):
        extractor = ClipTextHiddenStateExtractor(tokenizer=FakeTokenizer(), model=FakeModel())

        records = extractor.encode(["A cat", "Dog"], include_special_tokens=True)

        self.assertEqual(len(records), 7)
        self.assertEqual(records[0].token, "<|startoftext|>")
        self.assertTrue(records[0].is_special_token)
        self.assertNotIn("<pad>", [record.token for record in records])

    def test_offsetless_tokenizer_still_extracts_hidden_states(self):
        extractor = ClipTextHiddenStateExtractor(tokenizer=OffsetlessTokenizer(), model=FakeModel())

        records = extractor.encode(["A cat"])

        self.assertIsNone(records[0].start_char)
        self.assertIsNone(records[0].end_char)

    def test_rejects_mismatched_sample_ids(self):
        extractor = ClipTextHiddenStateExtractor(tokenizer=FakeTokenizer(), model=FakeModel())

        with self.assertRaisesRegex(ValueError, "same length"):
            extractor.encode(["A cat"], sample_ids=["one", "two"])


class ClipFeatureFileIoTest(unittest.TestCase):
    def test_load_prompt_rows_infers_columns_and_writers_serialize_vectors(self):
        extractor = ClipTextHiddenStateExtractor(tokenizer=FakeTokenizer(), model=FakeModel())
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "prompts.csv"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["File Path", "Caption"])
                writer.writeheader()
                writer.writerow({"File Path": "image.jpg", "Caption": "A cat"})

            sample_ids, prompts = load_prompt_rows(input_path)
            records = extractor.encode(prompts, sample_ids=sample_ids)
            jsonl_path = Path(directory) / "features.jsonl"
            csv_path = Path(directory) / "features.csv"

            write_jsonl(records, jsonl_path)
            write_csv(records, csv_path)

            self.assertEqual(sample_ids, ["image.jpg"])
            self.assertEqual(prompts, ["A cat"])
            json_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(json_rows[0]["hidden_state"], [10.0, 21.0])
            with csv_path.open(encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(json.loads(csv_rows[0]["hidden_state"]), [10.0, 21.0])


if __name__ == "__main__":
    unittest.main()
