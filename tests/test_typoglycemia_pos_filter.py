import unittest

import pandas as pd

from pos_filter import PosFilteredTypoglycemia


def fixed_pos_tagger(words):
    tags = {
        "the": "DT",
        "quick": "JJ",
        "artist": "NN",
        "paints": "VBZ",
        "bright": "JJ",
        "murals": "NNS",
        ".": ".",
    }
    return [(word, tags[word.lower()]) for word in words]


def fixed_tokenizer(text):
    return text.replace(".", " .").split()


class PosFilterTest(unittest.TestCase):
    def test_count_words_keeps_only_nouns_and_verbs(self):
        df = pd.DataFrame(
            {
                "File Path": ["image.jpg"],
                "Caption": ["The quick artist paints bright murals."],
            }
        )
        events = []

        def recording_tokenizer(text):
            events.append(("tokenize", text))
            return fixed_tokenizer(text)

        def recording_pos_tagger(words):
            events.append(("tag", list(words)))
            return fixed_pos_tagger(words)

        typoglycemia = PosFilteredTypoglycemia(
            seed=1,
            pos_tagger=recording_pos_tagger,
            tokenizer=recording_tokenizer,
        )

        typoglycemia.load_data_frame(df)
        typoglycemia.count_words_in_text(text_column="Caption")

        self.assertEqual(
            events,
            [
                ("tokenize", "The quick artist paints bright murals."),
                ("tag", ["The", "quick", "artist", "paints", "bright", "murals", "."]),
            ],
        )
        self.assertEqual(
            set(typoglycemia.all_words_in_text_dict["word"]),
            {"artist", "paints", "murals"},
        )

    def test_poisoning_does_not_change_adjectives(self):
        original_caption = "The quick artist paints bright murals."
        df = pd.DataFrame(
            {
                "File Path": ["image.jpg"],
                "Caption": [original_caption],
            }
        )
        typoglycemia = PosFilteredTypoglycemia(
            seed=1,
            pos_tagger=fixed_pos_tagger,
            tokenizer=fixed_tokenizer,
        )

        typoglycemia.load_data_frame(df)
        typoglycemia.count_words_in_text(text_column="Caption")
        typoglycemia.calculate_DF_scores()
        typoglycemia.gen_shuffled_word()

        poisoned_df = typoglycemia.gen_poisoned_text(
            max_changed_words=3,
            text_column="Caption",
            image_column="File Path",
        )
        poisoned_caption = poisoned_df.loc[0, "Caption"]

        self.assertNotEqual(poisoned_caption, original_caption)
        self.assertIn("quick", poisoned_caption)
        self.assertIn("bright", poisoned_caption)
        self.assertTrue(poisoned_caption.endswith("."))

    def test_replacement_preserves_case_and_punctuation(self):
        typoglycemia = PosFilteredTypoglycemia(
            seed=1,
            pos_tagger=fixed_pos_tagger,
            tokenizer=fixed_tokenizer,
        )

        self.assertEqual(
            typoglycemia._replace_token_word("Murals.", "mualrs"),
            "Mualrs.",
        )


if __name__ == "__main__":
    unittest.main()
