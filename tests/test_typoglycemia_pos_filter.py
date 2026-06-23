import unittest

import pandas as pd

from gen_poison_typoglycemia_mscoco_DF_choice import Typoglycemia


def fixed_pos_tagger(words):
    tags = {
        "quick": "JJ",
        "artist": "NN",
        "paints": "VBZ",
        "bright": "JJ",
        "murals": "NNS",
    }
    return [(word, tags[word]) for word in words]


class TypoglycemiaPosFilterTest(unittest.TestCase):
    def test_count_words_keeps_only_nouns_and_verbs(self):
        df = pd.DataFrame(
            {
                "File Path": ["image.jpg"],
                "Caption": ["The quick artist paints bright murals."],
            }
        )
        typoglycemia = Typoglycemia(seed=1, pos_tagger=fixed_pos_tagger)

        typoglycemia.load_data_frame(df)
        typoglycemia.count_words_in_text(text_column="Caption")

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
        typoglycemia = Typoglycemia(seed=1, pos_tagger=fixed_pos_tagger)

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
        typoglycemia = Typoglycemia(seed=1, pos_tagger=fixed_pos_tagger)

        self.assertEqual(
            typoglycemia._replace_token_word("Murals.", "mualrs"),
            "Mualrs.",
        )


if __name__ == "__main__":
    unittest.main()
