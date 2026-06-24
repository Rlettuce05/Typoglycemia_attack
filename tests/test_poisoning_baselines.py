import unittest

import pandas as pd

from poisoning_baselines import CharmerBaseline, TextFoolerBaseline


class PoisoningBaselinesTest(unittest.TestCase):
    def test_charmer_baseline_changes_ranked_words_and_tracks_edits(self):
        attack = CharmerBaseline(seed=7, operations=("swap",))

        result = attack.poison_text("The artist paints murals.", max_changed_words=2)

        self.assertEqual(result.changed_words, 2)
        self.assertNotEqual(result.poisoned_text, result.original_text)
        self.assertTrue(result.poisoned_text.endswith("."))
        self.assertEqual({edit.attack_name for edit in result.edits}, {"charmer"})
        self.assertEqual(len({edit.token_index for edit in result.edits}), 2)

    def test_textfooler_baseline_uses_importance_ranked_synonyms(self):
        attack = TextFoolerBaseline(
            seed=1,
            synonym_provider={
                "artist": ["painter"],
                "murals": ["paintings"],
            },
            importance_scorer=lambda text, word: 10.0 if word == "murals" else 1.0,
        )

        result = attack.poison_text("The Artist paints murals.", max_changed_words=1)

        self.assertEqual(result.poisoned_text, "The Artist paints paintings.")
        self.assertEqual(result.changed_words, 1)
        self.assertEqual(result.edits[0].original, "murals.")
        self.assertEqual(result.edits[0].replacement, "paintings.")

    def test_poison_dataframe_adds_output_and_changed_word_counts(self):
        df = pd.DataFrame(
            {
                "File Path": ["image.jpg"],
                "Caption": ["A person walks on a street."],
            }
        )
        attack = TextFoolerBaseline(seed=1)

        poisoned_df = attack.poison_dataframe(
            df,
            text_column="Caption",
            image_column="File Path",
            max_changed_words=1,
        )

        self.assertEqual(
            list(poisoned_df.columns),
            ["File Path", "Caption", "poisoned_text", "changed_words"],
        )
        self.assertEqual(poisoned_df.loc[0, "Caption"], df.loc[0, "Caption"])
        self.assertNotEqual(poisoned_df.loc[0, "poisoned_text"], df.loc[0, "Caption"])
        self.assertEqual(poisoned_df.loc[0, "changed_words"], 1)


if __name__ == "__main__":
    unittest.main()
