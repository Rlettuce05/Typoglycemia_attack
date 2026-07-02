import unittest

import pandas as pd

from typoglycemia import Typoglycemia


def fixed_tokenizer(text):
    return text.replace(".", " .").split()


class TypoglycemiaTest(unittest.TestCase):
    def test_count_words_uses_all_valid_words_without_pos_filter(self):
        df = pd.DataFrame(
            {
                "File Path": ["image.jpg"],
                "Caption": ["The quick artist paints bright murals."],
            }
        )

        typoglycemia = Typoglycemia(seed=1, tokenizer=fixed_tokenizer)
        typoglycemia.load_data_frame(df)
        typoglycemia.count_words_in_text(text_column="Caption")

        self.assertEqual(
            set(typoglycemia.all_words_in_text_dict["word"]),
            {"quick", "artist", "paints", "bright", "murals"},
        )


if __name__ == "__main__":
    unittest.main()
