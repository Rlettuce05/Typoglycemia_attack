from typoglycemia import Typoglycemia as BaseTypoglycemia
from typoglycemia import run_typoglycemia

DEFAULT_ALLOWED_POS_PREFIXES = ("NN", "VB")


class PosFilteredTypoglycemia(BaseTypoglycemia):
    """
    Typoglycemia attack that only shuffles tokens matching allowed POS prefixes.
    """

    def __init__(self, seed, pos_tagger=None, allowed_pos_prefixes=DEFAULT_ALLOWED_POS_PREFIXES, tokenizer=None):
        super().__init__(seed=seed, tokenizer=tokenizer)
        self.allowed_pos_prefixes = tuple(allowed_pos_prefixes)
        self.pos_tagger = pos_tagger or self._default_pos_tag

    def _allowed_word_tokens(self, text):
        """
        Return (token_index, normalized_word) for alphabetic nouns and verbs.
        """
        text = str(text)
        sentence_tokens = list(self.tokenizer(text))
        if not sentence_tokens:
            return []

        tags = list(self.pos_tagger(sentence_tokens))
        if len(tags) != len(sentence_tokens):
            raise ValueError("pos_tagger must return one tag for each input token")

        split_tokens = text.split()
        split_index = 0
        allowed_tokens = []
        for token, tag_entry in zip(sentence_tokens, tags):
            word = self._normalize_token(token)
            if not self._is_valid_word(word):
                continue

            original_index = self._find_split_token_index(word, split_tokens, split_index)
            if original_index is None:
                continue
            split_index = original_index + 1

            pos_tag = tag_entry[1] if isinstance(tag_entry, (tuple, list)) else tag_entry
            if self._is_allowed_pos(pos_tag):
                allowed_tokens.append((original_index, word))
        return allowed_tokens

    def _is_allowed_pos(self, pos_tag):
        return any(pos_tag.startswith(prefix) for prefix in self.allowed_pos_prefixes)

    def _default_word_tokenize(self, text):
        try:
            import nltk
            return nltk.word_tokenize(text)
        except (ImportError, LookupError):
            raise ImportError(
                "NLTK is not installed or the required NLTK data is not downloaded. "
                "Please install NLTK and download the 'punkt' and 'averaged_perceptron_tagger' data."
            )

    def _default_pos_tag(self, words):
        try:
            import nltk
            return nltk.pos_tag(words)
        except (ImportError, LookupError):
            raise ImportError(
                "NLTK is not installed or the required NLTK data is not downloaded. "
                "Please install NLTK and download the 'averaged_perceptron_tagger' data."
            )

    def _heuristic_pos_tag(self, words):
        return [(word, self._guess_pos_tag(word)) for word in words]

    def _guess_pos_tag(self, word):
        """
        Conservative fallback used when no external POS tagger is supplied.
        A caller can pass an NLTK/spaCy-backed pos_tagger for stricter tagging.
        """
        word = word.lower()
        non_content_words = {
            "about", "above", "after", "again", "against", "almost", "along", "among",
            "around", "because", "before", "behind", "below", "between", "bright",
            "brown", "could", "every", "first", "from", "into", "large", "little",
            "other", "quick", "should", "small", "their", "there", "these", "those",
            "through", "under", "where", "while", "white", "would",
        }
        verb_words = {
            "carry", "carries", "carrying", "catch", "catches", "eating", "holding",
            "jumps", "jumping", "looks", "looking", "paint", "paints", "painting",
            "play", "plays", "playing", "ride", "rides", "riding", "runs", "running",
            "sits", "sitting", "stand", "stands", "standing", "walk", "walks",
            "walking", "wear", "wears", "wearing",
        }
        noun_words = {
            "airplane", "artist", "beach", "bicycle", "bottle", "bridge", "building",
            "child", "children", "computer", "field", "horse", "kitchen", "laptop",
            "man", "murals", "person", "people", "phone", "pizza", "player", "players",
            "road", "room", "sandwich", "skateboard", "snowboard", "street", "surfboard",
            "table", "train", "truck", "woman", "women",
        }

        if word in non_content_words:
            return "JJ"
        if word in verb_words or word.endswith(("ing", "ed")):
            return "VB"
        if word in noun_words or word.endswith(
            ("tion", "ment", "ness", "ity", "ship", "age", "ance", "ence", "er", "or", "ist", "ism")
        ):
            return "NN"
        if word.endswith("s") and not word.endswith(("ous", "less")):
            return "NNS"
        return "JJ"


Typoglycemia = PosFilteredTypoglycemia

__all__ = [
    "BaseTypoglycemia",
    "DEFAULT_ALLOWED_POS_PREFIXES",
    "PosFilteredTypoglycemia",
    "Typoglycemia",
    "main",
]


def main(argv=None):
    return run_typoglycemia(PosFilteredTypoglycemia, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
