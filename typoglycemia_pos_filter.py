import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import argparse
import re
import pandas as pd
import numpy as np
try:
    from mylib.utility import print_args
except ImportError:
    def print_args(args):
        for key, value in vars(args).items():
            print(f"{key}: {value}")
import warnings
warnings.filterwarnings("ignore")

'''
This script generates poisoned text using typoglycemia phenomenon.
'''

DATA_PATH = "/data1/share/Datasets/Other/MSCOCO_captioning/train2017_512x512.csv"
BATCH_SIZE = 32
DEFAULT_ALLOWED_POS_PREFIXES = ("NN", "VB")
WORD_TOKEN_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")


class Typoglycemia:
    '''
    add most efficient typoglycemia poisoning to texts in the dataframe.
    '''
    def __init__(self, seed, pos_tagger=None, allowed_pos_prefixes=DEFAULT_ALLOWED_POS_PREFIXES, tokenizer=None):
        # dataframe to store all words in the text, their indices dictionary(text_index, index), DF(Document Frequency), shuffled_word for poisoning
        self.all_words_in_text_dict = pd.DataFrame(columns=["word", "text_index", "DF", "shuffled_word"])
        # pandas dataframe to store original texts and image paths
        self.data_frame = None
        self.allowed_pos_prefixes = tuple(allowed_pos_prefixes)
        self.tokenizer = tokenizer or self._default_word_tokenize
        self.pos_tagger = pos_tagger or self._default_pos_tag
        # set random seed
        import random
        random.seed(seed)
        self.choice = random.choice
        # flag to check if DF scores are calculated
        self.DF_scores_calculated = False
        self.shuffled_words_generated = False
        print(f"Typoglycemia initialized with seed {seed}")
    
    def load_data_frame(self, data_frame):
        '''
        load pandas dataframe

        data_frame: pandas dataframe containing original texts
        return: None
        '''
        # error handling
        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError("data_frame must be a pandas dataframe")
        if data_frame.empty:
            raise ValueError("data_frame is empty")
        if self.data_frame is not None:
            raise ValueError("data_frame is already loaded, please create a new Typoglycemia instance to load a new data frame")
        self.data_frame = data_frame

    def count_words_in_text(self, text_column='text'):
        '''
        count all words in the text while ignoring stop words, their indices and counts.

        text_column: column name of the text in the dataframe
        return: None, updates self.all_words_in_text_dict
        '''
        # error handling
        if self.data_frame is None or self.data_frame.empty:
            raise ValueError("data_frame is empty, please load data frame first using load_data_frame()")
        if text_column not in self.data_frame.columns:
            raise ValueError(f"text_column '{text_column}' not found in data_frame columns")

        # initialize word counts
        self.all_words_in_text_dict = pd.DataFrame(columns=["word", "text_index", "DF", "shuffled_word"])
        self.DF_scores_calculated = False
        self.shuffled_words_generated = False

        #select one text from the dataframe
        for text_index, text in enumerate(self.data_frame[text_column]):
            for index, word in self._allowed_word_tokens(text):
                self._add_word_occurrence(word, text_index, index)
            print(f"Processed text {text_index+1}/{len(self.data_frame)}")

    def calculate_DF_scores(self):
        '''
        calculate Document Frequency (DF) scores for all words in self.all_words_in_text_dict.

        return: None, updates self.all_words_in_text_dict
        '''
        # error handling
        if self.all_words_in_text_dict.empty:
            raise ValueError("all_words_in_text_dict is empty, please run count_words_in_text() first")
        # calculate DF for each word
        for i in range(len(self.all_words_in_text_dict)):
            word = self.all_words_in_text_dict.loc[i]
            df = np.log1p(len(word['text_index']))
            self.all_words_in_text_dict.at[i, 'DF'] = df
        print("DF scores calculated")
        self.DF_scores_calculated = True

    def gen_shuffled_word(self):
        '''
        generate a shuffled word for each word in self.all_words_in_text_dict and update the dataframe.
        return: None, updates self.all_words_in_text_dict
        '''
        # error handling
        if self.all_words_in_text_dict.empty:
            raise ValueError("all_words_in_text_dict is empty, please run count_words_in_text() first")
        # generate shuffled words for each word in the dataframe
        for row in self.all_words_in_text_dict.itertuples():
            word = row.word
            shuffled_words = self.shuffle_word(word)
            # randomly select one shuffled word
            if len(shuffled_words) == 0:
                self.all_words_in_text_dict.drop(index=row.Index, inplace=True)
            else:
                shuffled_word = self.choice(shuffled_words)
                self.all_words_in_text_dict.at[row.Index, 'shuffled_word'] = shuffled_word
        print("Shuffled words generated")
        self.shuffled_words_generated = True

    def gen_poisoned_text(self, max_changed_words=2, text_column='text', image_column='image_path'):
        '''
        generate poisoned words for the most important words in the texts.

        max_changed_words: maximum number of words to be changed in each text
        text_column: column name of the text in the dataframe
        image_column: column name of the image path in the dataframe
        return: pandas dataframe with image path, original texts and poisoned texts
        '''
        #error handling
        if max_changed_words <= 0:
            raise ValueError("max_changed_words must be greater than 0")
        if self.data_frame is None or self.data_frame.empty:
            raise ValueError("data_frame is empty")
        if self.all_words_in_text_dict.empty:
            raise ValueError("all_words_in_text_dict is empty, please run count_words_in_text() first")
        if not self.DF_scores_calculated:
            raise ValueError("DF scores are not calculated, please run calculate_DF_scores() first")
        if text_column not in self.data_frame.columns:
            raise ValueError(f"text_column '{text_column}' not found in data_frame columns")
        if image_column not in self.data_frame.columns:
            raise ValueError(f"image_column '{image_column}' not found in data_frame columns")
        if not self.shuffled_words_generated:
            raise ValueError("shuffled words are not generated, please run gen_shuffled_word() first")
        
        # copy original texts to poisoned_texts_df and add a column to count number of changed words
        poisoned_texts_df = self.data_frame[[image_column, text_column]].copy()
        poisoned_texts_df = poisoned_texts_df.assign(changed_words = 0)
        # for each text, select words to be changed based on their DF probabilities
        for text_index, row in poisoned_texts_df.iterrows():
            original_text = row[text_column]
            tokens = original_text.split()
            candidate_tokens = []
            for index, word in self._allowed_word_tokens(original_text):
                if self.all_words_in_text_dict['word'].isin([word]).any():
                    candidate_tokens.append((index, word))
            # if there are no words to change, continue to next text
            if len(candidate_tokens) == 0:
                continue
            # get DF probabilities of the words in the text
            df_probs = []
            for _, word in candidate_tokens:
                row_index = self.all_words_in_text_dict.index[self.all_words_in_text_dict['word'] == word][0]
                df_score = self.all_words_in_text_dict.at[row_index, 'DF']
                df_probs.append(df_score)
            df_probs = np.array(df_probs) / np.sum(df_probs)
            # select words to be changed based on DF probabilities
            num_words_to_change = min(max_changed_words, len(candidate_tokens))
            np.random.seed(0)  # for reproducibility
            selected_token_indices = np.random.choice(len(candidate_tokens), size=num_words_to_change, replace=False, p=df_probs)
            # change the selected words in the text
            for selected_token_index in selected_token_indices:
                selected_word_index, selected_word = candidate_tokens[selected_token_index]
                # get the shuffled word
                row_index = self.all_words_in_text_dict.index[self.all_words_in_text_dict['word'] == selected_word][0]
                shuffled_word = self.all_words_in_text_dict.at[row_index, 'shuffled_word']
                # replace only the selected token and preserve punctuation/case
                tokens[selected_word_index] = self._replace_token_word(tokens[selected_word_index], shuffled_word)
                # increment the changed words count
                poisoned_texts_df.at[text_index, 'changed_words'] += 1
            poisoned_texts_df.at[text_index, text_column] = ' '.join(tokens)
            # initial letter capitalization check
            poisoned_text = poisoned_texts_df.at[text_index, text_column]
            if original_text[0].isupper():
                poisoned_text = poisoned_text[0].upper() + poisoned_text[1:]
                poisoned_texts_df.at[text_index, text_column] = poisoned_text
            # if the original text ends with a period, add a period to the end of the poisoned text if it doesn't already have one
            if original_text.endswith('.') and not poisoned_text.endswith('.'):
                poisoned_texts_df.at[text_index, text_column] = poisoned_texts_df.at[text_index, text_column] + '.'
            # print progress
            print(f"Processed text {text_index+1}/{len(poisoned_texts_df)}")
        # drop the changed_words column before returning
        poisoned_texts_df = poisoned_texts_df.drop(columns=['changed_words'])
        return poisoned_texts_df

    def _add_word_occurrence(self, word, text_index, index):
        # if the word is not in the dataframe, add it
        if not self.all_words_in_text_dict['word'].isin([word]).any():
            word_df = pd.DataFrame({"word": [word], "text_index": [{text_index: [index]}], "DF": [0], "shuffled_word": [word]})
            self.all_words_in_text_dict = pd.concat([self.all_words_in_text_dict, word_df], ignore_index=True)
            return

        # if the word is already in the dataframe, update its indices and count
        row_index = self.all_words_in_text_dict.index[self.all_words_in_text_dict['word'] == word][0]
        list_of_indices = self.all_words_in_text_dict.at[row_index, 'text_index'].get(text_index, False)
        if list_of_indices is False:
            self.all_words_in_text_dict.at[row_index, 'text_index'][text_index] = [index]
        else:
            list_of_indices.append(index)
            self.all_words_in_text_dict.at[row_index, 'text_index'][text_index] = list_of_indices

    def _allowed_word_tokens(self, text):
        '''
        Return (token_index, normalized_word) for alphabetic nouns and verbs.
        '''
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

    def _find_split_token_index(self, word, split_tokens, start_index):
        for index in range(start_index, len(split_tokens)):
            if self._normalize_token(split_tokens[index]) == word:
                return index
        return None

    def _normalize_token(self, token):
        match = WORD_TOKEN_RE.match(token)
        if not match:
            return ""
        return match.group(2).lower()

    def _replace_token_word(self, token, replacement):
        match = WORD_TOKEN_RE.match(token)
        if not match:
            return token

        prefix, original_word, suffix = match.groups()
        if original_word.isupper():
            cased_replacement = replacement.upper()
        elif original_word[0].isupper():
            cased_replacement = replacement.capitalize()
        else:
            cased_replacement = replacement.lower()
        return prefix + cased_replacement + suffix

    def _is_valid_word(self, word):
        return word.isalpha() and len(word) > 3

    def _is_allowed_pos(self, pos_tag):
        return any(pos_tag.startswith(prefix) for prefix in self.allowed_pos_prefixes)

    def _default_word_tokenize(self, text):
        try:
            import nltk
            return nltk.word_tokenize(text)
        except (ImportError, LookupError):
            raise ImportError("NLTK is not installed or the required NLTK data is not downloaded. Please install NLTK and download the 'punkt' and 'averaged_perceptron_tagger' data.")

    def _default_pos_tag(self, words):
        try:
            import nltk
            return nltk.pos_tag(words)
        except (ImportError, LookupError):
            raise ImportError("NLTK is not installed or the required NLTK data is not downloaded. Please install NLTK and download the 'averaged_perceptron_tagger' data.")
            #return self._heuristic_pos_tag(words)

    def _heuristic_pos_tag(self, words):
        return [(word, self._guess_pos_tag(word)) for word in words]

    def _guess_pos_tag(self, word):
        '''
        Conservative fallback used when no external POS tagger is supplied.
        A caller can pass an NLTK/spaCy-backed pos_tagger for stricter tagging.
        '''
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
        if word in noun_words or word.endswith(("tion", "ment", "ness", "ity", "ship", "age", "ance", "ence", "er", "or", "ist", "ism")):
            return "NN"
        if word.endswith("s") and not word.endswith(("ous", "less")):
            return "NNS"
        return "JJ"

    def shuffle_word(self, word):
        '''
        Shuffle the letters of a word, keeping the first and last letters in place.

        word: the word to be shuffled
        return: the list of all possible shuffled words
        '''
        # check if the word is valid
        if len(word) <= 3 or not word.isalpha():
            raise ValueError("word must be longer than 3 characters and contain only alphabetic characters")
        # get the middle letters of the word
        mid = list(word[1:-1])
        candidate_words = [] # list of all possible shuffled words
        # generate all possible pairs of indices to swap
        for i in range(len(mid)):
            for j in range(i+1, len(mid)):
                # if i and j are the same, skip
                if mid[i] == mid[j]:
                    continue
                changed = mid.copy()
                changed[i], changed[j] = changed[j], changed[i]
                shuffled_word = word[0] + ''.join(changed) + word[-1]
                candidate_words.append(shuffled_word)
        return candidate_words
    
    def save_all_words(self, file_path):
        '''
        save all words in the text, their indices dictionary(text_index, index), counts, and clip scores to a csv file.

        file_path: path to save the csv file
        return: None
        '''
        if self.all_words_in_text_dict.empty:
            raise ValueError("all_words_in_text_dict is empty, please run count_words_in_text() first")
        if not self.DF_scores_calculated:
            raise ValueError("DF scores are not calculated, please run calculate_DF_scores() first")
        if file_path.endswith('.csv'):
            self.all_words_in_text_dict.to_csv(file_path, index=False)
            print(f"All words saved to {file_path}")
        elif file_path.endswith('.tsv'):
            self.all_words_in_text_dict.to_csv(file_path, index=False, sep='\t')
        print(f"All words saved to {file_path}")

def parse_args():
    # arguments parse
    parser = argparse.ArgumentParser(description="Generate poisoned text using typoglycemia")
    parser.add_argument("--data_list", "-dl", type=str, default=DATA_PATH, help="Path to the file containing image and text list for poisoning")
    parser.add_argument("--output_csv_folder", "-oc", type=str, default="~/data/code/mscoco/poisoned", help="Path to save poisoned texts")
    return parser.parse_args()

def load_data_df(data_list_path):
    if not os.path.exists(data_list_path):
        raise FileNotFoundError(f"Data list file not found: {data_list_path}")
    if data_list_path.endswith('.csv'):
        df = pd.read_csv(data_list_path, header=0, encoding='utf-8')
    elif data_list_path.endswith('.tsv'):
        df = pd.read_csv(data_list_path, header=0, encoding='utf-8', sep='\t')
    return df

def main():    
    args = parse_args()
    print_args(args)

    # load data list
    df = load_data_df(args.data_list)
    
    seed = 42
    typoglycemia = Typoglycemia(seed)

    # load data frame
    typoglycemia.load_data_frame(df)
    # count words in all texts
    typoglycemia.count_words_in_text(text_column='Caption')
    # calculate DF scores
    typoglycemia.calculate_DF_scores()
    # generate shuffled words for all words
    typoglycemia.gen_shuffled_word()
    # save all words with their DF scores
    typoglycemia.save_all_words(file_path="~/data/code/all_words_typoglycemia_flickr_random.tsv")

    for i in range(20):
        # generate poisoned texts
        df_out = typoglycemia.gen_poisoned_text(max_changed_words=i+1, text_column='Caption', image_column='File Path')

        # save poisoned texts
        df_out.to_csv(f"{args.output_csv_folder}/poisoned_texts_{i}.tsv", index=False, sep='\t')
        print(f"Poisoned texts saved to {args.output_csv_folder}/poisoned_texts_{i}.tsv")

if __name__ == '__main__':
    main()
