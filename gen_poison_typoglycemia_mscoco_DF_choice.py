import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import argparse
import pandas as pd
import numpy as np
from mylib.utility import print_args
import warnings
warnings.filterwarnings("ignore")

'''
This script generates poisoned text using typoglycemia phenomenon.
'''

DATA_PATH = "C:/Users/okamu/OneDrive/デスクトップ/LAB/研究/dataset/MScoco/train2017_512x512.csv"
BATCH_SIZE = 32

class Typoglycemia:
    '''
    add most efficient typoglycemia poisoning to texts in the dataframe.
    '''
    def __init__(self, seed):
        # dataframe to store all words in the text, their indices dictionary(text_index, index), DF(Document Frequency), shuffled_word for poisoning
        self.all_words_in_text_dict = pd.DataFrame(columns=["word", "text_index", "DF", "shuffled_word"])
        # pandas dataframe to store original texts and image paths
        self.data_frame = None
        # set random seed
        import random
        random.seed(seed)
        self.choice = random.choice
        # flag to check if DF scores are calculated
        self.DF_scores_calculated = False
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
        if self.data_frame.empty:
            raise ValueError("data_frame is empty, please load data frame first using load_data_frame()")
        if text_column not in self.data_frame.columns:
            raise ValueError(f"text_column '{text_column}' not found in data_frame columns")

        # initialize word counts
        self.all_words_in_text_dict = pd.DataFrame(columns=["word", "text_index", "DF", "shuffled_word"])

        #select one text from the dataframe
        for text_index, text in enumerate(self.data_frame[text_column]):
            # split text into words and count
            words = text.split()
            # check and add each word to the dataframe
            for index, word in enumerate(words):
                # remove punctuation from the end of the word
                if word.endswith('.'):
                    word = word[:-1]
                # check if the word is valid
                word = word.lower()
                if word.isalpha() and len(word) > 3:
                    # if the word is not in the dataframe, add it
                    if not self.all_words_in_text_dict['word'].isin([word]).any():
                        self.all_words_in_text_dict = pd.concat([self.all_words_in_text_dict, pd.DataFrame({"word": [word], "text_index": [{text_index: [index]}], "DF": [0], "shuffled_word": [word]})], ignore_index=True)
                    # if the word is already in the dataframe, update its indices and count
                    else:
                        # find the row index of the word
                        row_index = self.all_words_in_text_dict.index[self.all_words_in_text_dict['word'] == word][0]
                        # update the indices dictionary and count
                        list_of_indices = self.all_words_in_text_dict.at[row_index, 'text_index'].get(text_index, False)
                        if list_of_indices is False:
                            self.all_words_in_text_dict.at[row_index, 'text_index'][text_index] = [index]
                        else:
                            list_of_indices.append(index)
                            self.all_words_in_text_dict.at[row_index, 'text_index'][text_index] = list_of_indices
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
        if self.data_frame.empty:
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
            words = original_text.split()
            # get words in the text that are in the all_words_in_text_dict
            words_in_text = []
            for index, word in enumerate(words):
                # remove punctuation from the end of the word
                if word.endswith('.'):
                    word = word[:-1]
                word_lower = word.lower()
                if self.all_words_in_text_dict['word'].isin([word_lower]).any():
                    words_in_text.append(words[index])
            # if there are no words to change, continue to next text
            if len(words_in_text) == 0:
                continue
            # get DF probabilities of the words in the text
            df_probs = []
            for word in words_in_text:
                word = word.lower()
                row_index = self.all_words_in_text_dict.index[self.all_words_in_text_dict['word'] == word][0]
                df_score = self.all_words_in_text_dict.at[row_index, 'DF']
                df_probs.append(df_score)
            df_probs = np.array(df_probs) / np.sum(df_probs)
            # select words to be changed based on DF probabilities
            num_words_to_change = min(max_changed_words, len(words_in_text))
            np.random.seed(0)  # for reproducibility
            selected_words = np.random.choice(words_in_text, size=num_words_to_change, replace=False, p=df_probs)
            # change the selected words in the text
            for selected_word in selected_words:
                # get the shuffled word
                selected_word_lower = selected_word.lower()
                row_index = self.all_words_in_text_dict.index[self.all_words_in_text_dict['word'] == selected_word_lower][0]
                shuffled_word = self.all_words_in_text_dict.at[row_index, 'shuffled_word']
                # replace the word in the text and update the poisoned text
                poisoned_texts_df.at[text_index, text_column] = poisoned_texts_df.at[text_index, text_column].replace(selected_word, shuffled_word)
                # increment the changed words count
                poisoned_texts_df.at[text_index, 'changed_words'] += 1
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
    parser.add_argument("--output_csv_folder", "-oc", type=str, default="C:\\Users\\okamu\\OneDrive\\デスクトップ\\LAB\\研究\\dataset\\MScoco\\poisoned", help="Path to save poisoned texts")
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
    typoglycemia.save_all_words(file_path="C:/Users/okamu/OneDrive/デスクトップ/LAB/研究/dataset/MScoco/all_words_typoglycemia_flickr_random.tsv")

    for i in range(20):
        # generate poisoned texts
        df_out = typoglycemia.gen_poisoned_text(max_changed_words=i+1, text_column='Caption', image_column='File Path')

        # save poisoned texts
        df_out.to_csv(f"{args.output_csv_folder}\\poisoned_texts_{i}.tsv", index=False, sep='\t')
        print(f"Poisoned texts saved to {args.output_csv_folder}\\poisoned_texts_{i}.tsv")

if __name__ == '__main__':
    main()
