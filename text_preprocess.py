from typing import List
import re

import numpy as np
import pandas as pd
import string
from string import punctuation
from nltk import pos_tag
from nltk.corpus import stopwords
from textblob import TextBlob

stop_words = set(stopwords.words('english'))


class TextPreprocesser:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def preprocess(self):
        self.data = TextPreprocesser.text_features(
            self.data,
            text='body',
            get_pos_feats=False,
            get_textblob_sentiment=False)
        self.clean_missing()

    def clean_missing(self):
        self.data['words_vs_unique'] = self.data['words_vs_unique'].fillna(0)
        self.data['mean_word_len'] = self.data['mean_word_len'].fillna(0)
        self.data['punct_percent'] = self.data['punct_percent'].fillna(0)

    @staticmethod
    def get_polarity(text: str) -> float:
        try:
            textblob = TextBlob(text)
            pol = textblob.sentiment.polarity
        except:
            pol = 0.0
        return pol

    @staticmethod
    def get_subjectivity(text: str) -> float:
        try:
            textblob = TextBlob(text)
            subj = textblob.sentiment.subjectivity
        except:
            subj = 0.0
        return subj

    @staticmethod
    def tag_part_of_speech(text: str) -> List[float]:
        text_splited = text.split(' ')
        text_splited = [
            ''.join(c for c in s if c not in string.punctuation)
            for s in text_splited
        ]
        text_splited = [s for s in text_splited if s]
        pos_list = pos_tag(text_splited)
        noun_count = len(
            [w for w in pos_list if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
        adjective_count = len(
            [w for w in pos_list if w[1] in ('JJ', 'JJR', 'JJS')])
        verb_count = len([
            w for w in pos_list
            if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
        ])
        return [noun_count, adjective_count, verb_count]

    @staticmethod
    def text_features(df: pd.DataFrame,
                      text: str = "body",
                      get_pos_feats=False,
                      get_textblob_sentiment=True) -> pd.DataFrame:
        """
        Extract and add in place many text/NLP features on a pandas dataframe
        for a given column
        """

        df[text] = df[text].progress_apply(
            lambda x: re.sub(r'.http\S+', ' ', x))
        df[f'{text}_char_count'] = df[text].str.len()
        df[f'{text}_num_words'] = df[text].str.split().str.len()

        df['capitals'] = df[text].progress_apply(
            lambda comment: sum(1 for c in comment if c.isupper()))
        df['caps_vs_length'] = df.progress_apply(
            lambda row: float(row['capitals']) / float(row[f'{text}_char_count'
                                                           ])
            if row[f'{text}_char_count'] != 0 else 0,
            axis=1)
        df['num_exclamation_marks'] = df[text].str.count('!')
        df['num_question_marks'] = df[text].str.count('\?')
        df['num_punctuation'] = df[text].progress_apply(
            lambda comment: sum(comment.count(w) for w in '.,;:'))
        df['num_symbols'] = df[text].progress_apply(
            lambda comment: sum(comment.count(w) for w in r'*&$%/:;'))

        df['num_unique_words'] = df[text].progress_apply(
            lambda comment: len(set(w for w in comment.split())))
        df["contains_html"] = np.where(df['body'].str.contains(r'&.*?;'), 1, 0)
        df['words_vs_unique'] = df['num_unique_words'] / df[f'{text}_num_words']

        df['word_density'] = df[f'{text}_char_count'] / (
            df[f'{text}_num_words'] + 1)
        df['punctuation_count'] = df[text].progress_apply(
            lambda x: len("".join(_ for _ in x if _ in punctuation)))

        df['upper_case_word_count'] = df[text].progress_apply(
            lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
        df['stopword_count'] = df[text].progress_apply(lambda x: len(
            [wrd for wrd in x.split() if wrd.lower() in stop_words]))
        df["count_words_title"] = df[text].progress_apply(
            lambda x: len([w for w in str(x).split() if w.istitle()]))
        df["contains_url"] = np.where(
            df[text].str.contains(r'(?:\s)(http|https):[^, ]*'), 1, 0)
        df["mean_word_len"] = df[text].progress_apply(
            lambda x: np.mean([len(w) for w in str(x).split()]))
        df['punct_percent'] = df['num_punctuation'] * 100 / df[
            f'{text}_num_words']

        if get_textblob_sentiment:
            df['polarity'] = df[text].progress_apply(
                TextPreprocesser.get_polarity)
            df['subjectivity'] = df[text].progress_apply(
                TextPreprocesser.get_subjectivity)

        if get_pos_feats:
            df['nouns'], df['adjectives'], df['verbs'] = zip(
                *df[text].progress_apply(lambda comment: TextPreprocesser.
                                         tag_part_of_speech(comment)))
            df['nouns_vs_length'] = df['nouns'] / df[f'{text}_char_count']
            df['adjectives_vs_length'] = df['adjectives'] / df[
                f'{text}_char_count']
            df['verbs_vs_length'] = df['verbs'] / df[f'{text}_char_count']
            df['nouns_vs_words'] = df['nouns'] / df[f'{text}_num_words']
            df['adjectives_vs_words'] = df['adjectives'] / df[
                f'{text}_num_words']
            df['verbs_vs_words'] = df['verbs'] / df[f'{text}_num_words']

            df.drop(['nouns', 'adjectives', 'verbs'], axis=1, inplace=True)
        return df
