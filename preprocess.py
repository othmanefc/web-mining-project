import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

tqdm.pandas()


class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_parent_time_score(self):
        '''
        Getting the parent of the score and the delta of time between
        both
        '''
        self.data = pd.merge(self.data,
                             self.data[['ups', 'name', 'created_utc']],
                             left_on='parent_id',
                             right_on='name',
                             suffixes=('', '_y'),
                             how='left')
        self.data['time_since_parent'] = self.data['created_utc'] - self.data[
            'created_utc_y']
        self.data.drop(columns=['name_y', 'created_utc_y'], inplace=True)
        self.data.rename(columns={'ups_y': 'parent_ups'}, inplace=True)
        self.data.drop_duplicates(inplace=True)

    def get_depth(self):
        '''
        Getting the number of comments between the root and the comment
        '''
        val = 0
        self.data['depth'] = np.nan
        assert 'is_root' in self.data.columns
        roots = self.data[self.data.is_root == True]
        self.data.loc[roots.index, 'depth'] = val
        seen_idx = list(roots.index)

        def depth(data, roots, val, seen_idx):
            if len(roots) == 0:
                return data
            data_copy = data.iloc[~data.index.isin(seen_idx), :]
            merged = pd.merge(roots,
                              data_copy[['name', 'parent_id']].reset_index(),
                              left_on='name',
                              right_on='parent_id',
                              how='inner')
            data.loc[data.index.isin(list(merged['index'])), 'depth'] = val + 1
            seen_idx.extend(list(merged['index']))

            new_roots = data.loc[data.index.isin(list(merged['index'])), :]
            depth(data, new_roots, val + 1, seen_idx)

        depth(self.data, roots, val, seen_idx)

    def get_time(self):
        '''
        Parsing the timestamp to get the weekday and the hour
        '''
        self.data['weekday'] = self.data.created_utc.progress_apply(
            lambda x: datetime.fromtimestamp(x).weekday())
        self.data['hour'] = self.data.created_utc.progress_apply(
            lambda x: datetime.fromtimestamp(x).hour)

    def get_root(self):
        '''
        Flagging if the comment is a root comment or not
        '''
        self.data['is_root'] = (
            self.data.parent_id == self.data.link_id).astype(int)

    def clean_missing(self):
        '''
        Imputing some missing values
        '''
        self.data['time_since_parent'] = self.data.progress_apply(
            lambda x: 0 if x.is_root == True else x.time_since_parent, axis=1)

        self.data['parent_ups'] = self.data.progress_apply(
            lambda x: 0
            if (x.is_root == True and not np.isnan(x.ups)) else x.parent_ups,
            axis=1)
        self.data['time_since_parent'] = self.data['time_since_parent'].fillna(
            self.data['time_since_parent'].mode()[0])
        self.data['body'] = np.where(self.data['body'] == '[deleted]', '',
                                     self.data['body'])
        self.data['body'] = self.data.body.fillna('')
        self.data['depth'] = self.data.depth.fillna(0)
        self.data['parent_ups'] = self.data['parent_ups'].fillna(
            self.data['parent_ups'].mode()[0])

    def get_counts(self):
        '''
        Getting aggregate features such as the number of comments until that 
        comment and the overall comment count of the author
        '''
        self.data['num_comments'] = self.data.sort_values(
            "created_utc",
            ascending=True).groupby('author').agg("cumcount") + 1
        self.data['deleted'] = (self.data["author"].str.contains(
            "deleted", case=False) | self.data["body"].str.contains(
                "deleted", case=False)).astype(int)
        self.data['author_count'] = self.data.author.map(
            self.data.author.value_counts())
        self.data['parent_id_count'] = self.data.parent_id.map(
            self.data.parent_id.value_counts())

    def preprocess(self):
        print('Getting parent and timing')
        self.get_parent_time_score()
        print('Getting root')
        self.get_root()
        print('Cleaning missing...')
        self.get_depth()
        print('Getting time...')
        self.clean_missing()
        print('Getting depth...')
        self.get_time()
        print('Getting counts...')
        self.get_counts()
