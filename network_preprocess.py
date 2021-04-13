from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx
import numpy as np
import pandas as pd


class NetworkTransformer(BaseEstimator, TransformerMixin):
    def _fit_author_attributes(self, X: pd.DataFrame):
        self.author_attributes = {}
        self.author_attributes['author_implication'] = (
            X.groupby('author').count().iloc[:, 0].to_dict()
        )  # to merge on author
        # Centrality
        self.author_attributes[
            "author_degree_centrality"] = nx.centrality.degree_centrality(
                self.g_author)
        self.author_attributes[
            "author_out_degree_centrality"] = nx.centrality.out_degree_centrality(
                self.g_author)
        self.author_attributes[
            "author_in_degree_centrality"] = nx.centrality.in_degree_centrality(
                self.g_author)
        # Voterank
        lst_voterank = nx.voterank(self.g_author, 1000)
        ordered_author = {}
        for elt in lst_voterank:
            ordered_author[elt] = 1
        self.author_attributes['author_is_influential'] = ordered_author

    def _fit_links_attributes(self, X: pd.DataFrame):
        self.link_attributes = {}
        self.link_attributes["link_popularity"] = (
            X.groupby('link_id').count().iloc[:, 0].to_dict())

    def _fit_network(self, X: pd.DataFrame):
        self.g_coms = nx.from_pandas_edgelist(X,
                                              'name',
                                              'parent_id',
                                              create_using=nx.DiGraph())
        author_to = (X.loc[X['author'] != "[deleted]", :].loc[
            X['author'] != "AutoModerator", :].groupby(
                ['name', 'author']).size().reset_index().drop(0, axis=1))

        author_from = (X.loc[X['author'] != "[deleted]", :].loc[
            X['author'] != "AutoModerator", :].groupby(
                ['parent_id', 'author']).size().reset_index().drop(0, axis=1))
        df_author_network = (author_from.merge(
            author_to, right_on="name", left_on="parent_id").rename(
                columns={
                    "author_x": "author_from",
                    "author_y": "author_to"
                }).drop(['parent_id', 'name'], axis=1).groupby([
                    'author_from', 'author_to'
                ]).size().reset_index().rename(columns={0: "weight"}))
        df_author_network.sort_values(by="weight", ascending=True)
        self.g_author = nx.from_pandas_edgelist(df_author_network,
                                                'author_from',
                                                'author_to',
                                                create_using=nx.DiGraph())

    def _fit_network_attributes(self):
        self.graph_metrics = {}

        # Ancestors & Descendants
        ancestors_dict = {}
        descendant_dict = {}
        for n in self.g_coms.nodes.keys():
            ancestors_dict[n] = len(nx.algorithms.dag.ancestors(
                self.g_coms, n))
            descendant_dict[n] = len(
                nx.algorithms.dag.descendants(self.g_coms, n))
        self.graph_metrics["n_ancestors"] = ancestors_dict
        self.graph_metrics["n_descendants"] = descendant_dict

        # Centrality
        self.graph_metrics[
            "com_degree_centrality"] = nx.centrality.degree_centrality(
                self.g_coms)
        self.graph_metrics[
            "com_out_degree_centrality"] = nx.centrality.out_degree_centrality(
                self.g_coms)
        self.graph_metrics[
            "com_in_degree_centrality"] = nx.centrality.in_degree_centrality(
                self.g_coms)

    def _transform_comments(self, X: pd.DataFrame):
        for key, value in self.graph_metrics.items():
            X[key] = X['name'].map(value)
        return X

    def _transform_author(self, X: pd.DataFrame):
        X['author_is_moderator'] = np.where(X['author'] == "AutoModerator", 1,
                                            0)
        X['author_is_deleted'] = np.where(X['author'] == "[deleted]", 1, 0)
        for key, value in self.author_attributes.items():
            X[key] = X['author'].map(value)
            X[key] = X[key].fillna(0)
            # Remove the indicators for deleted author
            X[X['author_is_deleted'] == 1][key] = 0
        return X

    def _transform_links(self, X: pd.DataFrame):
        for key, value in self.link_attributes.items():
            X[key] = X['link_id'].map(value)
        return X

    def fit(self, X: pd.DataFrame):
        print("Fit networks started")
        self._fit_network(X)
        print("Fit networks attributes")
        self._fit_network_attributes()
        print("Fit author attributes")
        self._fit_author_attributes(X)
        print("First link attributes")
        self._fit_links_attributes(X)
        print("Done")
        return self

    def transform(self, X: pd.DataFrame):
        print("Transform networks started")
        X_ = X.copy()
        print("Transforming comments")
        X_ = self._transform_comments(X_)
        print("Transforming authors")
        X_ = self._transform_author(X_)
        print("Transforming links")
        X_ = self._transform_links(X_)
        print("Done")
        return X_
