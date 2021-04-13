import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from tqdm import tqdm


class Clusterer:
    def __init__(self, data: pd.DataFrame, model=None):
        self.model = model
        self.data = data

    @staticmethod
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @staticmethod
    def get_embeddings(inputs: np.array, model: SentenceTransformer,
                       bs: int) -> np.array:
        tot = len(inputs) // bs
        inputs = Clusterer.chunks(inputs, bs)
        embeddings = []
        for _, inp in tqdm(enumerate(inputs), total=tot):
            embeddings.extend(model.encode(inp))
        return np.vstack(embeddings)

    def process(self, embeddings=None, with_gpu=True):
        if embeddings is None:
            print('Embeddings not passed, will produce them...')
            if not self.model:
                self.model = SentenceTransformer(
                    'distilroberta_base_paraphase-v1')
            self.embeddings = Clusterer.get_embeddings(self.data.body.values,
                                                       self.model, 80)
            np.save('sentence_embeddings.npy',
                    self.embeddings,
                    allow_pickle=True)
        else:
            self.embeddings = embeddings
        print('Clustering...')
        cluster = FaissKMeans(n_clusters=10, n_init=20, max_iter=300)
        cluster.fit(self.embeddings, with_gpu=with_gpu)
        print('Clustering done...')
        self.data['clusters'] = cluster.predict(self.embeddings)


class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, with_gpu=True):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu=with_gpu,
                                   verbose=True)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]
