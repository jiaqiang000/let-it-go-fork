import pickle
from typing import Self

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


class EmbeddingManager(Pipeline):
    def __init__(self, embedding_dim: int, reduce: bool = True, normalize: bool = True) -> None:
        steps = []

        if reduce:
            steps.append(('scaler', StandardScaler()))
            steps.append(('pca', PCA(n_components=embedding_dim)))

        if normalize:
            steps.append(('normalizer', Normalizer()))

        super().__init__(steps)

        self.embedding_dim = embedding_dim
        self.reduce = reduce
        self.normalize = normalize

    def save(self, filepath: str) -> None:
        with open(filepath, mode='wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath: str) -> Self:
        with open(filepath, mode='rb') as file:
            return pickle.load(file)
