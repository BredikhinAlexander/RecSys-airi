from typing import Tuple
from loguru import logger
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize


class EASEModel:
    def __init__(self, reg_weight: int):
        self.reg_weight = reg_weight
        self.item_matrix = None

    def fit(self, train: sparse.csr_matrix):
        """
        Counts item_matrix for future predictions
        """
        logger.info("start fitting")
        X = train
        logger.info("normalize")
        X = normalize(X, norm="l2", axis=1)
        X = normalize(X, norm="l2", axis=0)
        X = sparse.csr_matrix(X)
        # gram matrix
        logger.info("gram matrix")
        G = X.T @ X
        # add reg to diagonal
        G += self.reg_weight * sparse.identity(G.shape[0])
        # convert to dense because inverse will be dense
        G = G.todense()
        # invert. this takes most of the time
        logger.info("invert")
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        # zero out diag
        np.fill_diagonal(B, 0.0)
        self.item_matrix = B

    def predict(self, data: np.ndarray, remove_seen: bool = True) -> np.ndarray:
        """
        Counts scores
        """
        scores = data.dot(self.item_matrix)
        if remove_seen:
            scores[data > 0] = -1e13
        return scores

    @staticmethod
    def load_train_data(
            train_data: pd.DataFrame,
            shape_matrix: Tuple[int, int]
    ) -> sparse.csr_matrix:
        """
        Creates csr_matrix for train
        """
        rows, cols = train_data["userid"], train_data["movieid"]
        data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype="float64", shape=shape_matrix)
        return data
