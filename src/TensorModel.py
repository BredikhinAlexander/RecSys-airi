from typing import List, Tuple

import numpy as np
import pandas as pd
from polara.lib.tensor import hooi
from scipy import sparse
from tqdm import tqdm


class TensorModel:
    def __init__(
            self,
            n_items: int,
            core_shape: List,
            n_ratings: int = 10,
            num_iters: int = 5,
            rating_plus: Tuple = (8, 9),
            rating_minus: Tuple = (0, 1)
    ):
        self.n_items = n_items
        self.core_shape = core_shape
        self.n_ratings = n_ratings
        self.num_iters = num_iters
        self.rating_plus = rating_plus
        self.rating_minus = rating_minus
        self.u1, self.u2 = None, None

    def fit(self, train_data: pd.DataFrame) -> None:
        train = train_data.drop(['timestamp'], axis=1)
        train['rating'], rating_idx_map = pd.factorize(train['rating'], sort=True)

        idx = train.values
        val = np.ones(idx.shape[0])
        shape = [max(train.userid) + 1, self.n_items, self.n_ratings]

        _, self.u1, self.u2, _ = hooi(idx, val, shape, self.core_shape,
                                      num_iters=self.num_iters, verbose=True, seed=1509)

    def get_recommendation(self, test: pd.DataFrame, top_n: int = 10) -> np.ndarray:
        test['rating'], rating_idx_map = pd.factorize(test['rating'], sort=True)
        tensor_recommend, tensor_scores = [], []
        for user in tqdm(test.userid.unique()):
            current_mat = self.make_user_matrix(test[test['userid'] == user], self.n_items)
            inner_matrix = self.u1.T @ current_mat @ self.u2
            rating_matrix = self.u1 @ inner_matrix @ self.u2.T

            # "алгебра" рейтингов
            expression = (np.sum(rating_matrix[:, self.rating_plus], axis=1) -
                          np.sum(rating_matrix[:, self.rating_minus], axis=1))
            # items = np.argsort(expression)[::-1]
            # scores = expression[items]
            tensor_scores.append(expression)

            # tensor_scores.append(scores)
            # tensor_recommend.append(items)

        return np.vstack(tensor_scores)

    @staticmethod
    def make_user_matrix(train_data: pd.DataFrame,
                         n_items: int, n_ratings: int = 10) -> sparse.csr_matrix:
        cols, rows = train_data["rating"].values, train_data["movieid"].values
        data = np.zeros((n_items, n_ratings))
        data[rows, cols] = 1
        return sparse.csr_matrix(data)
