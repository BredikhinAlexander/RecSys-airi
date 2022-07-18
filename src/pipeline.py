from typing import Dict, Tuple, List, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger
from tqdm import tqdm

from src import EASEModel, TensorModel


class TwoLevelRecSystem:
    def __init__(self, n_items: int, ease_model_params: Dict, tensor_model_params: Dict, catboost_params: Dict):
        self.n_items = n_items

        self.ease_model = EASEModel(**ease_model_params)
        self.tensor_model = TensorModel(self.n_items, **tensor_model_params)
        self.cat_boost = CatBoostClassifier(auto_class_weights='Balanced', **catboost_params)

        self.features = ['ease_score', 'ease_rank', 'tensor_score', 'tensor_rank']

    def fit(self, stage1_train: pd.DataFrame, stage2_predict: pd.DataFrame, stage2_train: pd.DataFrame):
        self.fit_first_level_model(stage1_train)
        ease_scores, tensor_scores = self.predict_first_level_model(stage2_predict)
        cb_data = self.prepare_data_for_second_level(stage2_train, ease_scores, tensor_scores)

        logger.debug("Start fit catboost")
        self.cat_boost.fit(
            X=cb_data[self.features],
            y=cb_data["target"],
            verbose=False,
        )
        logger.debug("Finish fit catboost")

    def fit_first_level_model(self, stage1_train: pd.DataFrame):
        logger.debug("Start fit ease model")
        ease_data = self.ease_model.load_train_data(stage1_train,
                                                    shape_matrix=(max(stage1_train.userid) + 1, self.n_items))
        self.ease_model.fit(ease_data)
        logger.debug("Finish fit ease model")
        logger.debug("Start fit tensor model")
        self.tensor_model.fit(stage1_train)
        logger.debug("Finish fit tensor model")

    def predict(self, test: pd.DataFrame, top_n: int = 10) -> List:
        ease_scores, tensor_scores = self.predict_first_level_model(test)
        cb_data = self.prepare_data_for_second_level(test, ease_scores, tensor_scores)

        scores_users = self.cat_boost.predict_proba(cb_data[self.features])[:, 1]
        recommend = []
        for user in tqdm(cb_data.userid.unique()):
            user_ind = cb_data[cb_data['userid'] == user].index
            cur_user_scores = scores_users[user_ind]
            cur_user_items = cb_data.movieid[user_ind].values
            best_scores_ind = np.argsort(cur_user_scores)[::-1][:top_n]
            recommend.append(cur_user_items[best_scores_ind])
        return recommend

    def predict_first_level_model(self, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        ease_test = self.ease_model.load_train_data(test,
                                                    shape_matrix=(max(test.userid) + 1, self.n_items))
        ease_scores = self.ease_model.predict(ease_test.toarray())

        tensor_scores = self.tensor_model.get_recommendation(test)
        return ease_scores, tensor_scores

    def prepare_data_for_second_level(self, test, ease_scores, tensor_scores) -> pd.DataFrame:
        num_users = int(test.userid.nunique())

        ease_candidates = self.make_candidates(ease_scores, num_users, 'ease')
        tensor_candidates = self.make_candidates(tensor_scores, num_users, 'tensor')

        cb_data = pd.merge(test, ease_candidates, on=['userid', 'movieid'], how='right')
        cb_data = pd.merge(cb_data, tensor_candidates, on=['userid', 'movieid'], how='outer')

        cb_data = cb_data.drop(['timestamp'], axis=1)
        cb_data['target'] = 0
        cb_data.loc[cb_data.rating.notnull(), 'target'] = 1
        cb_data = cb_data.drop(['rating'], axis=1)
        return cb_data

    @staticmethod
    def get_metrics(rec: Union[np.ndarray, List], holdout: pd.DataFrame) -> Tuple[float, float]:
        n_test_users = len(rec)
        hr = np.any(rec == holdout.movieid.values.reshape(-1, 1), axis=1).mean()
        mrr = np.sum(np.where(rec == holdout.movieid.values.reshape(-1, 1))[1] + 1) / n_test_users
        return hr, mrr

    @staticmethod
    def make_candidates(scores: np.ndarray, n_users: int, model_name: str,
                        n_candidates: int = 200) -> pd.DataFrame:
        rec = pd.DataFrame()
        rec['userid'] = [i for i in range(n_users) for _ in range(n_candidates)]
        rec['movieid'] = np.asarray(np.argsort(-scores)[:, :n_candidates]).flatten()
        rec[f'{model_name}_score'] = np.asarray(np.sort(-scores)[:, :n_candidates]).flatten()
        rec[f'{model_name}_rank'] = np.array([np.arange(n_candidates)] * n_users).flatten()
        return rec
