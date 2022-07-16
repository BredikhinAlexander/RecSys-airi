from typing import Dict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src import EASEModel, TensorModel
from src.metrics import topn_recommendations


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

        self.cat_boost.fit(
            X=cb_data[self.features],
            y=cb_data["target"],
            verbose=False,
        )

    def fit_first_level_model(self, stage1_train: pd.DataFrame):
        ease_data = self.ease_model.load_train_data(stage1_train,
                                                    shape_matrix=(max(stage1_train.userid) + 1, self.n_items))
        self.ease_model.fit(ease_data)
        self.tensor_model.fit(stage1_train)

    def predict(self, test: pd.DataFrame):
        ease_scores, tensor_scores = self.predict_first_level_model(test)
        cb_data = self.prepare_data_for_second_level(test, ease_scores, tensor_scores)

        scores_users = self.cat_boost.predict_proba(cb_data)[:, 1]#.reshape(-1, 200)
        return scores_users

    def predict_first_level_model(self, stage2_predict):
        ease_test = self.ease_model.load_train_data(stage2_predict,
                                                    shape_matrix=(max(stage2_predict.userid) + 1, self.n_items))
        ease_scores = self.ease_model.predict(ease_test.toarray())

        tensor_scores = self.tensor_model.get_recommendation(stage2_predict)
        return ease_scores, tensor_scores

    def prepare_data_for_second_level(self, test, ease_scores, tensor_scores):
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
    def get_metrics(scores, holdout):
        recommended_items = topn_recommendations(scores)
        n_test_users = recommended_items.shape[0]
        rec = np.take(holdout.movieid.values.reshape(-1, 200), recommended_items)
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
