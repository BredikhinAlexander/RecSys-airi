from typing import Dict

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from loguru import logger

from src import TwoLevelRecSystem


def parse_params(hyperparameters: Dict):
    CatBoostParams = {"depth", "min_child_samples", "n_estimators", "subsample", "colsample_bylevel", "reg_lambda"}
    TensorModelParams = {"core_shape"}
    EaseModelParams = {"reg_weight"}
    ease_model_params = {key: val for key, val in hyperparameters.items() if key in EaseModelParams}
    tensor_model_params = {key: val for key, val in hyperparameters.items() if key in TensorModelParams}
    catboost_params = {key: val for key, val in hyperparameters.items() if key in CatBoostParams}
    return ease_model_params, tensor_model_params, catboost_params


def tune_params_and_fit(
        train: pd.DataFrame,
        to_predict: pd.DataFrame,
        test: pd.DataFrame,
        holdout: pd.DataFrame,
        n_items: int,
        num_params=10,
):
    # define a search space
    space = {
        # cat boost params
        "depth": hp.randint("depth", 6, 15),
        "min_child_samples": hp.randint("min_child_samples", 100, 500),
        "n_estimators": hp.randint("n_estimators", 1, 100),
        "subsample": hp.randint("subsample", 1, 10),
        "colsample_bylevel": hp.randint("colsample_bylevel", 1, 10),
        "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 5, 10, 20, 50, 100]),
        # TensorModel params
        "core_shape": hp.randint("core_shape", 50, 150),
        # EASEModel params
        "reg_weight": hp.randint("reg_weight", 100, 500)
    }

    best = fmin(
        lambda x: objective(
            hyperparameters=x,
            train=train.copy(),
            to_predict=to_predict.copy(),
            test=test.copy(),
            holdout=holdout.copy(),
            n_items=n_items
        ),
        space,
        algo=tpe.suggest,
        max_evals=num_params,
        rstate=np.random.default_rng(42),
        return_argmin=False,
    )

    best["subsample"] /= 10
    best["colsample_bylevel"] /= 10

    logger.info(f"The Best set of params is found. They are: {str(best)}")
    ease_param_best, tensor_param_best, catboost_param_best = parse_params(best)
    model = TwoLevelRecSystem(n_items, ease_param_best, tensor_param_best, catboost_param_best)
    model.fit(train, to_predict, test)

    return model, ease_param_best, tensor_param_best, catboost_param_best


def objective(
        hyperparameters: Dict,
        train: pd.DataFrame,
        to_predict: pd.DataFrame,
        test: pd.DataFrame,
        holdout: pd.DataFrame,
        n_items: int
) -> float:
    """
    Функция по заданным гиперпараметрам считает скор по кросс-валидации с n_splits и возвращает его
    """
    hyperparameters["subsample"] /= 10
    hyperparameters["colsample_bylevel"] /= 10

    ease_param_best, tensor_param_best, catboost_param_best = parse_params(hyperparameters)
    model = TwoLevelRecSystem(n_items, ease_param_best, tensor_param_best, catboost_param_best)
    model.fit(train, to_predict, test)
    to_predict_test = pd.concat([to_predict, test], axis=0, ignore_index=True, copy=False)
    recommend = model.predict(to_predict_test)
    hr, mrr = model.get_metrics(recommend, holdout)
    hyperparameters["metric"] = hr
    logger.info(str(hyperparameters))

    return -hr
