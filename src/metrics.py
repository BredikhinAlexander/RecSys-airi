from typing import Tuple
import numpy as np
import pandas as pd


def topn_recommendations(scores: np.ndarray, topn: int = 10) -> np.ndarray:
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def topidx(a: np.ndarray, topn: int = 10) -> np.ndarray:
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]


def get_metrics(scores: np.ndarray, holdout: pd.DataFrame, topn: int = 10) -> Tuple[float, float]:
    recommended_items = topn_recommendations(scores)
    holdout_items = holdout['movieid'].values
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    # HR calculation
    hr = float(np.mean(hits_mask.any(axis=1)))
    # MRR calculation
    n_test_users = recommended_items.shape[0]
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
    return hr, mrr
