from typing import Tuple, Dict

import numpy as np
import pandas as pd
from loguru import logger
from polara import get_movielens_data
from polara.preprocessing.dataframes import reindex, leave_one_out


def transform_indices(data: pd.DataFrame, users: str, items: str):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index


def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_timepoint = data['timestamp'].quantile(
        q=0.95, interpolation='nearest')

    test_data_ = data.query('timestamp >= @test_timepoint')

    train_data_ = data.query(
        'userid not in @test_data_.userid.unique() and timestamp < @test_timepoint'
    )
    return train_data_, test_data_


def get_train_test(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    train_data_, test_data_ = split_data(data)

    training, data_index = transform_indices(train_data_.copy(), 'userid', 'movieid')

    test_data = reindex(test_data_, data_index['items'])

    print(len(training), len(test_data))
    print(len(training['userid'].unique()))
    print(len(test_data['userid'].unique()))
    return training, test_data, data_index


def get_holdout(data: pd.DataFrame, data_index) -> Tuple[pd.DataFrame, pd.DataFrame]:
    testset_, holdout_ = leave_one_out(data, target='timestamp', sample_top=True, random_state=0)

    userid = data_index['users'].name
    test_users = pd.Index(
        # ensure test users are the same across testing data
        np.intersect1d(
            testset_[userid].unique(),
            holdout_[userid].unique()
        )
    )
    testset = (
        testset_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
    )
    holdout = (
        holdout_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
    )

    return testset, holdout


def save_data(
        train: pd.DataFrame,
        test: pd.DataFrame,
        holdout: pd.DataFrame,
        data_root: str,
        level: int
) -> None:
    train.to_csv(data_root + f'train{level}level.csv', index=False)
    test.to_csv(data_root + f'test{level}level.csv', index=False)
    holdout.to_csv(data_root + f'holdout{level}level.csv', index=False)


def train_test_split(
        dataset_path: str = './data/data.zip',
        data_root: str = './data/',
        save_files: bool = False
):
    logger.info("start split dataset")

    mldata = get_movielens_data(dataset_path, include_time=True)

    training, test_data, data_index = get_train_test(mldata)
    testset_2level, holdout_2level = get_holdout(test_data, data_index)

    training1level, test1level = split_data(training)
    testset_1level, holdout_1level = get_holdout(test1level, data_index)

    if save_files:
        save_data(training, testset_2level, holdout_2level, data_root, 2)
        save_data(training1level, testset_1level, holdout_1level, data_root, 1)

    logger.success("end split dataset")

    return training1level, testset_1level, holdout_1level, \
           training, testset_2level, holdout_2level
