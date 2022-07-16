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


# Final train-test split
def split_train_test_data(data, time_q=0.95, users='userid', items='movieid', time='timestamp'):
    test_timepoint = data[time].quantile(
        q=time_q, interpolation='nearest'
    )
    test_region = data.query(f'{time} >= @test_timepoint')
    test_users = test_region[users].unique()
    train_data_ = data.query(
        f'{users} not in @test_users and {time} < @test_timepoint'
    )
    training, data_index = transform_indices(train_data_.copy(), users, items)
    test_data_ = pd.concat(
        [
            # add histories of test users before timepoint
            data.query(f'{users} in @test_users and {time} < @test_timepoint'),
            test_region
        ],
        axis = 0,
        ignore_index = False
    )
    test_data = reindex(test_data_, data_index['items'])
    return training, test_data, data_index


# Final holdout split
def split_holdout_data(test_data, users='userid', time='timestamp'):
    testset_, holdout_ = leave_one_out(
        test_data, target=time, sample_top=True, random_state=0
    )
    test_users = pd.Index(
        # ensure test users are the same across testing data
        np.intersect1d(testset_[users].unique(), holdout_[users].unique()
        ),
        name = users
    )
    testset = (
        testset_
        # reindex warm-start users for convenience
        .assign(**{users: test_users.get_indexer(testset_[users])})
        .query(f'{users} >= 0')
        .sort_values(users)
    )
    holdout = (
        holdout_
        # reindex warm-start users for convenience
        .assign(**{users: test_users.get_indexer(holdout_[users])})
        .query(f'{users} >= 0')
        .sort_values(users)
    )
    return testset, holdout, test_users


def split_stage_data(
    data, time_q=0.8, users='userid', items='movieid', time='timestamp'
):
    stage_timepoint = data[time].quantile(
        q=time_q, interpolation='nearest'
    )
    stage2_region = data.query(f'{time} >= @stage_timepoint')
    stage2_users = stage2_region[users].unique()
    stage1_train = data.query(
        f'{users} not in @stage2_users and {time} < @stage_timepoint'
    )
    # store known items to exclude cold start problem from candidate generation step
    known_items = stage1_train[items].unique()
    stage2_predict = data.query(
        f'{users} in @stage2_users and {items} in @known_items and {time} < @stage_timepoint'
    )
    # stage 2 data may still contain cold items - stage 2 model hopefully deals with it
    valid_test_users = stage2_predict[users].unique()
    stage2_train, stage2_holdout, user_index = split_holdout_data(
        stage2_region.query(f'{users} in @valid_test_users')
    )
    # `split_holdout_data` assigns new index to users =>
    # need to ensure consistency in user index
    stage2_predict = reindex(stage2_predict, user_index)
    return stage1_train, stage2_predict, stage2_train, stage2_holdout


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
    """
    stage1_train - обучаем первый свд
    stage2_predict - на этом делаем предсказания с помощью свд и подаем кандидатов в кэтбуст
    stage2_train - это мапим с кандидатами и обучаем кэтбуст
    stage2_holdout - это для тюнинга свд+кэтбуст

    final_training - то на чем обучаем финальный свд
    final_testset - на этом делаем финальный предсказания свд и кэтбуста
    final_holdout - на этом считаем финальные метрики
    """
    logger.info("start split dataset")

    mldata = get_movielens_data(dataset_path, include_time=True)

    final_training, final_test_data, data_index = split_train_test_data(mldata)
    final_testset, final_holdout, _ = split_holdout_data(final_test_data)
    stage1_train, stage2_predict, stage2_train, stage2_holdout = split_stage_data(final_training)

    if save_files:
        save_data(stage2_train, stage2_predict, stage2_holdout, data_root, 2)
        save_data(stage1_train, final_testset, final_holdout, data_root, 1)

    logger.success("end split dataset")

    return stage1_train, stage2_predict, stage2_train, stage2_holdout,\
           final_training, final_testset, final_holdout