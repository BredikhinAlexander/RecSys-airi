import hydra
from omegaconf import DictConfig, OmegaConf

from src import train_test_split, tune_params_and_fit


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, True)

    stage1_train, stage2_predict, stage2_train, stage2_holdout, \
    final_training, final_testset, final_holdout = train_test_split(**cfg.dataset)
    n_items = max(final_training.movieid) + 1

    model, ease_param_best, tensor_param_best, catboost_param_best = tune_params_and_fit(
        stage1_train, stage2_predict, stage2_train, stage2_holdout, n_items, num_params=cfg.optuna.num_params)

    model.fit_first_level_model(final_training)
    recommend = model.predict(final_testset)
    hr, mrr = model.get_metrics(recommend, final_holdout)
    print(f'hr: {hr}, mrr: {mrr}')


if __name__ == '__main__':
    main()
