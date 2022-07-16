import hydra
from omegaconf import DictConfig, OmegaConf

from src import train_test_split, tune_params_and_fit


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, True)
    # if cfg.re_split_data:
    stage1_train, stage2_predict, stage2_train, stage2_holdout, \
    final_training, final_testset, final_holdout = train_test_split(**cfg.dataset)
    model, ease_param_best, tensor_param_best, catboost_param_best = tune_params_and_fit(
        stage1_train, stage2_predict, stage2_train, stage2_holdout, num_params=10)
    model.fit_first_level_model(final_training)
    scores = model.predict(final_testset)
    hr, mrr = model.get_metrics(scores, final_holdout)


if __name__ == '__main__':
    main()
