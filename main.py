import hydra
from omegaconf import DictConfig, OmegaConf

from src import train_test_split


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, True)
    if cfg.re_split_data:
        split_data = train_test_split(**cfg.dataset)


if __name__ == '__main__':
    main()
