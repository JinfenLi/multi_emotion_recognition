import os
import time

import hydra
from omegaconf import DictConfig
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # import here for faster auto completion
    from src.utils.conf import touch
    from src.run import run

    # additional set field by condition
    # assert no missing etc
    touch(cfg)

    start_time = time.time()
    metric = run(cfg)
    print(
        f'Time Taken for experiment {cfg.paths.save_dir}: {(time.time() - start_time) / 3600}h')

    return metric


if __name__ == '__main__':
    __spec__ = None
    main()
