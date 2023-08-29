import os
import time

import hydra
from omegaconf import DictConfig
import pyrootutils
from lightning import LightningModule

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import io
import os, shutil
import time
import uuid
from typing import Tuple, Optional
import lightning as L
import hydra
import torch
# import pytorch_lightning as pl
import yaml
from hydra.utils import instantiate, get_original_cwd
from omegaconf import open_dict, DictConfig
# from pytorch_lightning.callbacks import (
#     ModelCheckpoint, EarlyStopping
# )
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from transformers import AutoTokenizer

from src.utils.data import dataset_info, monitor_dict
from src.utils.logging import get_logger, log_hyperparameters
from src.utils.callbacks import BestPerformance

def get_callbacks(cfg: DictConfig):

    monitor = monitor_dict[cfg.data.dataset]
    mode = cfg.data.mode
    callbacks = [
        BestPerformance(monitor=monitor, mode=mode)
    ]
    # callbacks = []

    if cfg.save_checkpoint:
        callbacks.append(ModelCheckpoint(
                monitor=monitor,
                dirpath=os.path.join(cfg.paths.save_dir, 'checkpoints'),
                save_top_k=1,
                mode=mode,
                verbose=True,
                save_last=False,
                save_weights_only=True,
            )
        )


    if cfg.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                min_delta=0.00,
                patience=cfg.training.patience,
                verbose=False,
                mode=mode
            )
        )

    return callbacks


logger = get_logger(__name__)

def restore_config_params(config: DictConfig, cfg: DictConfig):
    cfg.model.arch = config['model']['arch']
    cfg.model.num_freeze_layers = config['model']['num_freeze_layers']
    cfg.model.use_hashtag = config['model']['use_hashtag']
    cfg.model.use_senti_tree = config['model']['use_senti_tree']
    cfg.model.use_emo_cor = config['model']['use_emo_cor']
    cfg.model.hashtag_emb_dim = config['model']['hashtag_emb_dim']
    cfg.model.phrase_emb_dim = config['model']['phrase_emb_dim']
    cfg.model.senti_emb_dim = config['model']['senti_emb_dim']
    cfg.model.phrase_num = config['model']['phrase_num']
    cfg.model.save_outputs = f'{cfg.paths.save_dir}/model_outputs/{cfg.data.dataset}'
    if not os.path.exists(cfg.model.save_outputs):
        os.makedirs(cfg.model.save_outputs)

    cfg.data.use_hashtag = config['model']['use_hashtag']
    cfg.data.use_senti_tree = config['model']['use_senti_tree']
    cfg.data.phrase_num = config['model']['phrase_num']


    if cfg.logger.logger == 'neptune':
        cfg.logger.tag_attrs = [cfg.data.dataset,
                                config['model']['arch'],
                                f"use_hashtag={config['model']['use_hashtag']}",
                                f"use_senti_tree={config['model']['use_senti_tree']}",
                                f"use_emo_cor={config['model']['use_emo_cor']}",
                                f"hashtag_emb_dim={config['model']['hashtag_emb_dim']}",
                                f"phrase_emb_dim={config['model']['phrase_emb_dim']}",
                                f"phrase_num={config['model']['phrase_num']}"]


    logger.info('Restored params from model config.')

    return cfg

def build(cfg) -> Tuple[LightningDataModule, LightningModule, Trainer]:

    offline_dir = f'{cfg.data.dataset}_hashtag-{cfg.model.use_hashtag}_senti-{cfg.model.use_senti_tree}_cor-{cfg.model.use_emo_cor}_{time.strftime("%d_%m_%Y")}_{str(uuid.uuid4())[: 8]}'
    config = None
    if cfg.training.evaluate_ckpt:
        cfg.paths.save_dir = os.path.join(cfg.paths.save_dir, cfg.training.exp_id)
        cfg.model.exp_id = cfg.training.exp_id
        with open(os.path.join(cfg.paths.save_dir, 'hydra', 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)

        restore_config_params(config, cfg)

        if cfg.logger.logger == "csv":
            eval_dir = os.path.join(cfg.paths.save_dir, 'eval_outputs', cfg.data.dataset)
            os.makedirs(eval_dir, exist_ok=True)
            cfg.logger.name = eval_dir
        # cfg.logger.exp_id = cfg.paths.save_dir
    else:
        if cfg.logger.logger == "csv":
            cfg.paths.save_dir = cfg.logger.name = os.path.join(cfg.paths.save_dir, offline_dir)

    dm = instantiate(
        cfg.data,
        train_shuffle=cfg.training.train_shuffle,
    )
    dm.setup(splits=cfg.training.eval_splits.split(","))

    logger.info(f'load {cfg.data.dataset} <{cfg.data._target_}>')



        # cfg.trainer.gpus = 1

    run_logger = instantiate(cfg.logger, cfg=cfg, _recursive_=False)


    with open_dict(cfg):
        if (not (cfg.debug or cfg.logger.offline)) and cfg.logger.logger == "neptune":
            cfg.paths.save_dir = os.path.join(cfg.paths.save_dir,
                                              f'{cfg.data.dataset}_hashtag-{cfg.model.use_hashtag}_senti-{cfg.model.use_senti_tree}_cor-{cfg.model.use_emo_cor}_{run_logger.experiment_id}')

    if not cfg.training.evaluate_ckpt:

        os.makedirs(cfg.paths.save_dir, exist_ok=True)
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        # copy hydra configs
        shutil.copytree(
            os.path.join(hydra_cfg['runtime']['output_dir'], ".hydra"),
            os.path.join(cfg.paths.save_dir, "hydra")
        )
        logger.info(f"saving to {cfg.paths.save_dir}")

    model = instantiate(
        cfg.model, num_classes=dataset_info[cfg.data.dataset]['num_classes'],
        _convert_="all"
    )
    logger.info(f'load {cfg.model.arch} <{cfg.model._target_}>')

    trainer = instantiate(
        cfg.trainer,
        callbacks=get_callbacks(cfg),
        # checkpoint_callback=cfg.save_checkpoint,
        logger=run_logger,
        _convert_="all",
    )
    object_dict = {
        "cfg": cfg,
        "datamodule": dm,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }
    if run_logger:
        log_hyperparameters(object_dict)
    return dm, model, trainer, config



def run(cfg: DictConfig) -> Optional[float]:
    L.seed_everything(cfg.seed)
    dm, model, trainer, config = build(cfg)
    L.seed_everything(cfg.seed)
    # from accelerate import Accelerator
    # accelerator = Accelerator()
    if cfg.save_rand_checkpoint:
        ckpt_path = os.path.join(cfg.paths.save_dir, 'checkpoints', 'rand.ckpt')
        logger.info(f"Saving randomly initialized model to {ckpt_path}")
        trainer.model = model
        trainer.save_checkpoint(ckpt_path)
    elif not cfg.training.evaluate_ckpt:
        # either train from scratch, or resume training from ckpt
        if cfg.training.finetune_ckpt:
            assert cfg.training.ckpt_path
            ckpt_path = os.path.join(cfg.paths.save_dir, "checkpoints", cfg.training.ckpt_path)
            model = model.load_from_checkpoint(ckpt_path, strict=False)
            # model = restore_config_params(model, config, cfg)
            logger.info(f"Loaded checkpoint (for fine-tuning) from {ckpt_path}")

        trainer.fit(model=model, datamodule=dm)

        if getattr(cfg, "tune_metric", None):
            metric = trainer.callback_metrics[cfg.tune_metric].detach()
            logger.info(f"best metric {metric}")
            return metric
    else:
        # evaluate the pretrained model on the provided splits
        assert cfg.training.ckpt_path

        ckpt_path = os.path.join(cfg.paths.save_dir, "checkpoints", cfg.training.ckpt_path)
        buffer = io.BytesIO()
        torch.save(ckpt_path, buffer)
        buffer.seek(0)  # Reset the buffer position to the beginning
        checkpoint = torch.load(buffer)
        model = model.load_from_checkpoint(checkpoint, strict=False)
        logger.info(f"Loaded checkpoint for evaluation from {cfg.training.ckpt_path}")
        # model = restore_config_params(model, config, cfg)
        model.exp_id = cfg.training.exp_id
        model.save_outputs = True
        print('Evaluating loaded model checkpoint...')
        for split in cfg.training.eval_splits.split(','):
            print(f'Evaluating on split: {split}')
            if split == 'train':
                loader = dm.train_dataloader()
            elif split == 'dev':
                loader = dm.val_dataloader(test=False)
            elif split == 'test':
                loader = dm.test_dataloader()

            trainer.test(model=model, dataloaders=loader)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # import here for faster auto completion
    from src.utils.conf import touch

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
