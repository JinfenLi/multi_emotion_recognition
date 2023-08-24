"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import io
import os, shutil
import time
import uuid
from typing import Tuple, Optional

import torch
import pytorch_lightning as pl
import yaml
from hydra.utils import instantiate
from omegaconf import open_dict, DictConfig
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping
)
from transformers import AutoTokenizer

from multi_emotion_recognition.utils.data import dataset_info, monitor_dict
from multi_emotion_recognition.utils.logging import get_logger
from multi_emotion_recognition.utils.callbacks import BestPerformance

def get_callbacks(cfg: DictConfig):

    monitor = monitor_dict[cfg.data.dataset]
    mode = cfg.data.mode
    callbacks = [
        BestPerformance(monitor=monitor, mode=mode)
    ]

    if cfg.save_checkpoint:
        callbacks.append(
            ModelCheckpoint(
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


def build(cfg) -> Tuple[pl.LightningDataModule, pl.LightningModule, pl.Trainer]:
    dm = instantiate(
        cfg.data,
        train_shuffle=cfg.training.train_shuffle,
    )
    dm.setup(splits=cfg.training.eval_splits.split(","))

    offline_dir = f'{cfg.data.dataset}_hashtag-{cfg.model.use_hashtag}_senti-{cfg.model.use_senti_tree}_cor-{cfg.model.use_emo_cor}_{time.strftime("%d_%m_%Y")}_{str(uuid.uuid4())[: 8]}'
    logger.info(f'load {cfg.data.dataset} <{cfg.data._target_}>')
    config = None
    if cfg.training.evaluate_ckpt:
        cfg.paths.save_dir = os.path.join(cfg.paths.save_dir, cfg.training.exp_id)
        cfg.model.exp_id = cfg.training.exp_id
        if cfg.logger.logger == "neptune":
            with open(os.path.join(cfg.paths.save_dir, 'hydra', 'config.yaml'), 'r') as f:
                config = yaml.safe_load(f)
            cfg.logger.tag_attrs = [cfg.data.dataset,
                                    config['model']['arch'],
                                    f"use_hashtag={config['model']['use_hashtag']}",
                                    f"use_senti_tree={config['model']['use_senti_tree']}",
                                    f"use_emo_cor={config['model']['use_emo_cor']}",
                                    f"hashtag_emb_dim={config['model']['hashtag_emb_dim']}",
                                    f"phrase_emb_dim={config['model']['phrase_emb_dim']}"]

        if cfg.logger.logger == "csv":
            cfg.logger.name = cfg.paths.save_dir
        # cfg.logger.exp_id = cfg.paths.save_dir
    else:
        if cfg.logger.logger == "csv":
            cfg.paths.save_dir = cfg.logger.name = os.path.join(cfg.paths.save_dir, offline_dir)


        # cfg.trainer.gpus = 1

    run_logger = instantiate(cfg.logger, cfg=cfg, _recursive_=False)

    with open_dict(cfg):
        if (not (cfg.debug or cfg.logger.offline)) and cfg.logger.logger == "neptune":
            cfg.paths.save_dir = os.path.join(cfg.paths.save_dir,
                                              f'{cfg.data.dataset}_hashtag-{cfg.model.use_hashtag}_senti-{cfg.model.use_senti_tree}_cor-{cfg.model.use_emo_cor}_{run_logger.experiment_id}')
        #     if cfg.logger.logger == 'csv':
        #
        #         cfg.logger.name = cfg.logger.exp_id = offline_dir
        # else:
        #     save_dir = f'{cfg.data.dataset}_{cfg.model.arch}_use-hashtag-{cfg.model.use_hashtag}_use-senti-tree-{cfg.model.use_senti_tree}_{run_logger.experiment_id}'
        #     if cfg.logger.logger == "neptune":
        #         cfg.logger.exp_id = save_dir
        #     else:
        #         raise NotImplementedError
    if not cfg.training.evaluate_ckpt:
        # cfg.paths.save_dir = os.path.join(cfg.paths.save_dir,
        #                                   cfg.logger.exp_id)
        os.makedirs(cfg.paths.save_dir, exist_ok=True)
        # copy hydra configs
        shutil.copytree(
            os.path.join(os.getcwd(), ".hydra"),
            os.path.join(cfg.paths.save_dir, "hydra")
        )
        logger.info(f"saving to {cfg.paths.save_dir}")

    model = instantiate(
        cfg.model, num_classes=dataset_info[cfg.data.dataset]['num_classes'],
        _recursive_=False
    )
    logger.info(f'load {cfg.model.arch} <{cfg.model._target_}>')

    trainer = instantiate(
        cfg.trainer,
        callbacks=get_callbacks(cfg),
        checkpoint_callback=cfg.save_checkpoint,
        logger=run_logger,
        _convert_="all",
    )

    return dm, model, trainer, config



def run(cfg: DictConfig) -> Optional[float]:
    pl.seed_everything(cfg.seed)
    dm, model, trainer, config = build(cfg)
    pl.seed_everything(cfg.seed)
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
                loader = dm.val_dataloader(test=True)
            elif split == 'test':
                loader = dm.test_dataloader()

            trainer.test(model=model, dataloaders=loader)