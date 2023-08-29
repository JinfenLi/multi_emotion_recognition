import os
import getpass, logging, socket
from typing import Any, List
import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from lightning.pytorch.loggers import NeptuneLogger, CSVLogger
from src.utils.metrics import calc_preds, get_step_metrics, get_epoch_metrics
from dotenv import load_dotenv
import logging
load_dotenv(override=True)

# API_LIST = {
#     "neptune": {
#         'api-key': os.environ.get('NEPTUNE_API_TOKEN'),
#         'name': os.environ.get('NEPTUNE_NAME'),
#     },
# }
log = logging.getLogger(__name__)

def get_username():
    return getpass.getuser()

def flatten_cfg(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        ret = {}
        for k, v in cfg.items():
            flatten: dict = flatten_cfg(v)
            ret.update({
                f"{k}/{f}" if f else k: fv
                for f, fv in flatten.items()
            })
        return ret
    return {"": cfg}

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def get_csv_logger(
    cfg: DictConfig, logger:str, save_dir: str, name: str, offline: bool
):
    csv_logger = CSVLogger(
        save_dir=save_dir,
        name=name
    )

    return csv_logger


def get_neptune_logger(
    cfg: DictConfig,
    tag_attrs: List[str], log_db: str,
    api_key: str, project_name: str,
    offline: bool, logger: str,
):
    # neptune_api_key = API_LIST["neptune"]["api-key"]
    # name = API_LIST["neptune"]["name"]
    # flatten cfg
    args_dict = {
        **flatten_cfg(OmegaConf.to_object(cfg)),
        "hostname": socket.gethostname()
    }
    tags = tag_attrs
    if cfg.model.expl_reg:
        tags.append('expl_reg')

    tags.append(log_db)

    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project_name=project_name,
        # name="Prediction model",
        params=args_dict,
        tags=tags,
        offline_mode=offline,
    )

    try:
        # for unknown reason, must access this field otherwise becomes None
        print(neptune_logger.experiment)
    except BaseException:
        pass

    return neptune_logger

def log_data_to_neptune(model_class, data, data_name, data_type, suffix, split, ret_dict=None, detach_data=True):

    data_key = 'loss' if f'{data_name}_{data_type}' == 'total_loss' else f'{data_name}_{data_type}'
    model_class.log(f'{split}_{data_key}_{suffix}', data.detach(), prog_bar=True, sync_dist=(split != 'train'))
    if ret_dict is not None:
        ret_dict[data_key] = data.detach() if detach_data else data
    
    return ret_dict

def log_step_losses(model_class, loss_dict, ret_dict, split):
    ret_dict = log_data_to_neptune(model_class, loss_dict['loss'], 'total', 'loss', 'step', split, ret_dict, detach_data=False)
    return ret_dict


def log_epoch_losses(model_class, outputs, split):
    loss = outputs['loss'].mean()
    log_data_to_neptune(model_class, loss, 'total', 'loss', 'epoch', split, ret_dict=None)

def log_epoch_metrics(model_class, outputs, split):
    probs = outputs['probs']
    targets = outputs['targets']
    # preds = calc_preds(logits)
    get_step_metrics(probs, targets, model_class.perf_metrics)
    perf_metrics = get_epoch_metrics(model_class.perf_metrics)

    log_data_to_neptune(model_class, perf_metrics['acc'], 'acc', 'metric', 'epoch', split, ret_dict=None)
    log_data_to_neptune(model_class, perf_metrics['macro_f1'], 'macro_f1', 'metric', 'epoch', split, ret_dict=None)
    # log_data_to_neptune(model_class, perf_metrics['micro_f1'], 'micro_f1', 'metric', 'epoch', split, ret_dict=None)
    if model_class.num_classes == 2:
        log_data_to_neptune(model_class, perf_metrics['binary_f1'], 'binary_f1', 'metric', 'epoch', split, ret_dict=None)


    if 'delta' in outputs.keys():
        delta = torch.abs(outputs['delta']).mean()
        log_data_to_neptune(model_class, delta, 'convergence_delta', 'metric', 'epoch', split, ret_dict=None)

@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)