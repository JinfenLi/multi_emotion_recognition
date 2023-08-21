from typing import Literal
import torch
import torchmetrics
import torch.nn.functional as F

def init_best_metrics():
    return {
        'best_epoch': 0,
        'dev_best_perf': None,
        'test_best_perf': None,
    }

def init_perf_metrics(num_classes):
    if num_classes > 2:
        task: Literal["multiclass"] = "multiclass"
    else:
        task: Literal["binary"] = "binary"
    perf_metrics = torch.nn.ModuleDict({
        'acc': torchmetrics.Accuracy( num_classes=num_classes),
        'macro_f1': torchmetrics.F1(num_classes=num_classes, average='macro'),
        'micro_f1': torchmetrics.F1(num_classes=num_classes, average='micro'),
    })

    assert num_classes >= 2
    return perf_metrics

def calc_preds(logits):
    probabilities = F.softmax(logits, dim=1)
    binary_labels = (probabilities > .5).int()
    return binary_labels


def get_step_metrics(preds, targets, metrics):
    res = {}
    for key, metric_fn in metrics.items():
        res.update({key: metric_fn(preds, targets) * 100})
    return res

def get_epoch_metrics(metrics):
    res = {}
    for key, metric_fn in metrics.items():
        res.update({key: metric_fn.compute() * 100})
        metric_fn.reset()
    return res