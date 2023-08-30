import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
def init_best_metrics():
    return {
        'best_epoch': 0,
        'dev_best_perf': None,
        'test_best_perf': None,
    }

def init_perf_metrics(num_classes):
    # if num_classes > 2:
    #     task: Literal["multiclass"] = "multiclass"
    # else:
    #     task: Literal["binary"] = "binary"
    perf_metrics = torch.nn.ModuleDict({
        'acc': BinaryAccuracy(),
        'macro_f1': BinaryF1Score(),
        # 'micro_f1': BinaryF1Score(multidim_average='samplewise'),
    })

    # assert num_classes >= 2
    return perf_metrics

def calc_preds(logits):
    probabilities = logits.sigmoid()
    # binary_labels = (probabilities > .5).int()
    return probabilities


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