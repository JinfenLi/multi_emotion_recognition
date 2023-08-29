from typing import Optional

from lightning import LightningModule
import torch
from itertools import chain

class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()
        # update in `setup`
        self.total_steps = None
        self.training_step_outputs = []
        self.validation_step_outputs = [[], []]
        self.test_step_outputs = []
        self.predict_step_outputs = []
        self.results = None


    def forward(self, **kwargs):
        raise NotImplementedError

    def calc_loss(self, preds, targets):
        raise NotImplementedError

    def calc_acc(self, preds, targets):
        raise NotImplementedError

    def run_step(self, batch, split, batch_idx):
        raise NotImplementedError

    def aggregate_epoch(self, outputs, split):
        raise NotImplementedError

    def collater(self, outputs):
        total_outputs = {
            'loss': torch.stack([x['loss'] for x in outputs]) if 'loss' in outputs[0] else None,
            'probs': torch.cat([x['probs'] for x in outputs]),
            'targets': torch.cat([x['targets'] for x in outputs]) if outputs[0]['targets'] is not None else None,
            'eval_split': [x['eval_split'] for x in outputs],
            'input_ids': torch.cat([x['input_ids'] for x in outputs]),

        }

        return total_outputs

    def training_step(self, batch, batch_idx):
        # # freeze encoder for initial few epochs based on p.freeze_epochs
        # if self.current_epoch < self.freeze_epochs:
        # 	freeze_net(self.text_encoder)
        # else:
        # 	unfreeze_net(self.text_encoder)
        # print("training step")
        ret_dict = self.run_step(batch, 'train', batch_idx)
        # loss = ret_dict['loss']
        self.training_step_outputs.append(ret_dict)
        return ret_dict


    def on_train_epoch_end(self):
        training_step_outputs = self.collater(self.training_step_outputs)
        self.aggregate_epoch(training_step_outputs, 'train')
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # print("validation_step step")
        assert dataloader_idx in [0, 1]
        eval_splits = {0: 'dev', 1: 'test'}
        ret_dict = self.run_step(batch, eval_splits[dataloader_idx], batch_idx)
        self.validation_step_outputs[dataloader_idx].append(ret_dict)
        return ret_dict

    def on_validation_epoch_end(self):
        for dl_idx in range(len(self.validation_step_outputs)):
            validation_step_outputs = self.collater(self.validation_step_outputs[dl_idx])
            self.aggregate_epoch(validation_step_outputs, 'dev')
            self.validation_step_outputs[dl_idx].clear()

        monitor = self.trainer.callbacks[0].monitor
        test_monitor = self.trainer.callbacks[0].test_monitor
        if self.trainer.callbacks[0].mode == 'max':
            if self.best_metrics['dev_best_perf'] == None:
                assert self.best_metrics['test_best_perf'] == None
                self.best_metrics['dev_best_perf'] = -float('inf')

            if self.trainer.callback_metrics[monitor] > self.best_metrics['dev_best_perf']:
                self.best_metrics['dev_best_perf'] = self.trainer.callback_metrics[monitor]
                self.best_metrics['test_best_perf'] = self.trainer.callback_metrics[test_monitor]
                self.best_metrics['best_epoch'] = self.trainer.current_epoch
        else:
            if self.best_metrics['dev_best_perf'] == None:
                assert self.best_metrics['test_best_perf'] == None
                self.best_metrics['dev_best_perf'] = float('inf')

            if self.trainer.callback_metrics[monitor] < self.best_metrics['dev_best_perf']:
                self.best_metrics['dev_best_perf'] = self.trainer.callback_metrics[monitor]
                self.best_metrics['test_best_perf'] = self.trainer.callback_metrics[test_monitor]
                self.best_metrics['best_epoch'] = self.trainer.current_epoch

        self.log('dev_best_perf', self.best_metrics['dev_best_perf'], prog_bar=True, sync_dist=True)
        self.log('test_best_perf', self.best_metrics['test_best_perf'], prog_bar=True, sync_dist=True)
        self.log('best_epoch', self.best_metrics['best_epoch'], prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self.run_step(batch, 'test', batch_idx))
        return self.run_step(batch, 'test', batch_idx)

    def on_test_epoch_end(self):
        test_step_outputs = self.collater(self.test_step_outputs)
        self.aggregate_epoch(test_step_outputs, 'test')
        self.test_step_outputs.clear()
        # self.results = results
        # return results

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.predict_step_outputs.append(self.run_step(batch, 'pred', batch_idx))
        return self.run_step(batch, 'pred', batch_idx)

    def on_predict_epoch_end(self):
        predict_step_outputs = self.collater(self.predict_step_outputs)
        results = self.aggregate_epoch(predict_step_outputs, 'pred')
        self.predict_step_outputs.clear()
        self.results = results
        return results

    def setup(self, stage: Optional[str] = None):
        """calculate total steps"""
        if stage == 'fit':
            # Get train dataloader
            train_loader = self.trainer.datamodule.train_dataloader()
            ngpus = self.trainer.num_devices

            # Calculate total steps
            eff_train_batch_size = (self.trainer.datamodule.train_batch_size *
                                    max(1, ngpus) * self.trainer.accumulate_grad_batches)
            # assert eff_train_batch_size == self.trainer.datamodule.eff_train_batch_size
            self.total_steps = int(
                (len(train_loader.dataset) // eff_train_batch_size) * float(self.trainer.max_epochs))

    def configure_optimizers(self):
        raise NotImplementedError