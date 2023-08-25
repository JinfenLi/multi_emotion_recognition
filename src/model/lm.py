import os, pickle, warnings
from typing import Optional, List
from timeit import default_timer as timer

import torch
from torch import nn
import torch.nn.functional as F


from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from transformers import AutoModel, AutoTokenizer

from src.model.base_model import BaseModel
from src.model.attention import TokenAttention
from src.utils.data import dataset_info, English_Hashtag_Voc_Size, Spanish_Hashtag_Voc_Size
from src.utils.losses import calc_task_loss
from src.utils.metrics import init_best_metrics, init_perf_metrics, calc_preds

from src.utils.optim import setup_optimizer_params, setup_scheduler, freeze_layers
from src.utils.logging import log_step_losses, log_epoch_losses, log_epoch_metrics
from src.utils.utils import generate_output_file


class MultiEmoModel(BaseModel):
    def __init__(self,
                 arch: str, dataset: str, optimizer: DictConfig, num_classes: int,
                 scheduler: DictConfig, num_freeze_layers: int = 0, freeze_epochs=-1, neg_weight=1,
                 save_outputs: bool = True, exp_id: str = None,
                 measure_attrs_runtime: bool = False,
                 use_hashtag: bool = True, use_senti_tree: bool = True, use_emo_cor: bool = True, hashtag_emb_dim: int = 0,
                    phrase_emb_dim: int = 0, senti_emb_dim: int = 0,
                 **kwargs):

        super().__init__()

        self.save_hyperparameters()

        self.arch = arch
        self.dataset = dataset
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.max_length = dataset_info[dataset]['max_length'][arch]

        self.scheduler = scheduler
        self.freeze_epochs = freeze_epochs
        self.neg_weight = neg_weight
        self.use_hashtag = use_hashtag
        self.use_senti_tree = use_senti_tree
        self.use_emo_cor = use_emo_cor
        self.hashtag_emb_dim = hashtag_emb_dim
        self.phrase_emb_dim = phrase_emb_dim
        self.senti_emb_dim = senti_emb_dim


        self.best_metrics = init_best_metrics()
        self.perf_metrics = init_perf_metrics(num_classes)


        self.register_buffer('empty_tensor', torch.LongTensor([]))
        # if num_classes == 2:
        #     self.register_buffer('class_weights', torch.FloatTensor([neg_weight, 1]))
        # else:
        #     self.class_weights = None

        self.tokenizer = AutoTokenizer.from_pretrained(arch)
        self.task_encoder = AutoModel.from_pretrained(arch)
        task_head_input_size = self.task_encoder.config.hidden_size

        hashtag_voc_size = English_Hashtag_Voc_Size if dataset not in ['se_spanish', 'se_arabic'] else (Spanish_Hashtag_Voc_Size if dataset == 'se_spanish' else None)
        self.hashtag_encoder = nn.Embedding(hashtag_voc_size, self.hashtag_emb_dim) if (use_hashtag and hashtag_voc_size is not None) else None
        if use_hashtag:
            task_head_input_size += self.hashtag_emb_dim

        # sentiment for each phrase node
        self.senti_encoder = nn.Embedding(5, self.senti_emb_dim) if (
                    use_senti_tree) else None
        self.phrase_encoder = nn.Linear(self.task_encoder.config.hidden_size, self.phrase_emb_dim) if use_senti_tree else None
        self.phrase_head = nn.Linear(self.phrase_emb_dim, self.phrase_emb_dim) if use_senti_tree else None
        self.tok_attn = TokenAttention(self.phrase_emb_dim, 1, self.task_encoder.config.hidden_size) if use_senti_tree else None
        if use_senti_tree:
            task_head_input_size += self.phrase_emb_dim + self.senti_emb_dim

        self.task_head = nn.Linear(
            task_head_input_size,
            num_classes
        )

        if use_emo_cor:
            self.task_head = nn.Linear(
                task_head_input_size,
                task_head_input_size
            )
            self.cor_head = nn.Bilinear(task_head_input_size, task_head_input_size, num_classes)



        assert num_freeze_layers >= 0
        if num_freeze_layers > 0:
            freeze_layers(self, num_freeze_layers)

        self.model_dict = {
            'task_encoder': self.task_encoder,
            'task_head': self.task_head,
            'hashtag_encoder': self.hashtag_encoder,
            'phrase_encoder': self.phrase_encoder,
            'phrase_head': self.phrase_head,
            'tok_attn': self.tok_attn,

        }

        if save_outputs:
            assert exp_id is not None
        self.save_outputs = save_outputs
        self.exp_id = exp_id

        self.measure_attrs_runtime = measure_attrs_runtime


    def generate_phrase_embs(self, hidden_states, phrase_ids, sentiment_ids):
        senti_enc = self.senti_encoder(sentiment_ids)
        enc = self.phrase_encoder(hidden_states)
        # enc = torch.mean(enc, dim=1)
        enc = self.phrase_head(enc)
        total_encs = []

        for i, cur_pids in enumerate(phrase_ids):
            total_enc = []
            for j, pid in enumerate(cur_pids):
                if pid[1] == 0:
                    phrase_encs = enc[i, 0].unsqueeze(0).unsqueeze(0)
                    # get neutral sentiment
                    senti_encs = self.senti_encoder.weight[2].unsqueeze(0)
                    word_encs = hidden_states[i, 0].unsqueeze(0).unsqueeze(0)
                    # print("here")
                    # print(phrase_encs.shape)
                    # print(senti_encs.shape)
                else:
                    phrase_encs = enc[i, pid[0]: pid[1]].unsqueeze(0)
                    senti_encs = senti_enc[i, j].unsqueeze(0)
                    word_encs = hidden_states[i, pid[0]: pid[1]].unsqueeze(0)
                    # print(phrase_encs.shape)
                    # print(senti_encs.shape)
                    # print(pid[1])

                cur_phrase_encs = self.tok_attn(word_encs, phrase_encs)
                total_enc.append(torch.cat([cur_phrase_encs, senti_encs], dim=1))
            cur_total_enc = torch.stack(total_enc).mean(dim=0)
            total_encs.append(cur_total_enc)
        # take the average of total_enc
        total_encs = torch.stack(total_encs).squeeze(1)
        return total_encs


    def forward(self, inputs, attention_mask, hashtag_inputs, phrase_inputs, sentiment_ids):

        enc = self.task_encoder(input_ids=inputs, attention_mask=attention_mask)
        task_head_inputs = enc.pooler_output
        if hashtag_inputs is not None:
            hashtag_enc = self.hashtag_encoder(hashtag_inputs)
            hashtag_enc = torch.mean(hashtag_enc, dim=1)
            task_head_inputs = torch.cat([task_head_inputs, hashtag_enc], dim=1)
        if phrase_inputs is not None:
            phrase_enc = self.generate_phrase_embs(enc.last_hidden_state, phrase_inputs, sentiment_ids)
            task_head_inputs = torch.cat([task_head_inputs, phrase_enc], dim=1)

        logits = self.task_head(task_head_inputs)
        if self.use_emo_cor:
            logits = self.cor_head(logits, logits)

        return logits


    def run_step(self, batch, split, batch_idx):

        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        hashtag_ids = batch['hashtag_ids'] if self.use_hashtag else None
        phrase_ids = batch['phrase_span_ids'] if self.use_senti_tree else None
        sentiment_ids = batch['sentiment_ids']
        targets = batch['label']
        eval_split: str = batch['split']
        if split == 'train':
            assert split == eval_split

        ret_dict, loss_dict, metric_dict = {}, {}, {}

        logits = self.forward(input_ids, attn_mask, hashtag_ids, phrase_ids, sentiment_ids)
        loss = calc_task_loss(logits, targets)

        loss_dict['loss'] = loss


        # Log step losses
        ret_dict = log_step_losses(self, loss_dict, ret_dict, eval_split)
        ret_dict['logits'] = logits.detach()
        ret_dict['targets'] = targets.detach()
        ret_dict['eval_split'] = eval_split
        ret_dict['input_ids'] = input_ids.detach()
        return ret_dict

    def aggregate_epoch(self, outputs, split):
        if split == 'train':
            splits = ['train']
        elif split == 'dev':
            splits = ['dev', 'test']
        elif split == 'test':
            splits = [outputs[0]['eval_split']]
        outputs_list = outputs if split == 'dev' else [outputs]

        for dataset_idx, eval_split in enumerate(splits):
            outputs = outputs_list[dataset_idx]
            log_epoch_losses(self, outputs, eval_split)  # Log epoch losses
            log_epoch_metrics(self, outputs, eval_split)  # Log epoch metrics

        # Save outputs to file            
        if self.save_outputs:
            out_dir = f'{get_original_cwd()}/save/{self.exp_id}/model_outputs/{self.dataset}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            input_ids = torch.cat([x['input_ids'] for x in outputs])
            targets = torch.cat([x['targets'] for x in outputs])
            logits = torch.cat([x['logits'] for x in outputs])
            out_data = calc_preds(logits)
            predictions = out_data.cpu().detach()
            generate_output_file(self.dataset, self.tokenizer, input_ids, targets, predictions, out_dir)


    def configure_optimizers(self):
        optimizer_params = setup_optimizer_params(self.model_dict, self.optimizer, self.use_hashtag, self.use_senti_tree)
        self.optimizer['lr'] = self.optimizer['lr'] * self.trainer.world_size
        optimizer = instantiate(
            self.optimizer, params=optimizer_params,
            _convert_="partial"
        )
        if self.scheduler.lr_scheduler == 'linear_with_warmup':
            scheduler = setup_scheduler(self.scheduler, self.total_steps, optimizer)
            return [optimizer], [scheduler]
        elif self.lr_scheduler == 'fixed':
            return [optimizer]
        else:
            raise NotImplementedError
