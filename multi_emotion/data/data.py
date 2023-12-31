import os
from pathlib import Path
import pickle
from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from multi_emotion.utils.data import data_keys


class DataModule(LightningDataModule):

    def __init__(self,
                 use_hashtag: bool = False,
                 use_senti_tree: bool = False, phrase_num: int = 0, dataset: str = None,
                 data_path: str = None, mode: str = None, num_classes: int = None,
                 train_batch_size: int = 1, eval_batch_size: int = 1, eff_train_batch_size: int = 1, num_workers: int = 0,
                 num_train: int = None, num_dev: int = None, num_test: int = None,
                 num_train_seed: int = None, num_dev_seed: int = None, num_test_seed: int = None,
                 train_shuffle: bool = False

                 ):
        super().__init__()

        self.use_hashtag = use_hashtag
        self.use_senti_tree = use_senti_tree
        self.phrase_num = phrase_num

        self.dataset = dataset
        self.data_path = data_path # ${data_dir}/${.dataset}/${model.arch}/

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.eff_train_batch_size = eff_train_batch_size
        self.num_workers = num_workers

        self.num_samples = {'train': num_train, 'dev': num_dev, 'test': num_test}
        self.num_samples_seed = {'train': num_train_seed, 'dev': num_dev_seed, 'test': num_test_seed}

        self.train_shuffle = train_shuffle


    def load_dataset(self, split):
        dataset = {}
        data_path = os.path.join(self.data_path, split)
        assert Path(data_path).exists()
        
        for key in tqdm(data_keys, desc=f'Loading {split} set'):
            if self.num_samples[split] is not None:
                filename = f'{key}_{self.num_samples[split]}_{self.num_samples_seed[split]}.pkl'
            else:
                filename = f'{key}.pkl'

            with open(os.path.join(data_path, filename), 'rb') as f:
                dataset[key] = pickle.load(f)

        return dataset

    def setup(self, splits=['all'], stage=None, dataset=None):
        self.data = {}
        if dataset is None:
            splits = ['train', 'dev', 'test'] if splits == ['all'] else splits
            for split in splits:
                dataset = self.load_dataset(split)
                self.data[split] = TextClassificationDataset(dataset, split, self.use_hashtag, self.use_senti_tree, self.phrase_num)
        else:
            self.data['pred'] = TextClassificationDataset(dataset, 'pred', self.use_hashtag, self.use_senti_tree, self.phrase_num)


    def train_dataloader(self):

        return DataLoader(
            self.data['train'],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data['train'].collater,
            shuffle=self.train_shuffle,
            pin_memory=True
        )

    def val_dataloader(self, test=False):
        if test:
            return DataLoader(
                self.data['dev'],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.data['dev'].collater,
                pin_memory=True
            )

        return [
            DataLoader(
            self.data[eval_split],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data[eval_split].collater,
            pin_memory=True)
            
            for eval_split in ['dev', 'test']
        ]

    def test_dataloader(self):
        return DataLoader(
            self.data['test'],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data['test'].collater,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data['pred'],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data['pred'].collater,
            pin_memory=True
        )

class TextClassificationDataset(Dataset):
    def __init__(self, dataset, split, use_hashtag, use_senti_tree, phrase_num):
        self.data = dataset
        self.split = split
        self.use_hashtag = use_hashtag
        self.use_senti_tree = use_senti_tree
        self.phrase_num = phrase_num

    def __len__(self):
        return len(self.data['item_idx'])

    def __getitem__(self, idx):
        item_idx = torch.LongTensor([self.data['item_idx'][idx]])
        input_ids = torch.LongTensor(self.data['input_ids'][idx])
        attention_mask = torch.LongTensor(self.data['attention_mask'][idx])

        hashtag_ids = torch.LongTensor(self.data['hashtag_ids'][idx])

        result_tuple = (item_idx, input_ids, attention_mask, hashtag_ids)


        if self.use_senti_tree:
            phrase_span_ids = torch.LongTensor(self.data['tree'][idx]['spans'])
            sentiment_ids = torch.LongTensor(self.data['tree'][idx]['sentiment'])
            if phrase_span_ids.shape[0] == 0:
                phrase_span_ids = torch.LongTensor([[0, 0]])
            if phrase_span_ids.shape[1] > self.phrase_num:
                phrase_span_ids = phrase_span_ids[:, :self.phrase_num]
                sentiment_ids = sentiment_ids[:, :self.phrase_num]
            elif phrase_span_ids.shape[1] < self.phrase_num:
                phrase_span_ids = torch.cat([phrase_span_ids, torch.LongTensor([[0, 0]] * (self.phrase_num - phrase_span_ids.shape[0]))])
                sentiment_ids = torch.cat([sentiment_ids, torch.LongTensor([0] * (self.phrase_num - sentiment_ids.shape[0]))])
            result_tuple += (phrase_span_ids, sentiment_ids)

        if 'label' in self.data:
            label = torch.LongTensor([self.data['label'][idx]])
            result_tuple += (label,)
        return result_tuple



    def collater(self, items):

        batch = {
            'item_idx': torch.cat([x[0] for x in items]),
            'input_ids': torch.stack([x[1] for x in items], dim=0),
            'attention_mask': torch.stack([x[2] for x in items], dim=0),
            'hashtag_ids': torch.stack([x[3] for x in items], dim=0) if self.use_hashtag else None,
            'phrase_span_ids': torch.stack([x[4] for x in items]) if self.use_senti_tree else None,
            'sentiment_ids': torch.stack([x[5] for x in items]) if self.use_senti_tree else None,
            'label': torch.cat([x[-1] for x in items]) if self.split != 'pred' else None,
            'split': self.split,
        }
        
        return batch