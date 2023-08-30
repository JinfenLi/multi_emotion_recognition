"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import argparse, json, math, os, sys, random, logging
import pickle
import re

import datasets
from collections import defaultdict as ddict, Counter
import socket, subprocess
import emotlib
import nltk
import numpy as np
import pandas as pd
import torch
from nltk import TweetTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from stanfordnlp.server import CoreNLPClient

from dotenv import load_dotenv
import os

from multi_emotion.utils.utils import preprocess_dataset, sample_dataset, save_datadict

load_dotenv(override=True)
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from multi_emotion.utils.data import dataset_info, data_keys
from multi_emotion.utils.sentiTree.data_utils import get_CoreNLPClient, annotate_text, sentiment_tree
from multi_emotion.utils.utils import nrc_hashtag_lexicon, spanish_hashtag_lexicon, get_hashtag_inputs, update_dataset_dict, transform_data

os.environ["CORENLP_HOME"] = os.environ.get("CORENLP_HOME")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False







def load_datadict(data_path, split, num_samples, seed):
    dataset_dict = {}
    for key in tqdm(data_keys, desc=f'Loading {split} dataset'):
        if key in dataset_dict:
            filename = f'{key}.pkl' if num_samples is None else f'{key}_{num_samples}_{seed}.pkl'
            with open(os.path.join(data_path, filename), 'rb') as f:
                dataset_dict[key] = pickle.load(f)
    return dataset_dict




def main():



    set_random_seed(args.seed)

    assert args.split is not None and args.arch is not None
    assert args.num_samples is None or args.num_samples >= 1
    if args.dataset != 'new_data':
        split, num_examples = dataset_info[args.dataset][args.split]
        if args.num_samples is not None:
            assert args.num_samples < num_examples
            num_examples = args.num_samples
        max_length = dataset_info[args.dataset]['max_length'][args.arch]
        # num_special_tokens = dataset_info[args.dataset]['num_special_tokens']
    else:
        max_length = 128
        num_special_tokens = 2

    tokenizer = AutoTokenizer.from_pretrained(args.arch, strip_accents=False)
    data_path = os.path.join(args.data_dir, args.dataset, args.arch, args.split)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    dataset_dict = ddict(list)
    # actual_max_length = 0

    if args.dataset in ['se_english', 'new_data']:
        with open(os.path.join(args.resource_dir, "NRC-Emotion-Lexicon.txt"), 'r') as file:
            hashtag_lexicon = file.readlines()
        hashtag_vocab = nrc_hashtag_lexicon(hashtag_lexicon)
    elif args.dataset in ['se_spanish']:
        with open(os.path.join(args.resource_dir, "SEL.txt"), 'r') as file:
            hashtag_lexicon = file.readlines()
        hashtag_vocab = spanish_hashtag_lexicon(hashtag_lexicon)
    else:
        raise ValueError(f'Not identified dataset: {args.dataset}')
    # 0 represents no hashtag
    hashtag_dict = {h: i + 1 for i, h in enumerate(hashtag_vocab)}
    if args.dataset == 'se_english':
        if not os.path.exists(os.path.join(args.data_dir, args.dataset, f"se_{args.split}.pkl")):
            dataset = datasets.load_dataset('sem_eval_2018_task_1', "subtask5.english")[args.split.replace('dev', 'validation')]
            with open(os.path.join(args.data_dir, args.dataset, f"se_{args.split}.pkl"), 'wb') as f:
                pickle.dump(dataset, f)
        else:
            with open(os.path.join(args.data_dir, args.dataset, f"se_{args.split}.pkl"), 'rb') as f:
                dataset = pickle.load(f)
        # datadict
        for idx in tqdm(range(0, num_examples), desc=f'Building {args.split} dataset'):
            text = f'{dataset[idx]["Tweet"]}'
            label = [int(dataset[idx][x]) for x in dataset_info[args.dataset]['classes']]
            text = preprocess_dataset(text)
            input_ids, hashtag_inputs, offset_mapping = transform_data(tokenizer, hashtag_dict, text, max_length)
            dataset_dict = update_dataset_dict(idx, dataset_dict, input_ids, hashtag_inputs, max_length, tokenizer, text, offset_mapping,
                                               label=label)
        if args.use_senti_tree:
            dataset_dict['tree'] = sentiment_tree(dataset_dict['truncated_texts'],
                                               args.num_samples if args.num_samples else num_examples,
                                               dataset_dict['offsets'],
                                               max_length)

    elif args.dataset == 'new_data':
        df = pd.read_csv(os.path.join(args.data_dir, args.dataset, "new_data.csv"))
        args.split = 'test'
        num_examples = len(df) if args.num_samples is None else args.num_samples
        for idx in tqdm(range(0, num_examples), desc=f'Building {args.split} dataset'):
            text = f'{df["text"][idx]}'
            label = f'{df["label"][idx]}'
            text = preprocess_dataset(text)
            input_ids, hashtag_inputs, offset_mapping = transform_data(tokenizer, hashtag_dict, text, max_length)
            dataset_dict = update_dataset_dict(idx, dataset_dict, input_ids, hashtag_inputs, max_length, tokenizer, text, offset_mapping,
                                               label=label)
        if args.use_senti_tree:
            dataset_dict['tree'] = sentiment_tree(dataset_dict['truncated_texts'],
                                                 args.num_samples if args.num_samples else num_examples,
                                                 dataset_dict['offsets'],
                                                 max_length)


    if args.num_samples is not None and args.stratified_sampling:
        assert all([os.path.exists(os.path.join(data_path, f'{x}.pkl')) for x in (data_keys.remove('tree') if not args.use_senti_tree else data_keys)])
        dataset_dict = sample_dataset(data_path, dataset_dict, args.split, args.num_samples, args.seed)

    save_datadict(data_path, dataset_dict, args.split, args.num_samples, args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--data_dir', type=str, default='data/', help='Root directory for datasets')
    parser.add_argument('--resource_dir', type=str, default='resources/', help='Root directory for resources')
    parser.add_argument('--dataset', type=str, default='se_english',
                        choices=['new_data', 'se_english', 'se_arabic', 'se_spanish'])
    parser.add_argument('--arch', type=str, default='bert-base-uncased',
                        choices=['google/bigbird-roberta-base', 'bert-base-uncased'])
    parser.add_argument('--split', type=str, default='train', help='Dataset split', choices=['train', 'dev', 'test'])
    parser.add_argument('--stratified_sampling', type=bool, default=False, help='Whether to use stratified sampling')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of examples to sample. None means all available examples are used.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_senti_tree', default=False, help='Use sentiment composition')
    args = parser.parse_args()
    main()
