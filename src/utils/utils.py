"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import json
import os
import pickle
import random
from collections import Counter

import emotlib
import numpy as np
from tqdm import tqdm

from src.utils.data import dataset_info, data_keys
import pandas as pd

def nrc_hashtag_lexicon(nrc_lexicon):
    hashtag_vocab = []
    for nl in nrc_lexicon:
        hashtag_vocab.append(nl.split('\t')[1].replace("#", ""))
    return hashtag_vocab


def spanish_hashtag_lexicon(emo_lexicon):
    hashtag_lexicon = []
    for el in emo_lexicon[1:]:
        hashtag_lexicon.append(el.split('\t')[0])
    return hashtag_lexicon

def get_hashtag_inputs(tokens, hashtag_dict):
    hashtag_inputs = []
    cur_word = []
    tid = 0
    while tid < len(tokens):
        if tokens[tid] in ['[PAD]', '[SEP]']:
            hashtag_inputs.extend([0] * (len(tokens) - len(hashtag_inputs)))
            break
        if tid < len(tokens) and tokens[tid].startswith('##'):
            while tid < len(tokens) and tokens[tid].startswith('##'):
                cur_word.append(tokens[tid][2:])
                tid += 1
            # let tid point to the last token of the word
            tid -= 1
        else:
            cur_word = [tokens[tid]]


        if ''.join(cur_word) in hashtag_dict:
            hashtag_id = hashtag_dict[''.join(cur_word)]
            # the hashtags of word: #, xx, ##xx, ##xx are all 1
            if 0 < (tid - len(cur_word)) < len(tokens) and tokens[tid - len(cur_word)] == "#":
                hashtag_inputs[-1] = hashtag_id
            hashtag_inputs.extend([hashtag_id] * len(cur_word))
        else:
            hashtag_inputs.extend([0] * len(cur_word))

        tid += 1
    return hashtag_inputs


def update_dataset_dict(
    idx, dataset_dict, input_ids, hashtag_inputs, max_length, tokenizer, text, token_offsets, label=None):
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    hashtag_inputs = [0] + hashtag_inputs + [0]
    num_tokens = len(input_ids)
    truncated_pos = token_offsets[-1][1] if text else 0
    # if num_tokens > actual_max_length:
    #     actual_max_length = num_tokens

    num_pad_tokens = max_length - num_tokens
    assert num_pad_tokens >= 0

    input_ids += [tokenizer.pad_token_id] * num_pad_tokens
    attention_mask = [1] * num_tokens + [0] * num_pad_tokens
    hashtag_inputs += [0] * num_pad_tokens
    if token_offsets is not None:
        token_offsets = [(0, 0)] + token_offsets + [(0, 0)]
        token_offsets += [(0, 0)] * num_pad_tokens


    dataset_dict['item_idx'].append(idx)
    dataset_dict['input_ids'].append(input_ids)
    dataset_dict['hashtag_ids'].append(hashtag_inputs)
    dataset_dict['attention_mask'].append(attention_mask)
    if label is not None:
        dataset_dict['label'].append(label)
    dataset_dict['offsets'].append(token_offsets)
    dataset_dict['truncated_texts'].append(text[: truncated_pos])

    return dataset_dict

def preprocess_dataset(text):
    text = text.replace("\\n", " ")
    text = text.lower()
    text = emotlib.demojify(text)
    return text

def transform_data(tokenizer, hashtag_dict, text, max_length):
    encode_dict = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    raw_tokens = encode_dict.encodings[0].tokens
    hashtag_inputs = get_hashtag_inputs(raw_tokens, hashtag_dict)
    # num_tokens = encode_dict['attention_mask'].count(1)
    input_ids = encode_dict['input_ids']
    offset_mapping = encode_dict['offset_mapping']
    # if num_tokens > actual_max_length:
    #     actual_max_length = num_tokens

    input_length = min(len(input_ids), max_length - 2)
    input_ids = input_ids[:input_length]
    offset_mapping = offset_mapping[:input_length]
    hashtag_inputs = hashtag_inputs[:input_length]
    return input_ids, hashtag_inputs, offset_mapping

def generate_output_file(dataset, tokenizer, input_ids, probabilities, output_dir=None, targets=None):
    texts = []
    for i in range(len(input_ids)):
        text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        texts.append(text)
    labels = [','.join([dataset_info[dataset]['classes'][i] for i, pp in enumerate(p) if pp > 0.5]) for p in probabilities]
    labels = ['neutral' if len(l) == 0 else l for l in labels]
    probs = [json.dumps([{dataset_info[dataset]['classes'][i]: pp.item()} for i, pp in enumerate(p)]) for p in probabilities]
    results = [{'text': t, 'pred_label': l, 'probability': p} for t, l, p in zip(texts, labels, probs)]
    result_df = pd.DataFrame({'text': texts, 'prediction': labels, 'prob': probs})
    if targets is not None:
        targets = [','.join([dataset_info[dataset]['classes'][i] for i, tt in enumerate(t) if tt != 0]) for t in
                   targets]
        result_df['target'] = targets
    if output_dir:
        result_df.to_csv(os.path.join(output_dir, 'result.csv'), index=False)
    return results

def stratified_sampling(data, num_samples):
    num_instances = len(data)
    assert num_samples < num_instances

    counter_dict = Counter(data)
    unique_vals = list(counter_dict.keys())
    val_counts = list(counter_dict.values())
    num_unique_vals = len(unique_vals)
    assert num_unique_vals > 1

    num_stratified_samples = [int(c*num_samples/num_instances) for c in val_counts]
    assert sum(num_stratified_samples) <= num_samples
    if sum(num_stratified_samples) < num_samples:
        delta = num_samples - sum(num_stratified_samples)
        delta_samples = np.random.choice(range(num_unique_vals), replace=True, size=delta)
        for val in delta_samples:
            num_stratified_samples[unique_vals.index(val)] += 1
    assert sum(num_stratified_samples) == num_samples

    sampled_indices = []
    for i, val in enumerate(unique_vals):
        candidates = np.where(data == val)[0]
        sampled_indices += list(np.random.choice(candidates, replace=False, size=num_stratified_samples[i]))
    random.shuffle(sampled_indices)

    return sampled_indices
def sample_dataset(data_path, dataset_dict, split, num_samples, seed):
    sampled_split_filename = f'{split}_split_{num_samples}_{seed}.pkl'
    if os.path.exists(os.path.join(data_path, sampled_split_filename)):
        with open(os.path.join(data_path, sampled_split_filename), 'rb') as f:
            sampled_split = pickle.load(f)
    else:
        sampled_split = stratified_sampling(dataset_dict['label'], num_samples)
        with open(os.path.join(data_path, sampled_split_filename), 'wb') as f:
            pickle.dump(sampled_split, f)

    for key in data_keys:
        dataset_dict[key] = sampled_split if key == 'item_idx' else [dataset_dict[key][i] for i in sampled_split]

    return dataset_dict

def save_datadict(data_path, dataset_dict, split, num_samples, seed):
    for key in tqdm(data_keys, desc=f'Saving {split} dataset'):
        if key in dataset_dict:
            filename = f'{key}.pkl' if num_samples is None else f'{key}_{num_samples}_{seed}.pkl'
            with open(os.path.join(data_path, filename), 'wb') as f:
                pickle.dump(dataset_dict[key], f)