"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import os


def nrc_hashtag_lexicon(resource_path):
    hashtag_lexicon = []
    with open(os.path.join(resource_path, "NRC-Hashtag-Emotion-Lexicon-v0.2.txt"), 'r') as file:
        nrc_lexicon = file.readlines()
        for nl in nrc_lexicon:
            hashtag_lexicon.append(nl.split('\t')[1].replace("#", ""))
    return hashtag_lexicon


def spanish_hashtag_lexicon(resource_path):
    hashtag_lexicon = []
    with open(os.path.join(resource_path, "SEL.txt"), 'r') as file:
        emo_lexicon = file.readlines()
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


