"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import os
import socket
import subprocess

import requests


def nrc_hashtag_lexicon(resource_path):
    hashtag_lexicon = []
    with open(os.path.join(resource_path, "NRC-Emotion-Lexicon.txt"), 'r') as file:
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


def get_CoreNLPClient(resource_dir):
    def find_available_port(start_port, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
        return None

    available_port = find_available_port(start_port=9000)

    if available_port:
        print("Available port for CoreNLPClient:", available_port)
    else:
        print("No available port  for CoreNLPClient found")
    # Define the command to start the CoreNLP server
    corenlp_command = [
        "java", "-mx16g", "-cp", f"{resource_dir}/stanford-corenlp-4.5.4/*",
        "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
        "-port", f"{available_port}", "-timeout", "60000", "-threads", "4", "-maxCharLength", "100000", "-quiet",
        "True",
        "-annotators", "tokenize,sentiment",
        "properties", "{'tokenize.codepoint': 'true'}"
    ]

    # Start the CoreNLP server
    subprocess.Popen(corenlp_command)
    return available_port

def annotate_text(text, available_port):

    corenlp_url = f"http://localhost:{available_port}"
    properties = {
        "annotators": "parse,sentiment",
        "properties": "{'tokenize.codepoint': 'true'}",
        "outputFormat": "json"
    }

    response = requests.post(f"{corenlp_url}/?properties={properties}", data=text.encode("utf-8"))
    json_response = response.json()
    return json_response
