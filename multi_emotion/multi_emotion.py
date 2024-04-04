"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import io
import logging
import os
import tarfile
import torch
import yaml
from lightning.pytorch import Trainer
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict as ddict

from .model.lm import MultiEmoModel
from .data.data import DataModule
from .utils.sentiTree import data_utils
from .utils.utils import nrc_hashtag_lexicon, preprocess_dataset, transform_data, update_dataset_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_configuration():
    path_to_exist = os.path.join(os.path.dirname(__file__), f"model_dependencies")
    os.makedirs(path_to_exist, exist_ok=True)
    if not os.path.exists(os.path.join(path_to_exist, "multi-emo-bert.tar.gz")):
        logger.info(f"Downloading model configuration files to {path_to_exist} ...")
        torch.hub.download_url_to_file("https://pytorch-libs.s3.us-east-2.amazonaws.com/multi-emo-bert.tar.gz",
                                       os.path.join(path_to_exist, "multi-emo-bert.tar.gz"))

        with tarfile.open(os.path.join(path_to_exist, "multi-emo-bert.tar.gz"), 'r:gz') as tar:
            tar.extractall(path=path_to_exist)
    with open(os.path.join(path_to_exist, "multi-emo-bert", 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    with open(os.path.join(path_to_exist, "multi-emo-bert", 'hashtag_lexicon.txt'), 'r') as f:
        hashtag_lexicon = f.readlines()

    ckpt_path = os.path.join(path_to_exist, "multi-emo-bert", 'pytorch_model.ckpt')
    return config, hashtag_lexicon, ckpt_path


def convert_text_to_features(texts, tokenizer, max_length, hashtag_dict, use_senti_tree=False):
    """
    convert text to features
    Args:
        texts: text to convert
        tokenizer: tokenizer
        max_length: max length of text
        use_senti_tree: whether to use senti tree
        labels: labels can be None or List[List[]]
    """
    dataset_dict = ddict(list)
    for idx in tqdm(range(0, len(texts)), desc=f'Building dataset'):

        label = None
        text = preprocess_dataset(texts[idx])
        input_ids, hashtag_inputs, offset_mapping = transform_data(tokenizer, hashtag_dict, text, max_length)
        dataset_dict = update_dataset_dict(idx, dataset_dict, input_ids, hashtag_inputs, max_length, tokenizer, text, offset_mapping,
                                           label)
    if use_senti_tree:
        dataset_dict['tree'] = data_utils.sentiment_tree(dataset_dict['truncated_texts'],
                                              len(texts),
                                              dataset_dict['offsets'],
                                              max_length)
    return dataset_dict

def load_checkpoint(ckpt_path):

    buffer = io.BytesIO()
    torch.save(ckpt_path, buffer)
    buffer.seek(0)  # Reset the buffer position to the beginning
    checkpoint = torch.load(buffer)
    model = MultiEmoModel.load_from_checkpoint(checkpoint, strict=False, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
    logger.info(f"Loaded checkpoint for evaluation from {ckpt_path}")
    return model

def predict(texts: list):
    """

    Args:
        texts: a list of texts

    Returns:
        [{text: [text1, text2, ...], label: [emotion1, emotion2, ...], probability: [{anger: 0.1}, {anticipation: 0.2}, {...}]}]

    """
    assert isinstance(texts, list), "texts must be a list of texts"
    config, hashtag_lexicon, ckpt_path = get_configuration()
    max_length = config['max_length']
    arch = config['arch']
    use_hashtag = config['use_hashtag']
    use_senti_tree = config['use_senti_tree']
    phrase_num = config['phrase_num']
    use_emo_cor = config['use_emo_cor']
    hashtag_emb_dim = config['hashtag_emb_dim']
    phrase_emb_dim = config['phrase_emb_dim']
    senti_emb_dim = config['senti_emb_dim']
    num_classes = config['num_classes']

    tokenizer = AutoTokenizer.from_pretrained(arch, strip_accents=False)
    hashtag_vocab = nrc_hashtag_lexicon(hashtag_lexicon)
    hashtag_dict = {h: i + 1 for i, h in enumerate(hashtag_vocab)}
    dataset_dict = convert_text_to_features(texts, tokenizer, max_length, hashtag_dict, use_senti_tree=use_senti_tree)
    dm = DataModule(use_hashtag, use_senti_tree, phrase_num)
    dm.setup(dataset=dataset_dict)
    loader = dm.predict_dataloader()

    model = MultiEmoModel(arch, use_hashtag, use_senti_tree, use_emo_cor,
                 hashtag_emb_dim,
                 phrase_emb_dim, senti_emb_dim, max_length, num_classes)
    model = load_checkpoint(ckpt_path)

    trainer = Trainer(logger=False)
    trainer.predict(model=model, dataloaders=loader)
    test_results = trainer.lightning_module.results
    return test_results

