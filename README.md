# Multi-emotion Recognition Using Multi-EmoBERT and Emotion Analysis in Fake News

This is the official PyTorch repo for [Multi-EmoBERT](https://dl.acm.org/doi/abs/10.1145/3578503.3583595), a learning framework for multi-emotion recognition.

```
Multi-emotion Recognition Using Multi-EmoBERT and Emotion Analysis in Fake News
Jinfen Li, Lu Xiao
WebSci 2023
```


If Multi-EmoBERT is helpful for your research, please consider citing our paper:

```
@inproceedings{chan2022unirex,
  title={Unirex: A unified learning framework for language model rationale extraction},
  author={Chan, Aaron and Sanjabi, Maziar and Mathias, Lambert and Tan, Liang and Nie, Shaoliang and Peng, Xiaochang and Ren, Xiang and Firooz, Hamed},
  booktitle={International Conference on Machine Learning},
  pages={2867--2889},
  year={2022},
  organization={PMLR}
}
```

## Basics
### Resources
[Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/download.html)
[NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.html)
[Spanish Hashtag Lexicon](https://www.cic.ipn.mx/~sidorov/#SEL)
### Neptune
Before running the code, you need to complete the following steps:
1. Create a [Neptune](https://neptune.ai/) account and project.
2. Edit the [project name](https://github.com/aarzchan/UNIREX/blob/main/configs/logger/neptune.yaml#L12), [local username](https://github.com/aarzchan/UNIREX/blob/main/src/utils/logging.py#L11), and [Neptune API token](https://github.com/aarzchan/UNIREX/blob/main/src/utils/logging.py#L11) fields in the code.


### Multirun
Do grid search over different configs.
```
python main.py -m \
    dataset=se_english \
    seed=0,1,2,3,4,5 \
```

### Evaluate checkpoint
This command evaluates a checkpoint on the train, dev, and test sets.
```
python main.py \
    training=evaluate \
    training.ckpt_path=/path/to/ckpt \
    training.eval_splits=train,dev,test \
```

### Finetune checkpoint
```
python main.py \
    training=evaluate \
    training.ckpt_path=/path/to/ckpt \
```

### Offline Mode
In offline mode, results are not logged to Neptune.
```
python main.py logger.offline=True
```

### Debug Mode
In debug mode, results are not logged to Neptune, and we only train/evaluate for limited number of batches and/or epochs.
```
python main.py debug=True
```

### Hydra Working Directory

Hydra will change the working directory to the path specified in `configs/hydra/default.yaml`. Therefore, if you save a file to the path `'./file.txt'`, it will actually save the file to somewhere like `logs/runs/xxxx/file.txt`. This is helpful when you want to version control your saved files, but not if you want to save to a global directory. There are two methods to get the "actual" working directory:

1. Use `hydra.utils.get_original_cwd` function call
2. Use `cfg.work_dir`. To use this in the config, can do something like `"${data_dir}/${.dataset}/${model.arch}/"`


### Config Key

- `work_dir` current working directory (where `src/` is)

- `data_dir` where data folder is

- `log_dir` where log folder is (runs & multirun)

- `root_dir` where the saved ckpt & hydra config are


---


## Example Commands

Here, we assume the following:
- The `data_dir` is `data`, which means `data_dir=${work_dir}/../data`.
- The dataset is `semEval 2018 task 1-english`.

### 1. Build dataset
The commands below are used to build pre-processed datasets, saved as pickle files. The model architecture is specified so that we can use the correct tokenizer for pre-processing.

```
python scripts/build_dataset.py --data_dir data \
    --dataset se_english --arch bert-base-uncased --split train

python scripts/build_dataset.py --data_dir data \
    --dataset se_english --arch bert-base-uncased --split dev

python scripts/build_dataset.py --data_dir data \
    --dataset se_english --arch bert-base-uncased --split test

```

If the dataset is very large, you have the option to subsample part of the dataset for smaller-scale experiements. For example, in the command below, we build a train set with only 1000 train examples (sampled with seed 0).
```
python scripts/build_dataset.py --data_dir data \
    --dataset se_english --arch bert-base-uncased --split train \
    --num_samples 1000 --seed 0
```

### 2. Train Multi-EmoBERT

The command below is the most basic way to run `main.py`

```
python main.py -m \
    data=se_english \
    model=lm \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

### 3. Train Model with Hashtag Encoding, Sentiment Composition and Emotion Correlation
This repo implements a number of different methods for training the Task LM. Below are commands for running each method.


**Task LM + Hashtag Encoding**
```
python main.py -m \
    data=se_english \
    model=lm \
    model.use_hashtag=True \
    model.hashtag_emb_dim=80 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Task LM + Sentiment Composition**

```
python main.py -m \
    data=se_english \
    model=lm \
    model.use_senti_tree=True \
    model.phrase_emb_dim=80 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Task LM + Emotion Correlation**

```
python main.py -m \
    data=se_english \
    model=lm \
    model.use_emo_cor=True \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

### 4. Evaluate Model
exp_id is the folder name under your save_dir (e.g., "se_english_bert-base-uncased_use-hashtag-True_use-senti-tree-True_xxx"), ckpt_path is the checkpoint under the checkpoints folder in the exp_id folder
```
python main.py -m \
    data=se_english \
    training=evaluate \
    ckpt_path = xxx \
    exp_id = xxx \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```