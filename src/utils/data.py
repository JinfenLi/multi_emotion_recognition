English_Hashtag_Voc_Size = 32389 + 1
# Arabic_Hashtag_Voc_Size = 100
Spanish_Hashtag_Voc_Size = 2037 + 1

dataset_info = {
    'se_english': {
        'train': ['train', 6838],
        'dev': ['dev', 886],
        'test': ['test', 3259],
        'num_classes': 11,
        'classes': ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness",
                    "surprise", "trust"],
        'max_length': {
            'bert-base-uncased': 128,
        },
        'num_special_tokens': 2,
    }
}

benchmark_datasets = ['se_english', 'se_arabic', 'se_spanish']

monitor_dict = {
    'se_english': 'dev_micro_f1_metric_epoch',
    'se_arabic': 'dev_micro_f1_metric_epoch',
    'se_spanish': 'dev_micro_f1_metric_epoch',
    # 'multirc': 'dev_macro_f1_metric_epoch',
    # 'fever': 'dev_macro_f1_metric_epoch',
    # 'sst': 'dev_acc_metric_epoch',
    # 'amazon': 'dev_acc_metric_epoch',
    # 'yelp': 'dev_acc_metric_epoch',
    # 'stf': 'dev_binary_f1_metric_epoch',
    # 'olid': 'dev_macro_f1_metric_epoch',
    # 'irony': 'dev_binary_f1_metric_epoch',
}

data_keys = ['item_idx', 'input_ids', 'hashtag_ids', 'attention_mask', 'tree', 'label']

