English_Hashtag_Voc_Size = 32389 + 1
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
    'se_english': 'dev_macro_f1_metric_epoch',
    'se_arabic': 'dev_macro_f1_metric_epoch',
    'se_spanish': 'dev_macro_f1_metric_epoch',
}

data_keys = ['item_idx', 'input_ids', 'hashtag_ids', 'attention_mask', 'tree', 'label']

