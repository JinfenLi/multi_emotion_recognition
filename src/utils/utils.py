"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import os
from src.utils.data import dataset_info
import pandas as pd


def generate_output_file(dataset, tokenizer, input_ids, targets, probabilities, output_dir):
    texts = []
    for i in range(len(input_ids)):
        text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        texts.append(text)
    labels = [','.join([dataset_info[dataset]['classes'][i] for i, pp in enumerate(p) if pp > 0.5]) for p in probabilities]
    labels = ['neutral' if len(l) == 0 else l for l in labels]
    probs = [','.join([dataset_info[dataset]['classes'][i] for i, pp in enumerate(p) if pp > 0.5]) for p in probabilities]
    targets = [','.join([dataset_info[dataset]['classes'][i] for i, tt in enumerate(t) if tt !=0]) for t in targets]
    result_df = pd.DataFrame({'text': texts, 'target': targets, 'prediction': labels, 'probs': probs})
    result_df.to_csv(os.path.join(output_dir, 'result.csv'), index=False)
    return result_df

