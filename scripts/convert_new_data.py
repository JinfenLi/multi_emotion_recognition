"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import argparse
import os
from os.path import exists

import pandas as pd
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def main():
    # df = pd.DataFrame([])
    # for file in ['2017_neg_trans_halfway.csv', '2018_neg_trans.csv', '2018_pos_trans.csv', '2017_pos_trans.csv']:
    os.makedirs(os.path.join(args.data_dir, "new_data"), exist_ok=True)
    df = pd.read_csv(os.path.join(args.data_dir, "new_data", args.file_name))
    # sub_df['file_name'] = file
    # df = pd.concat([df, sub_df], ignore_index=True)

    # rename columns
    df = df.rename(columns={'Text': 'text'})
    df['label'] = 1
    df.to_csv(os.path.join(args.data_dir, "new_data", "new_data.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert New Dataset')
    parser.add_argument('--data_dir', type=str, default='data/', help='Root directory for datasets')
    parser.add_argument('--file_name', type=str, default='xxx', help='which file to convert')
    args = parser.parse_args()
    main()