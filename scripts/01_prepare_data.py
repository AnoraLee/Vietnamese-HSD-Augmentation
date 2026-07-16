from huggingface_hub import login
import os

token = os.getenv('HF_TOKEN')
if token:
    login(token=token)

import sys
import os
sys.path.append('.')

import pandas as pd
from datasets import load_dataset

from src.utils.data import save_data, push_to_hf


def main():
    
    REPO_ID = 'HoaiAn001/tdtu-vietnamese-hsd'
    LABEL_MAP = {0: 'CLEAN', 1: 'OFFENSIVE', 2: 'HATE'}
    
    vihsd_raw = load_dataset('uitnlp/vihsd')
    
    def normalize(df, split_name):
        df = df.to_pandas()
        out = pd.DataFrame()
        out['text'] = df['free_text']
        out['label'] = df['label_id'].map(LABEL_MAP)
        out['source'] = 'ViHSD'
        out['split'] = split_name
        return out
    
    train_df = normalize(vihsd_raw['train'], 'train')
    dev_df = normalize(vihsd_raw['validation'], 'dev')
    test_df = normalize(vihsd_raw['test'], 'test')
    
    print(f"   Train: {len(train_df):,}")
    print(f"   Dev: {len(dev_df):,}")
    print(f"   Test: {len(test_df):,}")
    
    os.makedirs('data/processed', exist_ok=True)
    save_data(train_df, 'data/processed/train.csv')
    save_data(dev_df, 'data/processed/dev.csv')
    save_data(test_df, 'data/processed/test.csv')
    
    push_to_hf(train_df, REPO_ID, config_name='baseline')
    
    print("\nData Preparation completed!")

if __name__ == "__main__":
    main()