import sys
import os
sys.path.append('.')

import pandas as pd
from tqdm import tqdm

from src.utils.data import load_hf_dataset, save_data, push_to_hf
from src.augmentation.eda import EDAAugmenter


def main():
    REPO_ID = 'HoaiAn001/tdtu-vietnamese-hsd'
    AUGMENT_LABELS = ['HATE', 'OFFENSIVE']
    SAMPLE_PER_LABEL = 500
    
    print(f"Loading data from {REPO_ID}...")
    train_df = load_hf_dataset(REPO_ID, split='tdtu_train')
    print(f"   Total samples: {len(train_df):,}")
    print(f"   Columns: {train_df.columns.tolist()}")
    
    dfs = []
    for label in AUGMENT_LABELS:
        subset = train_df[train_df['label'] == label]
        n = min(len(subset), SAMPLE_PER_LABEL)
        sampled = subset.sample(n, random_state=42)
        dfs.append(sampled)
        print(f"   {label}: {n} samples")
    
    df_minority = pd.concat(dfs, ignore_index=True)
    print(f"Total samples to augment: {len(df_minority):,}")
    
    print("\nApplying EDA...")
    augmenter = EDAAugmenter(alpha=0.1, num_aug=2)
    
    augmented_rows = []
    for _, row in tqdm(df_minority.iterrows(), total=len(df_minority)):
        results = augmenter.augment_one(row['text'], row['label'])
        for aug in results:
            augmented_rows.append({
                'text': aug.text,
                'label': aug.label,
                'source': f"{row.get('source', 'unknown')}_eda",
                'original_text': row['text'],
            })
    
    df_eda = pd.DataFrame(augmented_rows)
    
    print(f"\nTotal Augmented: {len(df_eda):,}")
    print("Label Distribution:")
    print(df_eda['label'].value_counts())
    
    os.makedirs('data/augmented', exist_ok=True)
    save_data(df_eda, 'data/augmented/eda_augmented.csv')
    
    push_to_hf(df_eda, 'HoaiAn001/tdtu-hsd-aug', config_name='eda')

if __name__ == "__main__":
    main()