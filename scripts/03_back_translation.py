import sys
import os
import time
sys.path.append('.')

import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

from src.utils.data import load_hf_dataset, save_data, push_to_hf
from src.utils.config import load_config


def back_translate(text: str, pivot: str = 'en', delay: float = 0.5) -> str:
    """Back translation: VI -> pivot -> VI"""
    try:
        intermediate = GoogleTranslator(source='vi', target=pivot).translate(text)
        time.sleep(delay / 2)
        result = GoogleTranslator(source=pivot, target='vi').translate(intermediate)
        time.sleep(delay / 2)
        return result
    except Exception as e:
        print(f"Error translating: {e}")
        return None


def main():
    print("Starting Back Translation...")
    
    config = load_config()
    bt_config = config['augmentation']['back_translation']
    hf_config = config['huggingface']
    
    REPO_ID = hf_config['dataset_repo_id']
    SAMPLE_PER_LABEL = config['augmentation']['samples_per_label']['HATE']
    PIVOT_LANG = bt_config['intermediate_language']
    DELAY_SEC = 0.5
    BATCH_SAVE = 100
    MAX_SAMPLES = bt_config.get('max_samples', 1000)
    
    print(f"Loading data from {REPO_ID}...")
    train_df = load_hf_dataset(REPO_ID, split='tdtu_train')
    print(f"Total samples: {len(train_df):,}")
    print(f"Columns: {train_df.columns.tolist()}")
    
    AUGMENT_LABELS = ['HATE', 'OFFENSIVE']
    dfs = []
    for label in AUGMENT_LABELS:
        subset = train_df[train_df['label'] == label]
        n = min(len(subset), SAMPLE_PER_LABEL)
        sampled = subset.sample(n, random_state=config['project']['seed'])
        dfs.append(sampled)
        print(f"   {label}: {n} samples")
    
    df_minority = pd.concat(dfs, ignore_index=True)
    
    if MAX_SAMPLES and len(df_minority) > MAX_SAMPLES:
        df_minority = df_minority.sample(MAX_SAMPLES, random_state=config['project']['seed'])
    
    print(f"Total samples to translate: {len(df_minority):,}")
    
    CHECKPOINT_PATH = 'data/augmented/bt_checkpoint.csv'
    os.makedirs('data/augmented', exist_ok=True)
    
    augmented_rows = []
    start_index = 0
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nFound checkpoint: {CHECKPOINT_PATH}")
        df_checkpoint = pd.read_csv(CHECKPOINT_PATH)
        augmented_rows = df_checkpoint.to_dict('records')
        start_index = len(augmented_rows)
        print(f"Resuming from sample {start_index} / {len(df_minority)}")
    else:
        print("\nStarting from scratch...")
    
    print(f"\nTranslating with pivot: {PIVOT_LANG}")
    failed = []
    
    for i in tqdm(range(start_index, len(df_minority)), desc='Back Translation'):
        row = df_minority.iloc[i]
        try:
            bt_text = back_translate(row['text'], PIVOT_LANG, DELAY_SEC)
            if bt_text and bt_text.strip() != row['text'].strip():
                augmented_rows.append({
                    'text': bt_text,
                    'label': row['label'],
                    'source': f"{row.get('source', 'unknown')}_bt_{PIVOT_LANG}",
                    'original_text': row['text'],
                    'split': row.get('split', 'train'),
                })
        except Exception as e:
            failed.append({'index': i, 'error': str(e), 'text': row['text']})
        
        if (i + 1) % BATCH_SAVE == 0:
            pd.DataFrame(augmented_rows).to_csv(CHECKPOINT_PATH, index=False)
    
    if augmented_rows:
        pd.DataFrame(augmented_rows).to_csv(CHECKPOINT_PATH, index=False)
    
    df_bt = pd.DataFrame(augmented_rows)
    df_failed = pd.DataFrame(failed) if failed else pd.DataFrame()
    
    print(f"\nAugmented: {len(df_bt):,} | Failed: {len(df_failed):,}")
    
    if len(df_bt) > 0:
        print("\nCalculating BLEU scores...")
        try:
            from sacrebleu.metrics import BLEU
            bleu = BLEU(effective_order=True)
            scores = []
            for h, r in zip(df_bt['text'].tolist(), df_bt['original_text'].tolist()):
                try:
                    scores.append(bleu.sentence_score(h, [r]).score)
                except:
                    scores.append(0.0)
            df_bt['bleu'] = scores
            
            before = len(df_bt)
            df_bt = df_bt[df_bt['bleu'] < 100].reset_index(drop=True)
            print(f"   Filtered identical: {before - len(df_bt)}")
            print(f"   Remaining: {len(df_bt):,}")
            print(f"   BLEU - Mean: {df_bt['bleu'].mean():.2f} | Median: {df_bt['bleu'].median():.2f}")
        except Exception as e:
            print(f"   Skipping BLEU: {e}")
    
    if len(df_bt) > 0:
        output_file = bt_config.get('output_file', 'data/augmented/bt_only.csv')
        save_data(df_bt, output_file)
        
        keep_cols = ['text', 'label', 'source', 'split']
        available_cols = [col for col in keep_cols if col in df_bt.columns]
        train_bt = pd.concat([train_df, df_bt[available_cols]], ignore_index=True)
        train_bt = train_bt.sample(frac=1, random_state=config['project']['seed']).reset_index(drop=True)
        
        save_data(train_bt, 'data/augmented/train_bt.csv')
        
        push_to_hf(train_bt, REPO_ID, config_name='bt')
    
    if len(df_failed) > 0:
        save_data(df_failed, 'data/augmented/bt_failed.csv')
    
    print(f"Final stats:")
    print(f"Augmented: {len(df_bt):,}")
    print(f"Failed: {len(df_failed):,}")


if __name__ == "__main__":
    main()