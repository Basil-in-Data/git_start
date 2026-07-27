"""
Creates augmented training data by combining:
  1. Wojood NER data → PERSON / LOCATION / ORGANIZATION (via schema mapper)
  2. Synthetic data  → PHONE / EMAIL / ID_NUMBER / ADDRESS (via rules + templates)

Output: token-level BIO CSV ready for NERPreprocessor.
Follows akschneider1/arabic-pii-ner-model data_augmentation.py methodology.
"""

import os
import random
import pandas as pd
from typing import List, Dict

from arabic_pii.schema_mapper import WojoodToPIIMapper
from arabic_pii.synthetic_generator import SyntheticPIIGenerator

WOJOOD_TRAIN = os.path.join(
    os.path.dirname(__file__), '..', 'arabic_pii_source', 'Wojood', 'Wojood1_1_flat', 'train.csv'
)
WOJOOD_VAL = os.path.join(
    os.path.dirname(__file__), '..', 'arabic_pii_source', 'Wojood', 'Wojood1_1_flat', 'val.csv'
)
WOJOOD_TEST = os.path.join(
    os.path.dirname(__file__), '..', 'arabic_pii_source', 'Wojood', 'Wojood1_1_flat', 'test.csv'
)


class DataAugmentation:
    """
    Builds token-level training data by merging:
      - Wojood sentences (remapped to PII labels)
      - Synthetic sentences (self-labeled via the rule engine)
    """

    def __init__(self):
        self.mapper = WojoodToPIIMapper()
        self.generator = SyntheticPIIGenerator()

    # ------------------------------------------------------------------ Wojood

    def _load_wojood_split(self, csv_path: str, max_sentences: int = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df['pii_tag'] = df['tags'].apply(self.mapper.map_tag)

        id_col = 'global_sentence_id'
        records = []
        for i, (sid, sent) in enumerate(df.groupby(id_col)):
            if max_sentences and i >= max_sentences:
                break
            sent = sent.sort_values('token_position')
            for pos, row in enumerate(sent.itertuples()):
                records.append({
                    'sentence_id': f'wojood_{sid}',
                    'token_id': pos,
                    'token': row.token,
                    'tag': row.pii_tag,
                    'source': 'wojood',
                })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------ Synthetic

    def _generate_synthetic(self, num_sentences: int) -> pd.DataFrame:
        print(f"  Generating {num_sentences} synthetic sentences...")
        sentences = self.generator.generate_dataset(num_sentences)

        records = []
        for sent in sentences:
            tokens = sent.text.split()
            token_positions = []
            pos = 0
            for tok in tokens:
                start = sent.text.find(tok, pos)
                token_positions.append((tok, start, start + len(tok)))
                pos = start + len(tok)

            bio_tags = ['O'] * len(tokens)
            for pii in sent.detected_pii:
                first = True
                for i, (tok, ts, te) in enumerate(token_positions):
                    if ts < pii['end'] and te > pii['start']:
                        bio_tags[i] = f"B-{pii['type']}" if first else f"I-{pii['type']}"
                        first = False

            for i, (tok, _, _) in enumerate(token_positions):
                records.append({
                    'sentence_id': f'synthetic_{sent.sentence_id}',
                    'token_id': i,
                    'token': tok,
                    'tag': bio_tags[i],
                    'source': 'synthetic',
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------ public API

    def build(
        self,
        max_wojood_sentences: int = 5000,
        synthetic_sentences: int = 3000,
        output_path: str = 'train_augmented.csv',
        seed: int = 42,
    ) -> pd.DataFrame:
        random.seed(seed)
        print("Step 1/3 — loading Wojood training data...")
        wojood_df = self._load_wojood_split(WOJOOD_TRAIN, max_sentences=max_wojood_sentences)
        print(f"  Wojood tokens: {len(wojood_df):,}")

        print("Step 2/3 — generating synthetic PII sentences...")
        synthetic_df = self._generate_synthetic(synthetic_sentences)
        print(f"  Synthetic tokens: {len(synthetic_df):,}")

        print("Step 3/3 — merging and saving...")
        combined = pd.concat([wojood_df, synthetic_df], ignore_index=True)

        combined.to_csv(output_path, index=False)
        print(f"  Saved {len(combined):,} tokens → {output_path}")
        self._print_stats(combined)
        return combined

    def build_val(self, max_sentences: int = 1000, output_path: str = 'val_augmented.csv') -> pd.DataFrame:
        df = self._load_wojood_split(WOJOOD_VAL, max_sentences=max_sentences)
        df.to_csv(output_path, index=False)
        print(f"Val set: {len(df):,} tokens → {output_path}")
        return df

    def build_test(self, max_sentences: int = 2000, output_path: str = 'test_augmented.csv') -> pd.DataFrame:
        df = self._load_wojood_split(WOJOOD_TEST, max_sentences=max_sentences)
        df.to_csv(output_path, index=False)
        print(f"Test set: {len(df):,} tokens → {output_path}")
        return df

    def _print_stats(self, df: pd.DataFrame):
        from collections import Counter
        tag_counts = Counter(df['tag'])
        source_counts = Counter(df['source'])
        print("\nSource distribution:")
        for src, n in source_counts.items():
            print(f"  {src}: {n:,}")
        print("Tag distribution (non-O):")
        for tag, n in sorted(tag_counts.items(), key=lambda x: -x[1]):
            if tag != 'O':
                print(f"  {tag:25s} {n:,}")
        entity_n = sum(v for k, v in tag_counts.items() if k != 'O')
        total_n = sum(tag_counts.values())
        print(f"Entity ratio: {entity_n}/{total_n} ({entity_n/total_n*100:.1f}%)")
