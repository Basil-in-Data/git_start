"""
Arabic NER preprocessing pipeline.
Follows methodology from akschneider1/arabic-pii-ner-model:
  ArabicTextNormalizer → LabelAligner → NERPreprocessor
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer


@dataclass
class ProcessedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    original_text: str
    word_ids: List[Optional[int]]


class ArabicTextNormalizer:
    """Normalizes Arabic text: remove diacritics, normalize alef/taa/yaa, dialect chars."""

    def __init__(self):
        self.arabic_diacritics = re.compile(r'[ً-ٰٟـۖ-ۭ]')
        self.char_mappings = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
            'ة': 'ه',
            'ي': 'ى', 'ئ': 'ى',
            'ك': 'ك', 'ک': 'ك',
        }
        self.dialect_normalizations = {
            'چ': 'ج', 'پ': 'ب', 'گ': 'ج', 'ڤ': 'ف', 'ڨ': 'ق',
        }
        arabic_indic = '٠١٢٣٤٥٦٧٨٩'
        persian_nums = '۰۱۲۳۴۵۶۷۸۹'
        western = '0123456789'
        self._digit_table = str.maketrans(arabic_indic + persian_nums, western * 2)

    def normalize(self, text: str) -> str:
        if not text:
            return text
        text = self.arabic_diacritics.sub('', text)
        for old, new in self.dialect_normalizations.items():
            text = text.replace(old, new)
        for old, new in self.char_mappings.items():
            text = text.replace(old, new)
        text = text.translate(self._digit_table)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class LabelAligner:
    """
    Aligns word-level IOB2 labels to AraBERT subword tokens.
    Only the first subtoken of each word keeps the label; continuations get -100.
    """

    PII_LABELS = [
        'O',
        'B-PERSON', 'I-PERSON',
        'B-LOCATION', 'I-LOCATION',
        'B-ORGANIZATION', 'I-ORGANIZATION',
        'B-PHONE', 'I-PHONE',
        'B-EMAIL', 'I-EMAIL',
        'B-ID_NUMBER', 'I-ID_NUMBER',
        'B-ADDRESS', 'I-ADDRESS',
    ]

    def __init__(self):
        self.label_to_id = {label: idx for idx, label in enumerate(self.PII_LABELS)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.PII_LABELS)}

    def convert_iob_to_iob2(self, labels: List[str]) -> List[str]:
        """Ensure every entity starts with B-, fixing bare I- starts."""
        converted = []
        prev = 'O'
        for label in labels:
            if label == 'O':
                converted.append('O')
            elif label.startswith('I-'):
                entity = label[2:]
                prev_entity = prev[2:] if prev.startswith(('B-', 'I-')) else None
                converted.append(label if prev_entity == entity else f'B-{entity}')
            else:
                converted.append(label)
            prev = converted[-1]
        return converted

    def align(
        self,
        words: List[str],
        labels: List[str],
        tokenizer,
        max_length: int = 128,
    ) -> Tuple[List[int], List[int], List[int], List[Optional[int]]]:
        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        word_ids = tokenized.word_ids()

        aligned = [-100] * len(input_ids)
        prev_word = None
        for tok_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                aligned[tok_idx] = -100
            elif word_idx != prev_word:
                label = labels[word_idx] if word_idx < len(labels) else 'O'
                aligned[tok_idx] = self.label_to_id.get(label, 0)
            # continuation subtokens stay -100
            prev_word = word_idx

        return input_ids, attention_mask, aligned, word_ids


class NERPreprocessor:
    """
    Full preprocessing pipeline:
      1. Normalize Arabic text
      2. Convert Wojood/augmented tags to PII IOB2 labels
      3. Align to AraBERT subword tokens
    """

    MODEL_NAME = 'aubmindlab/bert-base-arabertv2'

    def __init__(self, model_name: str = MODEL_NAME):
        self.normalizer = ArabicTextNormalizer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.aligner = LabelAligner()
        self.label_to_id = self.aligner.label_to_id
        self.id_to_label = self.aligner.id_to_label

    def preprocess_sentence(self, sentence_df: pd.DataFrame, max_length: int = 128) -> ProcessedExample:
        token_col = next((c for c in ('token', 'Token') if c in sentence_df.columns), None)
        tag_col = next((c for c in ('tag', 'tags', 'Tag') if c in sentence_df.columns), None)
        if token_col is None or tag_col is None:
            raise ValueError(f"Missing token/tag columns. Found: {sentence_df.columns.tolist()}")

        words = sentence_df[token_col].astype(str).tolist()
        labels = sentence_df[tag_col].astype(str).tolist()

        normalized = [self.normalizer.normalize(w) for w in words]
        iob2_labels = self.aligner.convert_iob_to_iob2(labels)

        input_ids, attention_mask, aligned_labels, word_ids = self.aligner.align(
            normalized, iob2_labels, self.tokenizer, max_length
        )

        return ProcessedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=aligned_labels,
            original_text=' '.join(words),
            word_ids=word_ids,
        )

    def preprocess_dataset(
        self,
        df: pd.DataFrame,
        max_length: int = 128,
        max_sentences: Optional[int] = None,
    ) -> List[ProcessedExample]:
        id_col = next(
            (c for c in ('sentence_id', 'global_sentence_id', 'Sentence_ID') if c in df.columns),
            None,
        )
        if id_col is None:
            raise ValueError(f"No sentence ID column found. Columns: {df.columns.tolist()}")

        groups = list(df.groupby(id_col))
        if max_sentences:
            groups = groups[:max_sentences]

        examples = []
        for i, (sid, sent_df) in enumerate(groups):
            pos_col = next((c for c in ('token_position', 'token_id', 'Token_ID') if c in sent_df.columns), None)
            if pos_col:
                sent_df = sent_df.sort_values(pos_col)
            try:
                examples.append(self.preprocess_sentence(sent_df, max_length))
            except Exception as e:
                print(f"  [skip] sentence {sid}: {e}")
            if (i + 1) % 500 == 0:
                print(f"  Preprocessed {i+1}/{len(groups)} sentences...")

        print(f"  Done: {len(examples)}/{len(groups)} sentences processed.")
        return examples

    def print_stats(self, examples: List[ProcessedExample]):
        all_labels = [l for ex in examples for l in ex.labels if l != -100]
        from collections import Counter
        counts = Counter(self.id_to_label[l] for l in all_labels)
        print(f"\nLabel distribution ({len(all_labels)} tokens, {len(examples)} sentences):")
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {label:20s} {count:7,d}  ({count/len(all_labels)*100:.1f}%)")
