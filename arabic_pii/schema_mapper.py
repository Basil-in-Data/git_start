"""
Maps Wojood NER entity tags to the 7-category PII label scheme.
Follows akschneider1/arabic-pii-ner-model schema_mapper.py methodology.

Wojood tags → PII categories:
  PERS              → PERSON
  GPE, LOC          → LOCATION
  ORG               → ORGANIZATION
  (PHONE/EMAIL/ID already in PII scheme from synthetic data)
"""

from typing import Dict, List
from collections import defaultdict
import pandas as pd


WOJOOD_TO_PII: Dict[str, str] = {
    # Person
    'PERS': 'PERSON', 'B-PERS': 'PERSON', 'I-PERS': 'PERSON',
    # Location (geo-political entity + location)
    'GPE': 'LOCATION', 'B-GPE': 'LOCATION', 'I-GPE': 'LOCATION',
    'LOC': 'LOCATION', 'B-LOC': 'LOCATION', 'I-LOC': 'LOCATION',
    # Organization
    'ORG': 'ORGANIZATION', 'B-ORG': 'ORGANIZATION', 'I-ORG': 'ORGANIZATION',
    # Structured PII (from synthetic generator, already in scheme)
    'PHONE': 'PHONE', 'B-PHONE': 'PHONE', 'I-PHONE': 'PHONE',
    'EMAIL': 'EMAIL', 'B-EMAIL': 'EMAIL', 'I-EMAIL': 'EMAIL',
    'ID': 'ID_NUMBER', 'B-ID': 'ID_NUMBER', 'I-ID': 'ID_NUMBER',
    'NATIONAL_ID': 'ID_NUMBER', 'PASSPORT': 'ID_NUMBER', 'LICENSE': 'ID_NUMBER',
    'ADDRESS': 'ADDRESS', 'B-ADDRESS': 'ADDRESS', 'I-ADDRESS': 'ADDRESS',
    'STREET': 'ADDRESS', 'BUILDING': 'ADDRESS',
    # Tags that don't map to PII → O
    'DATE': 'O', 'TIME': 'O', 'CARDINAL': 'O', 'ORDINAL': 'O',
    'NORP': 'O', 'OCC': 'O', 'EVENT': 'O', 'PRODUCT': 'O',
    'FAC': 'O', 'LAW': 'O', 'MONEY': 'O', 'PERCENT': 'O',
    'QUANTITY': 'O', 'LANGUAGE': 'O', 'WEBSITE': 'O', 'CURR': 'O',
    'UNIT': 'O', 'O': 'O',
}


class WojoodToPIIMapper:
    """Maps Wojood tags to PII categories and preserves BIO structure."""

    def map_tag(self, wojood_tag: str) -> str:
        """Return the PII category for a Wojood tag, preserving B-/I- prefix."""
        if wojood_tag in WOJOOD_TO_PII:
            pii_cat = WOJOOD_TO_PII[wojood_tag]
        else:
            base = wojood_tag[2:] if wojood_tag.startswith(('B-', 'I-')) else wojood_tag
            pii_cat = WOJOOD_TO_PII.get(base, 'O')

        if pii_cat == 'O':
            return 'O'

        prefix = ''
        if wojood_tag.startswith('B-'):
            prefix = 'B-'
        elif wojood_tag.startswith('I-'):
            prefix = 'I-'
        else:
            prefix = 'B-'  # bare tag (no BIO prefix) treated as B-

        return f'{prefix}{pii_cat}'

    def convert_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a pii_tag column with the mapped label."""
        tag_col = next((c for c in ('tags', 'tag', 'Tag') if c in df.columns), None)
        if tag_col is None:
            raise ValueError(f"No tag column in DataFrame. Columns: {df.columns.tolist()}")
        df = df.copy()
        df['pii_tag'] = df[tag_col].apply(self.map_tag)
        return df

    def extract_entities(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Extract entity strings grouped by PII type."""
        mapped = self.convert_dataframe(df)
        token_col = next((c for c in ('token', 'Token') if c in df.columns), None)
        id_col = next((c for c in ('global_sentence_id', 'sentence_id') if c in df.columns), None)

        entities: Dict[str, List[str]] = defaultdict(list)
        for sid, sent in mapped.groupby(id_col):
            cur_tokens: List[str] = []
            cur_type: str = ''
            for _, row in sent.iterrows():
                tag = row['pii_tag']
                token = str(row[token_col])
                if tag.startswith('B-'):
                    if cur_tokens:
                        entities[cur_type].append(' '.join(cur_tokens))
                    cur_type = tag[2:]
                    cur_tokens = [token]
                elif tag.startswith('I-') and cur_type and tag[2:] == cur_type:
                    cur_tokens.append(token)
                else:
                    if cur_tokens:
                        entities[cur_type].append(' '.join(cur_tokens))
                    cur_tokens = []
                    cur_type = ''
            if cur_tokens:
                entities[cur_type].append(' '.join(cur_tokens))

        return dict(entities)

    def print_mapping_stats(self, df: pd.DataFrame):
        mapped = self.convert_dataframe(df)
        from collections import Counter
        counts = Counter(mapped['pii_tag'])
        print("Wojood → PII tag distribution:")
        for tag, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {tag:25s} {count:8,d}")
