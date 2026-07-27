"""
Arabic regex-based PII safety net.
Follows akschneider1/arabic-pii-ner-model rules.py methodology.
Covers: GCC phone numbers, national IDs, IBANs, email, credit cards,
        passports, IMEI, IP addresses, Arabic age context.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PIIMatch:
    text: str
    pii_type: str
    start_pos: int
    end_pos: int
    confidence: float
    pattern_name: str


class ArabicPIIDetector:
    """
    Rule-based detector for structured Arabic PII.
    Used as a safety net alongside the AraBERT ML model —
    mirrors the role of SAFETY_PATTERNS in the English pipeline.
    """

    def __init__(self):
        self._phone_patterns = self._build_phone_patterns()
        self._email_patterns = self._build_email_patterns()
        self._id_patterns = self._build_id_patterns()
        self._iban_patterns = self._build_iban_patterns()
        self._credit_card_patterns = self._build_credit_card_patterns()
        self._passport_patterns = self._build_passport_patterns()
        self._imei_patterns = self._build_imei_patterns()
        self._ip_patterns = self._build_ip_patterns()
        self._age_patterns = self._build_age_patterns()

    # ------------------------------------------------------------------ builders

    def _build_phone_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        return [
            ('SA_intl', re.compile(r'\+966\s*[5][0-9]\s*\d{3}\s*\d{4}'), 0.95),
            ('SA_00', re.compile(r'00966\s*[5][0-9]\s*\d{3}\s*\d{4}'), 0.95),
            ('SA_local', re.compile(r'\b05[0-9]\s*\d{3}\s*\d{4}\b'), 0.90),
            ('JO_intl', re.compile(r'\+962\s*[7][7-9]\s*\d{3}\s*\d{4}'), 0.95),
            ('JO_local', re.compile(r'\b07[7-9]\s*\d{3}\s*\d{4}\b'), 0.85),
            ('UAE_intl', re.compile(r'\+971\s*[5][0-6]\s*\d{3}\s*\d{4}'), 0.95),
            ('EG_intl', re.compile(r'\+20\s*[1][0-5]\s*\d{4}\s*\d{4}'), 0.95),
            ('generic', re.compile(r'\+\d{1,3}\s*\d{1,4}\s*\d{3,4}\s*\d{4}'), 0.80),
        ]

    def _build_email_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        return [
            ('standard', re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), 0.95),
            ('arabic_domain', re.compile(r'\b[A-Za-z0-9._%+-]+@[؀-ۿ\w.-]+\.[A-Za-z]{2,}\b'), 0.90),
        ]

    def _build_id_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        return [
            ('SA_national', re.compile(r'\b[12]\d{9}\b'), 0.90),
            ('UAE_emirates', re.compile(r'\b784[-\s]*\d{4}[-\s]*\d{7}[-\s]*\d\b'), 0.90),
            ('EG_national', re.compile(r'\b[23]\d{13}\b'), 0.85),
            ('generic_10d', re.compile(r'\b\d{10}\b'), 0.70),
        ]

    def _build_iban_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        return [
            ('SA_IBAN', re.compile(r'\bSA\d{2}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b', re.I), 0.95),
            ('UAE_IBAN', re.compile(r'\bAE\d{2}\s*\d{3}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{3}\b', re.I), 0.95),
            ('EG_IBAN', re.compile(r'\bEG\d{2}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{3}\b', re.I), 0.95),
            ('JO_IBAN', re.compile(r'\bJO\d{2}\s*[A-Z]{4}\s*\d{22}\b', re.I), 0.95),
        ]

    def _build_credit_card_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        return [
            ('visa', re.compile(r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), 0.90),
            ('mastercard', re.compile(r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), 0.90),
            ('amex', re.compile(r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b'), 0.90),
        ]

    def _build_passport_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        return [
            ('standard', re.compile(r'\b[A-Z]\d{7,8}\b'), 0.80),
            ('two_letter', re.compile(r'\b[A-Z]{2}\d{6,8}\b'), 0.75),
        ]

    def _build_imei_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        return [
            ('formatted', re.compile(r'\b\d{2}-\d{6}-\d{6}-\d{1}\b'), 0.95),
            ('plain_15', re.compile(r'\b\d{15}\b'), 0.80),
        ]

    def _build_ip_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        return [
            ('ipv4', re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'), 0.95),
            ('ipv6_full', re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'), 0.95),
        ]

    def _build_age_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        return [
            ('arabic_age', re.compile(r'(?:من\s+العمر|عمره|عمرها|البالغ)\s+(\d{1,3})\s*(?:عام|سنة)'), 0.85),
            ('years_suffix', re.compile(r'(\d{1,3})\s+(?:عامًا|سنة|عام)\b'), 0.75),
        ]

    # ------------------------------------------------------------------ detection

    def _scan(
        self, text: str, patterns: List[Tuple[str, re.Pattern, float]], pii_type: str
    ) -> List[PIIMatch]:
        matches: List[PIIMatch] = []
        seen: List[Tuple[int, int]] = []
        for name, pat, conf in patterns:
            for m in pat.finditer(text):
                s, e = m.span()
                if not any(s < es and e > ss for ss, es in seen):
                    matches.append(PIIMatch(m.group(), pii_type, s, e, conf, name))
                    seen.append((s, e))
        return matches

    def detect(self, text: str, min_confidence: float = 0.70) -> List[PIIMatch]:
        all_matches: List[PIIMatch] = []
        all_matches += self._scan(text, self._phone_patterns, 'PHONE')
        all_matches += self._scan(text, self._email_patterns, 'EMAIL')
        all_matches += self._scan(text, self._id_patterns, 'ID_NUMBER')
        all_matches += self._scan(text, self._iban_patterns, 'ID_NUMBER')
        all_matches += self._scan(text, self._credit_card_patterns, 'ID_NUMBER')
        all_matches += self._scan(text, self._passport_patterns, 'ID_NUMBER')
        all_matches += self._scan(text, self._imei_patterns, 'ID_NUMBER')
        all_matches += self._scan(text, self._ip_patterns, 'ID_NUMBER')
        all_matches += self._scan(text, self._age_patterns, 'PERSON')

        filtered = [m for m in all_matches if m.confidence >= min_confidence]
        return sorted(filtered, key=lambda m: m.start_pos)
