"""
Synthetic Arabic PII sentence generator.
Follows akschneider1/arabic-pii-ner-model synthetic_generator.py methodology.

Generates labeled Arabic sentences covering PHONE, EMAIL, ID_NUMBER, ADDRESS —
the four PII types absent from Wojood — using template + gazetteer substitution.
The generated sentences are self-labeled via the rules-based detector.
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

from arabic_pii.rules import ArabicPIIDetector


@dataclass
class SyntheticSentence:
    sentence_id: int
    text: str
    template_category: str
    expected_pii: Dict[str, str]   # type → value
    detected_pii: List[Dict]        # [{text, type, start, end, confidence}]


_TEMPLATES = {
    'NAME': [
        "تم تعيين {name} في منصب جديد.",
        "التقيت مع {name} في المكتب اليوم.",
        "قال الدكتور {name} أن النتائج ممتازة.",
        "تم ترقية {name} إلى منصب مدير عام.",
        "حضر الأستاذ {name} الاجتماع الصباحي.",
        "اتصل {name} لتأكيد الموعد غداً.",
        "المهندس {name} مسؤول عن هذا المشروع.",
        "تخرج {name} من الجامعة بتقدير ممتاز.",
        "نال {name} جائزة أفضل موظف.",
        "مثل {name} الشركة في المعرض.",
    ],
    'ADDRESS': [
        "العنوان الجديد هو {address}.",
        "انتقلت الشركة إلى {address}.",
        "يقع المكتب في {address}.",
        "تم تسليم الطلب إلى {address}.",
        "سيقام الحفل في {address}.",
        "المقر الرئيسي في {address}.",
        "تقع المدرسة في {address}.",
        "المستشفى الجديد في {address}.",
        "الفندق يقع في {address}.",
        "تم نقل المكتب إلى {address}.",
    ],
    'ID': [
        "رقم الهوية الوطنية: {id_number}.",
        "الرقم المدني للعميل هو {id_number}.",
        "يحمل رقم هوية {id_number}.",
        "بطاقة الهوية رقم {id_number} منتهية الصلاحية.",
        "رقم السجل المدني: {id_number}.",
        "تم تحديث بيانات الهوية {id_number}.",
        "الهوية الشخصية رقم {id_number}.",
        "تحقق من صحة الرقم {id_number}.",
        "الهوية المدنية {id_number} صالحة.",
        "رقم الوثيقة الشخصية: {id_number}.",
    ],
    'PHONE': [
        "رقم الهاتف: {phone}.",
        "اتصل على {phone} للاستفسار.",
        "رقم الجوال الجديد: {phone}.",
        "تواصل معنا على {phone}.",
        "للحجز اتصل على {phone}.",
        "رقم الطوارئ: {phone}.",
        "هاتف العمل: {phone}.",
        "يمكن الوصول إليه على {phone}.",
        "رقم الهاتف المحمول: {phone}.",
        "خط ساخن: {phone}.",
    ],
    'EMAIL': [
        "البريد الإلكتروني: {email}.",
        "راسلنا على {email}.",
        "للاستفسارات: {email}.",
        "أرسل إلى {email}.",
        "البريد الرسمي: {email}.",
        "للدعم الفني: {email}.",
        "تواصل عبر {email}.",
        "للمراسلات: {email}.",
        "عنوان الإيميل: {email}.",
        "البريد الشخصي: {email}.",
    ],
    'MULTI': [
        "السيد {name} يسكن في {address} ورقم هاتفه {phone}.",
        "تواصل مع {name} على {email} أو زره في {address}.",
        "هوية {name} رقم {id_number} والعنوان {address}.",
        "اتصل بـ {name} على {phone} أو راسله على {email}.",
        "بيانات العميل: {name}، هاتف: {phone}، هوية: {id_number}.",
        "المدير {name} مكتبه في {address} وإيميله {email}.",
    ],
}

_GAZETTEERS: Dict[str, List[str]] = {
    'name': [
        'أحمد محمد علي', 'فاطمة أحمد', 'محمد عبدالله', 'عائشة سالم',
        'خالد عبدالعزيز', 'نورا أحمد', 'سعد الدين', 'هند محمود',
        'عبدالرحمن يوسف', 'مريم علي', 'طارق السيد', 'زينب حسن',
        'عمر الفاروق', 'ليلى عبدالله', 'يوسف أحمد', 'سارة محمد',
        'إبراهيم خليل', 'خديجة أحمد', 'عثمان بن عفان', 'منى عبدالعزيز',
    ],
    'address': [
        'طريق الملك فهد، الرياض', 'شارع التحلية، جدة', 'طريق الأمير محمد بن فهد، الدمام',
        'حي النزهة، الرياض', 'شارع الأمير سلطان، جدة', 'طريق الملك عبدالعزيز، مكة',
        'حي المروج، الرياض', 'شارع المدينة المنورة، جدة', 'طريق الدائري الشرقي، الرياض',
        'شارع الشيخ زايد، دبي', 'طريق الكورنيش، أبوظبي', 'منطقة دبي مارينا',
        'شارع الملكة رانيا، عمان', 'شارع التحرير، القاهرة', 'كورنيش النيل، الأقصر',
    ],
    'id_number': [
        '1234567890', '2123456789', '1987654321', '2876543210',
        '1456789012', '2345678901', '1789012345', '2012345678',
        '784-2020-1234567-1', '784-2019-9876543-2', '784-2021-5555555-3',
        '29801011234567', '28505051234567', '29912121234567',
    ],
    'phone': [
        '+966 50 123 4567', '+966 55 987 6543', '+966 56 444 5555',
        '0501234567', '0559876543', '0564445555',
        '+971 50 123 4567', '+971 55 987 6543',
        '+962 77 123 4567', '+962 79 987 6543',
        '+20 10 1234 5678', '+20 11 9876 5432',
    ],
    'email': [
        'ahmed@gmail.com', 'fatima@hotmail.com', 'mohamed@outlook.com',
        'sara@yahoo.com', 'khaled@company.sa', 'nora@university.edu.sa',
        'hassan@bank.com', 'layla@hospital.org', 'omar@tech.ae',
        'maryam@consulting.com', 'youssef@trading.jo', 'zainab@clinic.eg',
    ],
}


class SyntheticPIIGenerator:
    """
    Generates labeled Arabic PII sentences using templates and gazetteers.
    Detects placed PII with the rule-based ArabicPIIDetector to produce
    BIO-tagged training examples for PHONE / EMAIL / ID_NUMBER / ADDRESS.
    """

    def __init__(self):
        self.detector = ArabicPIIDetector()

    def _pick(self, key: str) -> str:
        return random.choice(_GAZETTEERS[key])

    def generate_sentence(self, template: str, category: str) -> Tuple[str, Dict[str, str]]:
        sentence = template
        pii_info: Dict[str, str] = {}
        for placeholder in ('name', 'address', 'id_number', 'phone', 'email'):
            tag = '{' + placeholder + '}'
            if tag in sentence:
                value = self._pick(placeholder)
                sentence = sentence.replace(tag, value)
                pii_info[placeholder] = value
        return sentence, pii_info

    def generate_dataset(self, num_sentences: int = 5000) -> List[SyntheticSentence]:
        results: List[SyntheticSentence] = []
        all_templates = [
            (t, cat) for cat, templates in _TEMPLATES.items() for t in templates
        ]
        for i in range(num_sentences):
            tmpl, cat = random.choice(all_templates)
            text, pii_info = self.generate_sentence(tmpl, cat)
            detected = [
                {
                    'text': m.text, 'type': m.pii_type,
                    'start': m.start_pos, 'end': m.end_pos,
                    'confidence': m.confidence,
                }
                for m in self.detector.detect(text, min_confidence=0.70)
            ]
            results.append(SyntheticSentence(i, text, cat, pii_info, detected))
        return results
