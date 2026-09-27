"""
Comprehensive Preprocessing and Normalization for Business Entity Resolution.
Supports multilingual entities across US, India, and France.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Comprehensive legal entity forms across US, India, and France
LEGAL_FORMS = {
    # US / UK / International
    "private", "pvt", "limited", "ltd", "incorporated", "inc",
    "corporation", "corp", "company", "co", "llc", "llp", "plc",
    "services", "holdings", "enterprises", "group", "solutions",
    "industries", "associates", "partners", "management", "international", "intl",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "gie", "ei", "ste", "societe",
    # India (English & Devanagari)
    "प्राइवेट", "लिमिटेड", "प्रा", "लि", "एलएलसी", "एलएलपी", "कॉर्पोरेशन", "इंक",
    # Honorifics and Prefixes
    "mr", "mrs", "ms", "dr", "shri", "smt", "shree", "m/s",
}

SQL_CLEAN_STR = """
strip_accents(lower(trim(regexp_replace(regexp_replace(coalesce({col}, ''), '[^\\p{L}\\p{M}\\p{N}\\s]', ' ', 'g'), '\\s+', ' ', 'g'))))
"""

SQL_CLEAN_WEBSITE = """
regexp_replace(regexp_replace({col}, '(\\bwww\\.|\\.com|\\.in|\\.org|\\.net|\\.co|\\.fr|\\.gov)\\b', ' ', 'g'), '\\s+', ' ', 'g')
"""

SQL_STRIP_LEGAL = """
trim(regexp_replace(regexp_replace(
    {col},
    '(^| )(private|pvt|limited|ltd|incorporated|inc|corporation|corp|company|co|llc|llp|plc|services|holdings|enterprises|group|solutions|industries|associates|partners|management|international|intl|sarl|sas|sasu|sa|eurl|snc|sci|gie|ei|ste|societe|प्राइवेट|लिमिटेड|प्रा|लि|एलएलसी|एलएलपी|mr|mrs|ms|dr|shri|smt|shree)( |$)',
    ' ', 'g'),
    '\\s+', ' ', 'g'))
"""


def normalize_text(value: Optional[str]) -> str:
    """Normalize text by stripping accents, lowercasing, and removing punctuation."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)

    # Unicode NFKD decomposition and strip combining accents
    nfkd = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

    # Strip website extensions
    value = re.sub(r"(\bwww\.|\.com|\.in|\.org|\.net|\.co|\.fr|\.gov)\b", " ", value)

    # Keep only letters, marks, numbers, and spaces
    cleaned = []
    for char in value:
        cat = unicodedata.category(char)
        if cat[0] in {"L", "M", "N"} or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def tokenize(value: str, min_len: int = 2) -> list[str]:
    """Tokenize normalized string into list of non-trivial tokens."""
    if not value:
        return []
    return [t for t in value.split() if len(t) >= min_len]


def core_name(value: str) -> str:
    """Extract core business name by removing legal suffixes and honorifics."""
    norm = normalize_text(value)
    tokens = [t for t in norm.split() if t not in LEGAL_FORMS]
    return " ".join(tokens)


def sorted_core_name(value: str) -> str:
    """Alphabetically sorted unique core name tokens for word-order invariance."""
    tokens = sorted(set(tokenize(core_name(value), min_len=2)))
    return " ".join(tokens)


def compact_text(value: str) -> str:
    """Compact string without whitespace for concatenated/domain matching."""
    return "".join(tokenize(value, min_len=1))