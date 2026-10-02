"""
Normalization utilities for business_name and business_address fields.

Design goals:
- Country-agnostic: must work equally well on US / India / France (unseen
  in training) records, so no lookups keyed to specific country values.
- Handles legal-entity tokens as *prefixes or suffixes* in multiple
  jurisdictions (US: Inc/LLC/Corp..., India: Pvt/Ltd..., France: SARL/SAS/
  SASU/EURL/SA...). This is generic domain knowledge about corporate legal
  forms, not a lookup of any specific business's registration.
- Strips noise (leading punctuation runs, repeated tokens, URLs used as
  names, stray symbols) seen in the data profiling step.
- Preserves non-Latin scripts (Devanagari, Kannada, ...) rather than
  discarding them -- similarity features still work token/char-wise on them.
"""
import re
import unicodedata

# --- Legal entity forms across the three known jurisdictions + generic ones.
# Used for BOTH prefix and suffix stripping since the data shows both
# ("LLC Moncada Learning Center" and "Moncada Learning Center LLC").
LEGAL_FORM_TOKENS = {
    # US
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc",
    "l l c", "ltd", "limited", "llp", "l l p", "pllc", "pc", "plc",
    # India
    "pvt", "private", "pvtltd",
    # France
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "fils",
    # Generic connectors that show up glued to legal forms
    "dba",
}

# Sort longest-first so multi-word forms match before their sub-tokens
_LEGAL_FORM_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(t) for t in LEGAL_FORM_TOKENS),
                              key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Common address-component abbreviations -> canonical long form (EN-biased,
# but harmless no-ops on non-English tokens).
ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "pl": "place", "sq": "square", "hwy": "highway",
    "pkwy": "parkway", "apt": "apartment", "bldg": "building",
    "fl": "floor", "ste": "suite", "no": "number", "po": "po",
    "hno": "house number",
}

_WS_RE = re.compile(r"\s+")
_LEADING_NOISE_RE = re.compile(r"^[\-\#\<\>\|\*\.\s]+")
_PUNCT_RE = re.compile(r"[^\w\s&]", flags=re.UNICODE)
_URL_SUFFIX_RE = re.compile(r"\.(com|in|co|org|net|fr)\b", flags=re.IGNORECASE)
_TRAILING_URL_RE = re.compile(r"\|\s*www\.\S+$")


def _dedupe_consecutive_tokens(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def normalize_name(raw: str) -> str:
    """Normalize a business_name into a comparable canonical string."""
    if raw is None or (isinstance(raw, float)):
        return ""
    s = str(raw)
    s = unicodedata.normalize("NFKC", s)
    s = _TRAILING_URL_RE.sub("", s)
    s = _LEADING_NOISE_RE.sub("", s)
    s = s.replace("&", " and ")
    s = _URL_SUFFIX_RE.sub("", s)  # strip domain suffixes when name is a URL
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _LEGAL_FORM_PATTERN.sub(" ", s)
    tokens = _WS_RE.split(s.strip())
    tokens = [t for t in tokens if t]
    tokens = _dedupe_consecutive_tokens(tokens)
    return " ".join(tokens)


def normalize_address(raw: str) -> str:
    """Normalize a business_address into a comparable canonical string."""
    if raw is None or (isinstance(raw, float)):
        return ""
    s = str(raw)
    s = unicodedata.normalize("NFKC", s)
    s = _LEADING_NOISE_RE.sub("", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    tokens = _WS_RE.split(s.strip())
    tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens if t]
    tokens = _dedupe_consecutive_tokens(tokens)
    return " ".join(tokens)


def normalize_country(raw: str) -> str:
    """Light-touch normalization of the country label (open-set safe)."""
    if raw is None or (isinstance(raw, float)):
        return ""
    return str(raw).strip().lower()


def add_normalized_columns(df, name_col="business_name", addr_col="business_address",
                            country_col="country"):
    """Vectorized-ish normalization; returns df with new columns added in place."""
    df["norm_name"] = df[name_col].map(normalize_name)
    df["norm_address"] = df[addr_col].map(normalize_address)
    df["norm_country"] = df[country_col].map(normalize_country)
    df["blend_text"] = (df["norm_name"] + " " + df["norm_address"]).str.strip()
    return df
