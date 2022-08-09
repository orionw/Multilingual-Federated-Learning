

LANG_MAP_EUROPARL = {
    "en": 0,
    "cs": 1,
    "lt": 2,
    "es": 3,
    "pl": 4,
    "fi": 5,
    "pt": 6,
    "de": 7
}


LANG_MAP_UN_CORPUS = {
    "en": 0,
    "es": 1,
    "zh": 2,
    "ru": 3,
    "ar": 4,
    "fr": 5,
}

LANG_MAP_PAWSX = {
    "de": 0,
    "en": 1,
    "es": 2,
    "fr": 3,
    "ja": 4,
    "ko": 5,
    "zh": 6,
}


LANG_MAP_UN_MT_CORPUS = {
    "en-fr": 0,
    "ar-es": 1,
    "ru-zh": 2,
}

LANG_MAP_MTNT_MT_CORPUS = {
    "en-fr": 0,
    "en-ja": 1,
}


LANG_MAP_NC = {
    "en": 0,
    "es": 1,
    "fr": 2,
    "de": 3,
    "ru": 4
}


MBART_MAP = {
    "en": "en_XX",
    "cs": "cs_CZ",
    "lt": "lt_LT",
    "es": "es_XX",
    "pl": "pl_PL",
    "fi": "fi_FI",
    "pt": "pt_XX",
    "de": "de_DE",
}


POOL_SIZE = {
    "brown": 1,
    "wmt": 8,
    "un_corpus": 6,
    "un_mt_corpus": 3,
    "mtnt": 2,
    "pawsx": 7,
    "nc": 5,
}

DATA_TO_FILE_PATHS = {
    "brown": "data/brown",
    "wmt": "data/wmt",
    "un_corpus": 'data/un_corpus',
    "un_mt_corpus": 'data/un_mt_corpus',
    "mtnt": "data/mtnt_mt_corpus",
    "pawsx": "data/pawsx",
    "nc": "data/nc"
}

MAP_LANG_MAP = {
    "data/wmt": LANG_MAP_EUROPARL,
    "data/un_corpus": LANG_MAP_UN_CORPUS,
    "data/un_mt_corpus": LANG_MAP_UN_MT_CORPUS,
    "data/mtnt_mt_corpus": LANG_MAP_MTNT_MT_CORPUS,
    "data/pawsx": LANG_MAP_PAWSX,
    "data/nc": LANG_MAP_NC,
}
