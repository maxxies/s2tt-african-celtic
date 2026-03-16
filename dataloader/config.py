
DATASET_NAME = "McGill-NLP/african_celtic_dataset"

LANGUAGES = ["hausa", "igbo", "yoruba"] 
TARGET_LANG = "english"


TRAIN_BLOCKS = {
    "english": (0, 13000),
    "hausa": (13000, 26000),
    "igbo": (26000, 39000),
    "yoruba": (39000, 52000),
}

DEV_BLOCKS = {
    "english": (0, 1374),
    "hausa": (1374, 2748),
    "igbo": (2748, 4122),
    "yoruba": (4122, 5500),
}


SOURCE_CODES = {
    "train": {
        "MD": {"langs": ["english", "hausa", "igbo", "yoruba"], "pairs": 1503},
        "NX": {"langs": ["english", "hausa", "igbo", "yoruba"], "pairs": 1997},
        "TR": {"langs": ["english", "yoruba"], "pairs": 1000},
    },
    "dev": {
        "TE": {"langs": ["english", "hausa", "igbo", "yoruba"], "pairs": 500},
    },
}

# Alignment rule (as observed in the dataset)
# text_id[1:] is the shared sentence key across all languages.
# e.g. ENX_0001 → key "NX_0001" ↔ HNX_0001 / INX_0001 / YNX_0001
ALIGNMENT_KEY_FN = lambda text_id: text_id[1:] if text_id and len(text_id) > 1 else None

TARGET_SAMPLE_RATE = 16000 
ORIGINAL_SAMPLE_RATE = 48000
