DATASET_NAME = "McGill-NLP/african_celtic_dataset"

SOURCE_LANGUAGES = ["hausa", "igbo", "yoruba"]
TARGET_LANGUAGE = "english"
ALL_LANGUAGES = ["english", "hausa", "igbo", "yoruba"]

LANG_PREFIX = {
    "english": "E",
    "hausa": "H",
    "igbo": "I",
    "yoruba": "Y",
}

# This file saves all confirmed 4-language pairs — 
# This also what the dataloader will use directly at runtime rather than rebuilding the index every time.
OUT_PATH = "pair_index.json"

# Block boundaries confirmed by inspection
# Rows are sorted alphabetically by language within each split.
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

KNOWN_SOURCE_CODES = {
    "train": ["MD", "NX", "TR"],
    "dev": ["TE"],
}


#  Alignment rule
# Confirmed: text_id[1:] is the shared sentence key.
# ENX_0001 → key "NX_0001" ↔ HNX_0001 / INX_0001 / YNX_0001
def alignment_key(text_id: str):
    """Strip language prefix from text_id to get shared sentence key."""
    return text_id[1:] if text_id and len(text_id) > 1 else None


TARGET_SAMPLE_RATE = 16000  
ORIGINAL_SAMPLE_RATE = 48000  # raw recordings
