from datasets import load_dataset, Audio
from typing import Dict
from .config import DATASET_NAME, TRAIN_BLOCKS, DEV_BLOCKS, TARGET_SAMPLE_RATE


def load_split(split: str = "train",
               streaming: bool = False) -> Dict[str, any]:
    """
    Load a split and return a dict of per-language Dataset slices.

        ds = load_split("train")
        english_ds = ds["english"]   # Dataset with ~13k rows
        hausa_ds   = ds["hausa"]     # Dataset with ~13k rows

    Parameters
    ----------
    split     : "train" or "dev"
    streaming : if True, returns IterableDataset (no download required,
                but random access and len() unavailable).
                Use False for training, True for quick inspection.

    Returns
    -------
    {language: Dataset} — audio is cast to 16kHz float32 numpy array.
    Each row has the dataset info fields.
    """
    blocks = TRAIN_BLOCKS if split == "train" else DEV_BLOCKS

    ds = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=streaming,
    )

    if streaming:
        # Streaming: filter by language field (full scan, but no download)
        lang_datasets = {}
        for lang in blocks:
            lang_datasets[lang] = ds.filter(
                lambda row, l=lang: row["language"] == l
            ).cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
        return lang_datasets

    # Non-streaming: slice by known block boundaries — O(1), no scan
    lang_datasets = {}
    for lang, (start, end) in blocks.items():
        slice_ds = ds.select(range(start, end))
        slice_ds = slice_ds.cast_column(
            "audio", Audio(sampling_rate=TARGET_SAMPLE_RATE)
        )
        lang_datasets[lang] = slice_ds

    return lang_datasets