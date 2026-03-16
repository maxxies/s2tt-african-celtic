import os
import json
from torch.utils.data import DataLoader

from dataloader import load_language_slices, ASRDataset, S2TTDataset
from dataloader.collate import DynamicBatchSampler
from dataloader.stats import print_split_stats, print_pair_coverage
from dataloader.build_pair_index import build_index, find_pairs, summarise
from dataloader.config import SOURCE_LANGUAGES, OUT_PATH


# Step One — Build pair index (one-time setup)
def build_pair_index():
    """
    Downloads the full dataset and builds pair_index.json.
    Skipped automatically if pair_index.json already exists.
    """
    if os.path.exists(OUT_PATH):
        print(f"[Step 1] pair_index.json already exists — skipping build.")
        return

    print("[Step 1] Building pair index (downloads ~75 GB on first run) ...")

    output = {}
    for split in ["dev", "train"]:
        index = build_index(split)
        pairs = find_pairs(index)
        summarise(split, pairs)
        output[split] = {k: v for k, v in pairs.items() if len(v) == 4}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Step 1] Done. Saved → {OUT_PATH}")
    print(f"         train: {len(output['train']):,} 4-language pairs")
    print(f"         dev  : {len(output['dev']):,} 4-language pairs")



# STEP 2 — Load data
def load_data(split: str = "train"):
    """
    Loads a split and returns per-language dataset slices.

    source_slices : {lang: Dataset}  — with audio cast to 16kHz
    english_slice : Dataset          — text only, no audio decoded
    """
    print(f"\n[Step 2] Loading {split} split ...")

    source_slices = load_language_slices(
        split,
        languages=SOURCE_LANGUAGES,
        cast_audio=True,
    )
    english_slice = load_language_slices(
        split,
        languages=["english"],
        cast_audio=False,  # English is only needed for its text
    )["english"]

    print(f"[Step 2] Loaded:")
    for lang, ds in source_slices.items():
        print(f"         [{lang:<10}] {len(ds):,} rows")
    print(f"         [english   ] {len(english_slice):,} rows (text only)")

    return source_slices, english_slice


# STEP 3 — Create datasets
def create_asr_datasets(source_slices: dict) -> dict:
    """
    Wraps each source language slice in ASRDataset for Stage 1 training.
    Returns {lang: ASRDataset}
    """
    print(f"\n[Step 3a] Creating ASR datasets (Stage 1) ...")

    asr_datasets = {}
    for lang, ds in source_slices.items():
        asr_datasets[lang] = ASRDataset(ds)
        print(f"         [{lang}] {len(asr_datasets[lang]):,} samples")

    return asr_datasets


def create_s2tt_datasets(source_slices: dict, split: str) -> dict:
    """
    Wraps each source language slice in S2TTDataset for Stage 2 training.
    Pairs source audio with English text via pair_index.json.
    Returns {lang: S2TTDataset}
    """
    print(f"\n[Step 3b] Creating S2TT datasets (Stage 2) ...")

    s2tt_datasets = {}
    for lang, ds in source_slices.items():
        s2tt_datasets[lang] = S2TTDataset.from_pair_index(
            source_dataset=ds,
            OUT_PATH=OUT_PATH,
            split=split,
        )

    print()
    print_pair_coverage(list(s2tt_datasets.values()))
    return s2tt_datasets


# STEP 4 — Create DataLoaders
def step4_create_loaders(
    datasets: dict,
    stage: str = "s2tt",
    max_duration_sec: float = 120.0,
    num_workers: int = 4,
) -> dict:
    """
    Wraps datasets in DataLoaders with DynamicBatchSampler.
    
    Parameters:
    datasets         : {lang: ASRDataset or S2TTDataset}
    stage            : "asr" or "s2tt" — determines collate function
    max_duration_sec : max total audio seconds per batch (~120s ≈ 8-12 samples)
    num_workers      : DataLoader workers

    Returns
    {lang: DataLoader}
    """
    print(
        f"\n[Step 4] Creating DataLoaders ({stage.upper()}, "
        f"max_duration={max_duration_sec}s) ..."
    )

    collate_fn = ASRDataset.collate if stage == "asr" else S2TTDataset.collate

    loaders = {}
    for lang, ds in datasets.items():
        sampler = DynamicBatchSampler(
            ds, max_duration_sec=max_duration_sec, shuffle=True
        )
        loaders[lang] = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"         [{lang:<10}] {len(sampler):,} batches")

    return loaders

# VERIFY — sanity check one batch from each loader
def verify_loaders(asr_loaders: dict, s2tt_loaders: dict):
    """Fetch one batch from each loader and print shapes."""
    print("\n[Verify] Sampling one batch per language ...")

    print("\n  ASR loaders:")
    for lang, loader in asr_loaders.items():
        batch = next(iter(loader))
        print(
            f"  [{lang:<10}] "
            f"audio={tuple(batch['audio'].shape)}  "
            f"text={batch['text'][0][:50]!r}"
        )

    print("\n  S2TT loaders:")
    for lang, loader in s2tt_loaders.items():
        batch = next(iter(loader))
        print(
            f"  [{lang:<10}] "
            f"audio={tuple(batch['audio'].shape)}  "
            f"en={batch['english_text'][0][:50]!r}"
        )

# MAIN
if __name__ == "__main__":
    print("=" * 60)
    print("  African S2TT — Data Pipeline")
    print("=" * 60)

    # Step 1: build pair index once
    build_pair_index()

    # Steps 2–4: train split
    source_slices, english_slice = load_data("train")

    asr_datasets = create_asr_datasets(source_slices)
    s2tt_datasets = create_s2tt_datasets(source_slices, split="train")

    asr_loaders = step4_create_loaders(asr_datasets, stage="asr")
    s2tt_loaders = step4_create_loaders(s2tt_datasets, stage="s2tt")

    verify_loaders(asr_loaders, s2tt_loaders)

    # Dev split stats
    print("\n[Dev split stats]")
    dev_slices, _ = load_data("dev")
    print_split_stats(
        "dev",
        {
            **dev_slices,
            "english": load_language_slices("dev", ["english"], cast_audio=False)[
                "english"
            ],
        },
    )
