import json
import collections
from datasets import load_dataset
from config import DATASET_NAME, ALL_LANGUAGES, OUT_PATH



def build_index(split: str) -> dict:
    """
    Load the split, group rows by language, index by text_id[1:].
    Returns {lang: {shared_key: text}}
    """

    ds = load_dataset(DATASET_NAME, split=split, trust_remote_code=True)

    print(f"  {len(ds):,} rows loaded. Building index ...")

    index = {lang: {} for lang in ALL_LANGUAGES}
    skipped = 0

    for row in ds:
        lang = row.get("language")
        tid = row.get("text_id", "")
        text = row.get("text", "")

        if lang not in index:
            skipped += 1
            continue
        if not tid or len(tid) < 2 or "_" not in tid:
            skipped += 1
            continue

        key = tid[1:]  # strip language prefix → shared alignment key
        index[lang][key] = text

    if skipped:
        print(f"  ({skipped} rows skipped — unknown language or malformed text_id)")

    print(f"  Index built:")
    for lang in ALL_LANGUAGES:
        print(f"    [{lang:<10}] {len(index[lang]):,} keys")

    return index


def find_pairs(index: dict) -> dict:
    """
    Find all keys present in ≥2 languages.
    Returns {key: {lang: text}}
    """
    all_keys = set()
    for d in index.values():
        all_keys.update(d.keys())

    pairs = {}
    for key in all_keys:
        langs_with = {l: index[l][key] for l in ALL_LANGUAGES if key in index[l]}
        if len(langs_with) >= 2:
            pairs[key] = langs_with
    return pairs


def summarise(split: str, pairs: dict):
    """Print a concise summary of pair counts."""
    total = len(pairs)
    full4 = sum(1 for v in pairs.values() if len(v) == 4)
    three = sum(1 for v in pairs.values() if len(v) == 3)
    two = sum(1 for v in pairs.values() if len(v) == 2)

    by_src: dict = collections.Counter()
    for key in pairs:
        src = key[: key.index("_")] if "_" in key else "?"
        by_src[src] += 1

    print(f"\n  [{split}]")
    print(f"    Total aligned pairs  : {total:,}")
    print(f"    All 4 languages      : {full4:,}")
    print(f"    Any 3 languages      : {three:,}")
    print(f"    Any 2 languages      : {two:,}")
    print(f"    By source code       :", dict(by_src))


if __name__ == "__main__":
    print("  PAIR INDEX BUILDER — local dataset, complete data")
    print("=" * 55)

    full_output = {}

    for split in ["dev", "train"]:
        index = build_index(split)
        pairs = find_pairs(index)
        summarise(split, pairs)

        # Store only 4-language pairs in the output file
        full_output[split] = {k: v for k, v in pairs.items() if len(v) == 4}

    # Save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved → {OUT_PATH}")
    print(f"    dev   : {len(full_output['dev']):,} 4-language pairs")
    print(f"    train : {len(full_output['train']):,} 4-language pairs")
    print("=" * 55)
