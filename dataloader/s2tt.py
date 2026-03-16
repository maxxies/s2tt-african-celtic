import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any

from .config import alignment_key


class S2TTDataset(Dataset):
    """
    Wraps a source-language slice for Stage 2 training.
    Provides (audio, english_text) pairs.

    Each item is a dict with keys:
        audio        : np.ndarray float32, shape (T,), 16kHz
        english_text : str  ← translation target
        source_text  : str  ← source language text (kept for reference)
        text_id      : str
        mapping_key  : str  ← text_id[1:] used for alignment
        user_id      : str
        language     : str
        duration     : float
    """

    def __init__(self, source_dataset, english_dataset):
        """
        Build from live dataset slices. Scans English slice once at init.

        source_dataset  : HuggingFace Dataset for a source language
                          (with audio cast to 16kHz)
        english_dataset : HuggingFace Dataset for English
                          (audio cast NOT needed — only text is used)
        """
        self.source_ds = source_dataset

        # Build English index: {alignment_key: english_text}
        # Use only text columns — no audio decoding needed for English
        print("  Building English text index ...", end="", flush=True)
        en_index: Dict[str, str] = {}
        for row in english_dataset:
            key = alignment_key(row["text_id"])
            if key:
                en_index[key] = row["text"]
        print(f" {len(en_index):,} sentences indexed.")

        self._pairs = self._build_pairs(source_dataset, en_index)
        self._report()

    @classmethod
    def from_pair_index(
        cls,
        source_dataset,
        pair_index_path: str,
        split: str,
    ) -> "S2TTDataset":
        """
        Build from a pre-computed pair_index.json (output of build_pair_index.py).
        No English dataset scan needed — fast construction.

        source_dataset   : HuggingFace Dataset for a source language
        pair_index_path  : path to pair_index.json
        split            : "train" or "dev"
        """
        with open(pair_index_path, "r", encoding="utf-8") as f:
            pair_index = json.load(f)

        split_index = pair_index.get(split, {})

        # Build English index from the pair_index
        # pair_index[split][key] = {lang: text}
        en_index: Dict[str, str] = {}
        for key, lang_texts in split_index.items():
            if "english" in lang_texts:
                en_index[key] = lang_texts["english"]

        instance = cls.__new__(cls)
        instance.source_ds = source_dataset
        instance._pairs = cls._build_pairs(source_dataset, en_index)
        instance._report(instance)
        return instance

    @staticmethod
    def _build_pairs(source_dataset, en_index: Dict[str, str]) -> List[Dict]:
        """
        Match each source row to its English translation via alignment_key.
        Returns list of pair dicts — only rows with a confirmed English match.
        """
        pairs = []
        skipped = []

        for i in range(len(source_dataset)):
            row = source_dataset[i]
            tid = row.get("text_id", "")
            key = alignment_key(tid)

            if key and key in en_index:
                pairs.append(
                    {
                        "source_idx": i,
                        "english_text": en_index[key],
                        "source_text": row["text"],
                        "text_id": tid,
                        "mapping_key": key,
                        "user_id": row["user_id"],
                        "language": row["language"],
                        "duration": row["duration"],
                    }
                )
            else:
                skipped.append(tid)

        return pairs

    def _report(self):
        n_src = len(self.source_ds)
        n_pairs = len(self._pairs)
        n_skip = n_src - n_pairs
        lang = self.source_ds[0]["language"] if n_src > 0 else "?"
        pct = 100 * n_pairs / max(1, n_src)
        print(
            f"  [{lang}] {n_pairs:,} / {n_src:,} rows paired "
            f"({pct:.1f}%)  —  {n_skip:,} skipped (no English match)"
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self._pairs[idx]
        row = self.source_ds[pair["source_idx"]]
        return {
            "audio": np.array(row["audio"]["array"], dtype=np.float32),
            "english_text": pair["english_text"],
            "source_text": pair["source_text"],
            "text_id": pair["text_id"],
            "mapping_key": pair["mapping_key"],
            "user_id": pair["user_id"],
            "language": pair["language"],
            "duration": pair["duration"],
        }

    def summary(self) -> Dict[str, Any]:
        n = len(self._pairs)
        durs = [p["duration"] for p in self._pairs]
        lang = self._pairs[0]["language"] if n > 0 else "?"
        src_codes = {}
        for p in self._pairs:
            src = (
                p["mapping_key"][: p["mapping_key"].index("_")]
                if "_" in p["mapping_key"]
                else "?"
            )
            src_codes[src] = src_codes.get(src, 0) + 1
        return {
            "source_language": lang,
            "paired_rows": n,
            "skipped_rows": len(self.source_ds) - n,
            "total_hours": round(sum(durs) / 3600, 2),
            "avg_duration_s": round(sum(durs) / n, 2) if n else 0,
            "source_codes": src_codes,
        }

    @staticmethod
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pad waveforms to the longest in the batch.

        Returns:
        audio          : (B, T_max) float32 tensor
        audio_mask     : (B, T_max) bool tensor
        audio_lengths  : (B,) int64 tensor
        english_text   : list[str]  ← translation target
        source_text    : list[str]
        text_id        : list[str]
        mapping_key    : list[str]
        user_id        : list[str]
        language       : list[str]
        duration       : (B,) float32 tensor
        """
        lengths = [len(item["audio"]) for item in batch]
        T_max = max(lengths)

        padded = torch.zeros(len(batch), T_max)
        mask = torch.zeros(len(batch), T_max, dtype=torch.bool)

        for i, item in enumerate(batch):
            n = lengths[i]
            padded[i, :n] = torch.from_numpy(item["audio"])
            mask[i, :n] = True

        return {
            "audio": padded,
            "audio_mask": mask,
            "audio_lengths": torch.tensor(lengths, dtype=torch.long),
            "english_text": [item["english_text"] for item in batch],
            "source_text": [item["source_text"] for item in batch],
            "text_id": [item["text_id"] for item in batch],
            "mapping_key": [item["mapping_key"] for item in batch],
            "user_id": [item["user_id"] for item in batch],
            "language": [item["language"] for item in batch],
            "duration": torch.tensor(
                [item["duration"] for item in batch], dtype=torch.float32
            ),
        }
