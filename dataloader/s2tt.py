import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Any


def _alignment_key(text_id: str) -> Optional[str]:
    """Strip language prefix: 'HNX_0001' → 'NX_0001'"""
    return text_id[1:] if text_id and len(text_id) > 1 else None


class S2TTDataset(Dataset):
    """
    Wraps a source-language slice + English slice for Stage 2 training.

    At construction, builds an index {alignment_key: english_text} from
    the English slice. Each __getitem__ looks up the English translation
    for the source row. Rows with no English counterpart are skipped
    (stored in self.skipped_ids for inspection).

    Parameters
    source_dataset  : HuggingFace Dataset slice for a source language
    english_dataset : HuggingFace Dataset slice for English
    """

    def __init__(self, source_dataset, english_dataset):
        self.source_ds = source_dataset

        # Build English index: {alignment_key: english_text}
        print("  Building English text index ...", end="", flush=True)
        self._en_index: Dict[str, str] = {}
        for row in english_dataset:
            key = _alignment_key(row["text_id"])
            if key:
                self._en_index[key] = row["text"]
        print(f" done. {len(self._en_index):,} English sentences indexed.")

        # Build list of valid (source_row_idx, english_text) pairs
        self._pairs: List[Tuple[int, str]] = []
        self.skipped_ids: List[str] = []

        for i in range(len(source_dataset)):
            tid = source_dataset[i]["text_id"]
            key = _alignment_key(tid)
            if key and key in self._en_index:
                self._pairs.append((i, self._en_index[key]))
            else:
                self.skipped_ids.append(tid)

        pct_paired = 100 * len(self._pairs) / max(1, len(source_dataset))
        print(
            f"  Pairs found: {len(self._pairs):,} / {len(source_dataset):,} "
            f"({pct_paired:.1f}% paired)  "
            f"Skipped: {len(self.skipped_ids):,}"
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        src_idx, en_text = self._pairs[idx]
        row = self.source_ds[src_idx]
        audio = np.array(row["audio"]["array"], dtype=np.float32)
        return audio, en_text

    def summary(self) -> Dict[str, Any]:
        n = len(self._pairs)
        lang = self.source_ds[0]["language"] if len(self.source_ds) > 0 else "unknown"
        # Durations for paired rows only
        durs = [self.source_ds[i]["duration"] for i, _ in self._pairs]
        return {
            "source_language": lang,
            "paired_rows": n,
            "skipped_rows": len(self.skipped_ids),
            "total_hours": sum(durs) / 3600,
            "avg_duration_s": sum(durs) / n if n else 0,
        }

    @staticmethod
    def collate(batch: List[Tuple[np.ndarray, str]]):
        """
        Pad waveforms and return english texts as-is.
        Returns:
            waveforms  : (B, T_max) float32 tensor
            lengths    : (B,) int64 tensor
            en_texts   : list[str]
        """
        audios, en_texts = zip(*batch)
        lengths = torch.tensor([len(a) for a in audios], dtype=torch.long)
        T_max = lengths.max().item()

        padded = torch.zeros(len(audios), T_max)
        for i, a in enumerate(audios):
            padded[i, : len(a)] = torch.from_numpy(a)

        return padded, lengths, list(en_texts)
