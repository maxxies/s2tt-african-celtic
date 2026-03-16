import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any


class ASRDataset(Dataset):
    """
    Wraps one language slice for Stage 1 (ASR) training.

    Each item is (waveform: np.ndarray float32, text: str).
    The audio is already resampled to 16kHz by load_split().
    """

    def __init__(self, lang_dataset):
        """
        lang_dataset : HuggingFace Dataset slice for one language
                       (output of load_split()[lang])
        """
        self.ds = lang_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        row = self.ds[idx]
        audio = np.array(row["audio"]["array"], dtype=np.float32)
        text = row["text"]
        return audio, text

    def summary(self) -> Dict[str, Any]:
        """Quick stats without iterating the whole dataset."""
        n = len(self.ds)
        lang = self.ds[0]["language"] if n > 0 else "unknown"
        users = set(self.ds[i]["user_id"] for i in range(min(n, 500)))
        dur_col = self.ds["duration"]
        return {
            "language": lang,
            "rows": n,
            "unique_speakers": len(users),
            "total_hours": sum(dur_col) / 3600,
            "avg_duration_s": sum(dur_col) / n,
            "min_duration_s": min(dur_col),
            "max_duration_s": max(dur_col),
        }

    @staticmethod
    def collate(batch: List[Tuple[np.ndarray, str]]):
        """
        Pad waveforms to the longest in the batch.
        Returns:
            waveforms : (B, T_max) float32 tensor
            lengths   : (B,) int64 tensor — actual length of each waveform
            texts     : list[str]
        """
        audios, texts = zip(*batch)
        lengths = torch.tensor([len(a) for a in audios], dtype=torch.long)
        T_max = lengths.max().item()

        padded = torch.zeros(len(audios), T_max)
        for i, a in enumerate(audios):
            padded[i, : len(a)] = torch.from_numpy(a)

        return padded, lengths, list(texts)
