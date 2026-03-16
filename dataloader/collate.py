import torch
from torch.utils.data import Sampler
from typing import List, Iterator


class DynamicBatchSampler(Sampler):
    """
    Groups dataset indices into batches such that the total audio
    duration per batch stays within max_duration_sec.

    Sorts by duration descending so the longest sequences are in
    the first batch — this surfaces OOM errors early in training.

    dataset         : ASRDataset or S2TTDataset instance
    max_duration_sec: maximum total seconds of audio per batch
    shuffle         : if True, shuffles the sorted groups each epoch
                      (preserves roughly-equal-length grouping)
    """

    def __init__(self, dataset, max_duration_sec: float = 120.0, shuffle: bool = False):
        self.dataset = dataset
        self.max_duration_sec = max_duration_sec
        self.shuffle = shuffle
        self._batches = self._build_batches()

    def _get_durations(self) -> List[tuple]:
        """Returns [(idx, duration_seconds)] for every item in the dataset."""
        ds = self.dataset

        # S2TTDataset stores pre-computed pairs list
        if hasattr(ds, "_pairs"):
            return [(i, ds._pairs[i]["duration"]) for i in range(len(ds))]

        # ASRDataset wraps a HuggingFace Dataset slice
        if hasattr(ds, "ds"):
            durs = ds.ds["duration"]
            return [(i, durs[i]) for i in range(len(ds))]

        raise ValueError(
            "Cannot extract durations from dataset. "
            "Expected ASRDataset or S2TTDataset."
        )

    def _build_batches(self) -> List[List[int]]:
        durations = self._get_durations()
        durations.sort(key=lambda x: x[1], reverse=True) 

        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_total = 0.0

        for idx, dur in durations:
            if current_total + dur > self.max_duration_sec and current_batch:
                batches.append(current_batch)
                current_batch = [idx]
                current_total = dur
            else:
                current_batch.append(idx)
                current_total += dur

        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # Shuffle batch order but keep within-batch duration grouping
            import random

            order = list(range(len(self._batches)))
            random.shuffle(order)
            for i in order:
                yield self._batches[i]
        else:
            yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)
