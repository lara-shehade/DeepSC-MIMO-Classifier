"""Dataset loading without machine-specific paths or preprocessing requirements."""

from pathlib import Path
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent


def candidate_data_files(split, data_root=None):
    """Return supported locations for existing ``train_data.pkl``/``test_data.pkl``."""
    filename = "{}_data.pkl".format(split)
    roots = []
    if data_root:
        roots.append(Path(data_root).expanduser())
    env_root = os.environ.get("DEEPSC_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(PROJECT_ROOT)

    candidates = []
    for root in roots:
        candidates.extend((root / filename, root / "europarl" / filename, root / "txt" / filename))
    return list(dict.fromkeys(path.resolve() for path in candidates))


def resolve_data_file(split="train", data_root=None):
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    candidates = candidate_data_files(split, data_root)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find {}_data.pkl. No preprocessing or download is performed. "
        "Copy the server pickle into one of these locations or pass --data-root:\n  - {}"
        .format(split, searched)
    )


class EurDataset(Dataset):
    def __init__(self, split="train", data_root=None, indexed=False):
        self.path = resolve_data_file(split, data_root)
        self.indexed = bool(indexed)
        with self.path.open("rb") as handle:
            self.data = pickle.load(handle)

    def __getitem__(self, index):
        tokens = self.data[index]
        if self.indexed:
            return {"sample_id": index, "tokens": tokens}
        return tokens

    def __len__(self):
        return len(self.data)


def collate_data(batch):
    if not batch:
        raise ValueError("cannot collate an empty batch")
    ordered = sorted(batch, key=len, reverse=True)
    max_len = max(map(len, ordered))
    sents = np.zeros((len(ordered), max_len), dtype=np.int64)
    for row, sent in enumerate(ordered):
        sents[row, : len(sent)] = sent
    return torch.from_numpy(sents)


def collate_indexed_data(batch):
    if not batch:
        raise ValueError("cannot collate an empty batch")
    ordered = sorted(batch, key=lambda item: len(item["tokens"]), reverse=True)
    max_len = max(len(item["tokens"]) for item in ordered)
    sents = np.zeros((len(ordered), max_len), dtype=np.int64)
    sample_ids = np.zeros(len(ordered), dtype=np.int64)
    lengths = np.zeros(len(ordered), dtype=np.int64)
    for row, item in enumerate(ordered):
        tokens = item["tokens"]
        sents[row, : len(tokens)] = tokens
        sample_ids[row] = item["sample_id"]
        lengths[row] = len(tokens)
    return torch.from_numpy(sents), torch.from_numpy(sample_ids), torch.from_numpy(lengths)


__all__ = [
    "EurDataset",
    "candidate_data_files",
    "collate_data",
    "collate_indexed_data",
    "resolve_data_file",
]
