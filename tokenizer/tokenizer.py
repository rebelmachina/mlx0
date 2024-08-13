import torch
from typing import List, Optional
import string
import abc

import unicodedata
import time



class BPETokenizer:

    def __init__(vocab_size: int):
        pass

    def get_stats(self, ids: List[int]) -> dict:

        for p0,p1 in zip(ids, ids[1:]):
            stats = {}
            pair = (p0, p1)
            stats[pair] = stats.get(pair, 0) + 1

            return stats

    def train(self, text: str):
        pass

    def encode(self, text: str) -> torch.Tensor:
        
        text_tokens = text.encode("utf-8")
        ids = list(text_tokens)

        while len(ids) > 1:
            pass

    def decode(self, ids: List[int]) -> str:
        
        text_tokens = b"".join(self.vocab[idx] for idx in ids)
        return text_tokens.decode("utf-8", errors="replace")
    