from dataclasses import dataclass, field
import torch
from ssd.engine.sequence import Sequence
from abc import ABC, abstractmethod


@dataclass
class SpeculateResult:
    speculations: torch.Tensor
    logits_q: torch.Tensor
    cache_hits: torch.Tensor | None = None


@dataclass
class VerifyResult:
    new_suffixes: list[list[int]]
    recovery_tokens: list[int]
    eagle_acts: torch.Tensor | None = None  # Is this a tensor?
    # ------------------------------------------------------------------ #
    # Top-K target logits from the just-completed verification pass.      #
    # Shape: [B, K, top_k] for both vals and idxs.                       #
    # Only populated when use_phi=True on the speculator; None otherwise  #
    # so non-phi code paths are completely unaffected.                    #
    # ------------------------------------------------------------------ #
    target_top_k_vals: torch.Tensor | None = None  # [B, K, top_k]
    target_top_k_idxs: torch.Tensor | None = None  # [B, K, top_k]


class SpeculatorBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass

    @abstractmethod
    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass


class VerifierBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        pass

    @abstractmethod
    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        pass