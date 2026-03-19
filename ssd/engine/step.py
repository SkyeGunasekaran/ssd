from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import torch
from time import perf_counter
from transformers import AutoTokenizer

from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase, VerifyResult
from ssd.utils.misc import decode_tokens


class InferenceStep(ABC):

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def decode(self, seqs: list[Sequence]) -> int:
        pass

    @abstractmethod
    def prefill(self, seqs: list[Sequence]) -> int:
        pass


class AutoRegressiveStep(InferenceStep):

    def __init__(self, scheduler: Scheduler, model_runner: ModelRunner, tokenizer: AutoTokenizer):
        super().__init__(scheduler)
        self.model_runner = model_runner
        self.tokenizer = tokenizer

    def step(self, seqs: list[Sequence], is_prefill: bool) -> int:
        if __debug__:
            print(f'[auto_regressive_step] is_prefill={is_prefill}', flush=True)

        token_ids = self.model_runner.call("run", seqs, is_prefill)

        if __debug__:
            decoded_tokens = decode_tokens(token_ids, self.tokenizer)
            print(f"[auto_regressive_step] generated tokens: {decoded_tokens}", flush=True)

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return len(seqs) if not is_prefill else sum(len(seq) for seq in seqs)

    def prefill(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=True)

    def decode(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=False)


class SpecDecodeStep(InferenceStep):

    def __init__(
        self,
        scheduler: Scheduler,
        speculator: SpeculatorBase,
        verifier: VerifierBase,
        eagle: bool,
        tokenizer: AutoTokenizer,
        async_spec: bool,
    ):
        super().__init__(scheduler)
        self.speculator = speculator
        self.verifier = verifier
        self.eagle = eagle
        self.tokenizer = tokenizer
        self.async_spec = async_spec

        # ------------------------------------------------------------------ #
        # Stash top-K target logits from the previous verification step so   #
        # they can be forwarded into speculate() on the *next* decode step.  #
        # This is a 1-step delay: phi is updated with logits from round N-1  #
        # before speculation begins in round N, which is correct and         #
        # expected for an online gradient method.                             #
        # Both are None on the first step (no prior verification exists).    #
        # ------------------------------------------------------------------ #
        self._prev_top_k_vals = None  # [B, K, top_k] or None
        self._prev_top_k_idxs = None  # [B, K, top_k] or None

    def prefill(self, seqs: list[Sequence]) -> int:
        # When doing async speculation and not Eagle, we can do draft and target prefills in parallel.
        if not self.eagle and self.async_spec:
            empty_verify_result = VerifyResult([], [], None)
            self.speculator.prefill(seqs, empty_verify_result)
            verify_result = self.verifier.prefill(seqs, eagle=False)
        else:
            verify_result = self.verifier.prefill(seqs, eagle=self.eagle)
            self.speculator.prefill(seqs, verify_result)

        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.num_draft_cached_tokens = seq.num_prompt_tokens

        # Prefill does not produce draft-step logits, so clear any stale state.
        self._prev_top_k_vals = None
        self._prev_top_k_idxs = None

        return sum(len(seq) for seq in seqs)

    def decode(self, seqs: list[Sequence]) -> int:
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        if _prof:
            torch.cuda.synchronize()
            _t0 = perf_counter()

        # Save lightweight state instead of expensive clone_spec deep copy.
        # speculate() modifies: token_ids (append+extend), num_tokens, last_token, num_draft_cached_tokens
        # verify() modifies: num_cached_tokens (line 77 of verifier.py)
        # postprocess_speculate() needs the ORIGINAL state to apply new suffixes.
        saved = [(len(seq.token_ids), seq.num_tokens, seq.last_token, seq.num_draft_cached_tokens, seq.num_cached_tokens) for seq in seqs]

        eagle_sentinel = True if self.eagle else None
        in_verify_result = VerifyResult(
            new_suffixes=[],
            recovery_tokens=[],
            eagle_acts=eagle_sentinel,
        )

        #### STEP 1: SPECULATE ####
        # Forward top-K logits from the *previous* verification into speculate()
        # so SpeculatorAsync can update phi before sending it to the draft.
        # On the first decode step both are None and phi stays at its zero
        # initialisation, which is the correct default.
        speculate_result = self.speculator.speculate(
            seqs,
            in_verify_result,
            target_top_k_vals=self._prev_top_k_vals,
            target_top_k_idxs=self._prev_top_k_idxs,
        )

        if _prof:
            torch.cuda.synchronize()
            _t1 = perf_counter()

        if __debug__:
            speculations = speculate_result.speculations
            print(f"[SpecDecodeStep] speculations: {speculations}", flush=True)
            speculations_list = speculations.tolist()

            for i, speculation in enumerate(speculations_list):
                decoded_tokens = decode_tokens(speculation, self.tokenizer)
                print(f"[SpecDecodeStep] speculation {i}: {decoded_tokens}", flush=True)

        #### STEP 2: VERIFY ####
        out_verify_result = self.verifier.verify(seqs, speculate_result, eagle=self.eagle)

        # Stash top-K logits for the *next* decode step's phi update.
        # These may be None when use_phi=False (top_k_target==0 on the verifier).
        self._prev_top_k_vals = out_verify_result.target_top_k_vals
        self._prev_top_k_idxs = out_verify_result.target_top_k_idxs

        if _prof:
            torch.cuda.synchronize()
            _t2 = perf_counter()

        if __debug__:
            recovery_tokens = out_verify_result.recovery_tokens
            new_suffixes = out_verify_result.new_suffixes
            for i, new_suffix in enumerate(new_suffixes):
                decoded_tokens = decode_tokens(new_suffix + [recovery_tokens[i]], self.tokenizer)
                print(f"[SpecDecodeStep] verification {i}: {decoded_tokens}", flush=True)

        # Restore original seq state before postprocess (undo speculate + verify modifications)
        for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
            del seq.token_ids[orig_len:]
            seq.num_tokens = orig_nt
            seq.last_token = orig_lt
            seq.num_draft_cached_tokens = orig_ndc
            seq.num_cached_tokens = orig_nct

        #### STEP 3: POSTPROCESS ####
        self.scheduler.postprocess_speculate(
            seqs,
            out_verify_result.new_suffixes,
            out_verify_result.recovery_tokens,
            eagle_acts=out_verify_result.eagle_acts if self.eagle else None,
        )

        if _prof:
            torch.cuda.synchronize()
            _t3 = perf_counter()
            cache_hits = speculate_result.cache_hits
            hits_str = f"hits={cache_hits.sum().item()}/{len(cache_hits)}" if cache_hits is not None else ""
            toks = sum(len(s) for s in out_verify_result.new_suffixes)
            print(f"[PROFILE target] handshake={(_t1-_t0)*1000:.2f}ms verify={(_t2-_t1)*1000:.2f}ms postprocess={(_t3-_t2)*1000:.2f}ms total={(_t3-_t0)*1000:.2f}ms {hits_str} toks={toks}", flush=True)

        return sum(len(s) for s in out_verify_result.new_suffixes)