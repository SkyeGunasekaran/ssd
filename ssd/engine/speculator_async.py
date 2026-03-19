import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.helpers.runner_helpers import prepare_prefill_payload
from ssd.engine.sequence import Sequence
from ssd.utils.misc import decode_tokens
from ssd.utils.async_helpers.nccl_pack import send_int64, send_float32


class SpeculatorAsync(SpeculatorBase):

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        async_fan_out: int,
        max_blocks: int,
        vocab_size: int,
        draft_dtype: torch.dtype,
        kvcache_block_size: int,
        max_model_len: int,
        async_pg: dist.ProcessGroup,
        draft_runner_rank: int,
        tokenizer: AutoTokenizer,
        verbose: bool,
        # ------------------------------------------------------------------ #
        # Adaptive phi parameters (all optional — defaults leave behaviour    #
        # identical to the original Saguaro method when use_phi=False).      #
        # use_phi MUST match the same flag on Config so that the draft-side  #
        # recv in DraftRunner._service_spec_request is also gated identically#
        # and the NCCL send/recv pair is always balanced.                    #
        # ------------------------------------------------------------------ #
        use_phi: bool = False,          # master gate — disables all phi logic when False
        phi_lr: float = 0.01,           # gradient step size
        phi_beta: float = 0.9,          # EMA momentum coefficient
        phi_max: float = 3.0,           # hard clip upper bound (conservative)
        top_k_target: int = 32,         # how many top-K target logits to use
    ):
        super().__init__(lookahead, device)
        self.async_fan_out = async_fan_out
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.kvcache_block_size = kvcache_block_size
        self.max_model_len = max_model_len
        self.async_pg = async_pg
        self.draft_runner_rank = draft_runner_rank
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.K = lookahead

        # ------------------------------------------------------------------ #
        # Phi state.  Only allocated when use_phi=True so that non-phi runs  #
        # have zero overhead and the send in _speculation_request is never   #
        # reached, keeping the protocol in sync with DraftRunner which also  #
        # gates its recv on use_phi.                                          #
        # ------------------------------------------------------------------ #
        self.use_phi = use_phi
        self.phi_lr      = phi_lr
        self.phi_beta    = phi_beta
        self.phi_max     = phi_max
        self.top_k_target = top_k_target

        if use_phi:
            F = async_fan_out
            self.phi          = torch.zeros(F, dtype=torch.float32, device=device)
            self.phi_momentum = torch.zeros(F, dtype=torch.float32, device=device)
            # Pre-allocated buffer so the phi send never triggers a new allocation
            self._phi_buf = torch.zeros(F, dtype=torch.float32, device=device)

        # Pre-allocate handshake send/recv buffers (reused every step)
        self._alloc_handshake_bufs(1)

        # Pre-allocate speculate() output buffers (avoid torch.tensor(device=cuda) sync)
        self._recovery_buf = torch.empty(1, dtype=torch.int64, device=device)
        self._speculations_buf = torch.empty(1, lookahead + 1, dtype=torch.int64, device=device)

    # ---------------------------------------------------------------------- #
    # Phi gradient + update helpers                                           #
    # ---------------------------------------------------------------------- #

    def _compute_phi_grad(
        self,
        draft_logits: torch.Tensor,        # [B, K, V]  raw logits received from draft
        target_top_k_vals: torch.Tensor,   # [B, K, top_k]  target logit values
        target_top_k_idxs: torch.Tensor,   # [B, K, top_k]  corresponding token indices
    ) -> torch.Tensor:                     # [F]
        """
        Gradient of the full-vocabulary cross-entropy w.r.t. phi.

        For each (batch, draft-step) pair we identify S_F (top-F tokens in the
        raw draft logits), apply the current phi to get shaped distribution q̃,
        then approximate the target distribution p̂ on those positions from the
        provided top-K target logits.

        Gradient at rank j:
            g_j = mean_{b,k} [ q̃_j^{b,k}  −  p̂_j^{b,k} ]

        Positive g_j → draft over-represents rank-j relative to target →
        increase phi[j] to push more residual mass there.
        Negative g_j → over-penalised → decrease phi[j].
        """
        F = self.async_fan_out
        B, K, V = draft_logits.shape

        with torch.no_grad():
            draft_f = draft_logits.float()    # work in float32 throughout

            # top-F indices per (b, k): [B, K, F]
            top_f_idxs = draft_f.topk(F, dim=-1).indices

            # Apply current phi to get shaped logits (full vocab)
            shaped = draft_f.clone()
            phi_bkf = self.phi.view(1, 1, F).expand(B, K, F)
            shaped.scatter_add_(-1, top_f_idxs, -phi_bkf)

            # Full-vocabulary softmax → q̃, then extract at top-F positions
            q_tilde      = torch.softmax(shaped, dim=-1)          # [B, K, V]
            q_tilde_topf = q_tilde.gather(-1, top_f_idxs)         # [B, K, F]

            # Approximate p̂ at top-F positions via lookup into target top-K.
            # Tokens that fall outside the target top-K get p̂ = 0 (sparse
            # approximation; covers ≥70-90 % of residual mass in practice).
            p_target = torch.softmax(target_top_k_vals.float(), dim=-1)   # [B, K, top_k]

            # match[b, k, j, t] = 1 iff top_f_idxs[b,k,j] == target_top_k_idxs[b,k,t]
            match = (
                top_f_idxs.unsqueeze(-1)          # [B, K, F, 1]
                == target_top_k_idxs.unsqueeze(-2) # [B, K, 1, top_k]
            )                                      # [B, K, F, top_k]
            p_hat_topf = (match.float() * p_target.unsqueeze(-2)).sum(-1)  # [B, K, F]

            # Mean gradient over batch and draft steps
            grad = (q_tilde_topf - p_hat_topf).mean(dim=(0, 1))   # [F]

        return grad

    def _step_phi(self, grad: torch.Tensor):
        """EMA-smoothed gradient step with [0, phi_max] projection."""
        self.phi_momentum.mul_(self.phi_beta).add_(
            grad.to(self.device), alpha=1.0 - self.phi_beta
        )
        self.phi.add_(self.phi_lr * self.phi_momentum).clamp_(0.0, self.phi_max)

    # ---------------------------------------------------------------------- #

    def _alloc_handshake_bufs(self, B):
        self._hs_B = B
        d = self.device
        self._cmd = torch.zeros(1, dtype=torch.int64, device=d)
        self._meta = torch.tensor([B, self.K, self.async_fan_out], dtype=torch.int64, device=d)
        self._cache_keys = torch.empty(B, 3, dtype=torch.int64, device=d)
        self._num_tokens_buf = torch.empty(B, dtype=torch.int64, device=d)
        self._temps_buf = torch.empty(B, dtype=torch.float32, device=d)
        self._block_tables_buf = torch.full((B, self.max_blocks), -1, dtype=torch.int32, device=d)
        self._fused_response = torch.empty(B + B * self.K, dtype=torch.int64, device=d)
        self._logits_q = torch.empty(B, self.K, self.vocab_size, dtype=self.draft_dtype, device=d)
        self._extend_counts = torch.zeros(B, dtype=torch.int64, device=d)

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        eagle_acts = verify_result.eagle_acts
        input_id_list = [seq.token_ids for seq in seqs]

        # EAGLE token-conditioning shift: token at position j gets conditioning
        # from target act at position j-1. Skip first token per seq and drop
        # last eagle_act per seq so they align correctly.
        if eagle_acts is not None:
            sliced = []
            offset = 0
            for ids in input_id_list:
                seq_len = len(ids)
                sliced.append(eagle_acts[offset:offset + seq_len - 1])
                offset += seq_len
            eagle_acts = torch.cat(sliced, dim=0)
            input_id_list = [ids[1:] for ids in input_id_list]

        max_blocks = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
        cmd, metadata, input_ids, num_tokens, draft_block_table, eagle_acts = prepare_prefill_payload(
            input_id_list, eagle_acts, self.device, max_blocks,
            [seq.draft_block_table for seq in seqs],
        )
        dist.send(cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(metadata, dst=self.draft_runner_rank, group=self.async_pg)
        send_int64(self.async_pg, self.draft_runner_rank,
                   input_ids, num_tokens, draft_block_table.to(torch.int64))
        if eagle_acts is not None:
            dist.send(eagle_acts, dst=self.draft_runner_rank, group=self.async_pg)
        return SpeculateResult([], [])

    def speculate(
        self,
        seqs: list[Sequence],
        verify_result: VerifyResult,
        # ------------------------------------------------------------------ #
        # Optional: top-K target logits from the just-completed verification. #
        # When provided and use_phi=True, phi is updated before the next      #
        # speculation round.  Both tensors must live on self.device.          #
        # Shape: [B, K, top_k_target] — one entry per sequence per draft     #
        # step, covering the top_k_target highest-probability target tokens.  #
        # ------------------------------------------------------------------ #
        target_top_k_vals: torch.Tensor | None = None,  # [B, K, top_k]
        target_top_k_idxs: torch.Tensor | None = None,  # [B, K, top_k]
    ) -> SpeculateResult:
        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.append_token(seq.recovery_token_id)

        if self.verbose:
            sep = '=' * 80
            print(f"\n{sep}", flush=True)
            print(f"[TARGET SEQUENCE TRUNK] Batch size: {len(seqs)}", flush=True)
            for i, seq in enumerate(seqs):
                trunk = seq.token_ids[-20:] if len(seq.token_ids) > 20 else seq.token_ids
                print(f"  Seq {seq.seq_id} (len={len(seq.token_ids)}):", flush=True)
                print(f"    Trunk: ...{decode_tokens(trunk, self.tokenizer)}", flush=True)
                print(f"    Recovery: {seq.recovery_token_id} ({decode_tokens([seq.recovery_token_id], self.tokenizer)})", flush=True)
            print(f"{sep}\n", flush=True)

        eagle = verify_result.eagle_acts is not None
        speculations_tokens, logits_q, cache_hits = self._speculation_request(seqs, eagle)

        # ------------------------------------------------------------------ #
        # Phi update.                                                         #
        # We update phi AFTER receiving this round's draft logits, using the  #
        # target logits from the verification that triggered this call.       #
        # The updated phi is already in self.phi and will be sent at the      #
        # START of the next _speculation_request call.                        #
        # Only runs when use_phi=True AND the caller supplied top-K logits    #
        # (they are None on the first step or when use_phi=False).           #
        # ------------------------------------------------------------------ #
        if self.use_phi and target_top_k_vals is not None and target_top_k_idxs is not None:
            grad = self._compute_phi_grad(
                logits_q.float(),           # [B, K, V]
                target_top_k_vals,          # [B, K, top_k]
                target_top_k_idxs,          # [B, K, top_k]
            )
            self._step_phi(grad)

        # Build speculations using pre-allocated buffers (avoids torch.tensor(device=cuda) sync)
        B = len(seqs)
        if B != self._recovery_buf.shape[0]:
            self._recovery_buf = torch.empty(B, dtype=torch.int64, device=self.device)
            self._speculations_buf = torch.empty(B, self.K + 1, dtype=torch.int64, device=self.device)
        _rec_cpu = torch.tensor([seq.recovery_token_id for seq in seqs], dtype=torch.int64)
        self._recovery_buf.copy_(_rec_cpu, non_blocking=True)
        self._speculations_buf[:, 0] = self._recovery_buf
        self._speculations_buf[:, 1:] = speculations_tokens
        speculations = self._speculations_buf

        for i, seq in enumerate(seqs):
            seq.token_ids.extend(speculations_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.num_draft_cached_tokens += len(speculations_tokens[i]) + 1

        return SpeculateResult(speculations, logits_q, cache_hits)

    def _speculation_request(self, seqs: list[Sequence], eagle: bool):
        B = len(seqs)
        if B != self._hs_B:
            self._alloc_handshake_bufs(B)

        # Fill send buffers in-place (avoids torch.tensor from Python lists)
        for i, seq in enumerate(seqs):
            self._cache_keys[i, 0] = seq.seq_id
            self._cache_keys[i, 1] = seq.last_spec_step_accepted_len - 1
            self._cache_keys[i, 2] = seq.recovery_token_id
            self._num_tokens_buf[i] = seq.num_tokens
            self._temps_buf[i] = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            bt = seq.draft_block_table
            bt_len = len(bt)
            if bt_len > 0:
                self._block_tables_buf[i, :bt_len] = torch.tensor(bt, dtype=torch.int32, device=self.device)
            self._block_tables_buf[i, bt_len:] = -1

        # Send cmd + meta + fused payload (temps fused into int64 burst)
        dist.send(self._cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._meta, dst=self.draft_runner_rank, group=self.async_pg)
        temps_as_int64 = self._temps_buf.view(torch.int32).to(torch.int64)
        send_int64(
            self.async_pg, self.draft_runner_rank,
            self._cache_keys, self._num_tokens_buf,
            self._block_tables_buf.to(torch.int64), temps_as_int64,
        )

        if eagle:
            recovery_activations = torch.stack(
                [seq.last_target_hidden_state for seq in seqs], dim=0,
            ).to(self.device)
            dist.send(recovery_activations.to(self.draft_dtype),
                      dst=self.draft_runner_rank, group=self.async_pg)

            # Send extend data for glue decode with fused extend
            K = self.K
            act_dim = recovery_activations.shape[-1]
            for i, seq in enumerate(seqs):
                self._extend_counts[i] = seq.extend_count
            extend_eagle_acts = torch.zeros(B, K, act_dim, dtype=self.draft_dtype, device=self.device)
            extend_token_ids = torch.zeros(B, K, dtype=torch.int64, device=self.device)
            for i, seq in enumerate(seqs):
                n = seq.extend_count
                if n > 0 and seq.extend_eagle_acts is not None:
                    extend_eagle_acts[i, :n] = seq.extend_eagle_acts[:n].to(self.draft_dtype)
                    extend_token_ids[i, :n] = seq.extend_token_ids[:n]
            dist.send(self._extend_counts, dst=self.draft_runner_rank, group=self.async_pg)
            dist.send(extend_eagle_acts, dst=self.draft_runner_rank, group=self.async_pg)
            dist.send(extend_token_ids, dst=self.draft_runner_rank, group=self.async_pg)

        # ------------------------------------------------------------------ #
        # Send current phi to draft — ONLY when use_phi=True.                #
        # The draft-side recv in DraftRunner._service_spec_request is also   #
        # gated on config.use_phi, so the send/recv pair is always balanced  #
        # and there is no risk of NCCL deadlock on non-phi runs.             #
        # ------------------------------------------------------------------ #
        if self.use_phi:
            self._phi_buf.copy_(self.phi, non_blocking=False)
            send_float32(self.async_pg, self.draft_runner_rank, self._phi_buf)

        # Recv into pre-allocated buffers
        dist.recv(self._fused_response, src=self.draft_runner_rank, group=self.async_pg)
        cache_hits = self._fused_response[:B]
        speculations = self._fused_response[B:].view(B, self.K)
        dist.recv(self._logits_q, src=self.draft_runner_rank, group=self.async_pg)

        return speculations, self._logits_q, cache_hits
