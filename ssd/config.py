import os
from dataclasses import dataclass, field
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT

@dataclass
class Config:
    model: str = DEFAULT_TARGET
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    speculate_k: int = 1
    draft_async: bool = False
    
    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False 

    # ------------------------------------------------------------------ #
    # Adaptive phi parameters.                                            #
    # NOTE: trailing commas after default values were previously present  #
    # and caused each field to be a 1-tuple instead of a scalar — fixed. #
    # ------------------------------------------------------------------ #
    use_phi: bool = False
    phi_lr: float | None = None       # gradient step size; default set in __post_init__
    phi_beta: float | None = None     # EMA momentum; default set in __post_init__
    phi_max: float | None = None      # upper projection bound; default set in __post_init__
    top_k_target: int | None = None   # how many target top-K logits to recv; default set in __post_init__
    phi_lambda: float | None = None   # fidelity regularization weight (λ from adversarial loss); default set in __post_init__

    # eagle3
    use_eagle: bool = False 
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # Debugging
    verbose: bool = False 
    debug_mode: bool = False 
    max_steps: int | None = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    def __post_init__(self):
        model = self.model 
        assert os.path.isdir(model)

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        self.hf_config = AutoConfig.from_pretrained(model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings) 
        if self.speculate: 
            draft = self.draft
            self.draft_hf_config = AutoConfig.from_pretrained(draft)
            self.max_model_len = min(
                self.max_model_len, self.draft_hf_config.max_position_embeddings)
            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list 
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"

                # Pre-compute integer tensors used by DraftRunner._init_prealloc_buffers
                # and _build_tree_batch.  Kept on CPU here; DraftRunner moves them to
                # its device inside _init_prealloc_buffers via repeat_interleave.
                self.fan_out_t = torch.tensor(
                    self.fan_out_list, dtype=torch.int64)
                self.fan_out_t_miss = torch.tensor(
                    self.fan_out_list_miss, dtype=torch.int64)

        # ------------------------------------------------------------------ #
        # Phi defaults.  Only applied when use_phi=True so that             #
        # existing runs with use_phi=False are completely unaffected.        #
        # ------------------------------------------------------------------ #
        if self.use_phi:
            assert self.draft_async, "use_phi requires draft_async=True"
            if self.phi_lr is None:
                self.phi_lr = 0.01
            if self.phi_beta is None:
                self.phi_beta = 0.9
            if self.phi_max is None:
                self.phi_max = 3.0
            if self.top_k_target is None:
                self.top_k_target = 32
            if self.phi_lambda is None:
                self.phi_lambda = 0.5

        if self.use_eagle:
            if self.eagle_layers is None:
                L = self.hf_config.num_hidden_layers
                # self.eagle_layers = [3, L//2, L-3]
                self.eagle_layers = [2, L//2, L-3] # [2, 16, 29] outputs, ie. [3, L//2+1, L-2] inputs
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_rope_theta = getattr(self.hf_config, 'rope_theta', 500000.0)
                draft_rope_theta = getattr(self.draft_hf_config, 'rope_theta', 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    self.draft_hf_config.rope_theta = target_rope_theta
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft_hf_config.max_position_embeddings for RoPE.
                target_max_pos = getattr(self.hf_config, 'max_position_embeddings', 8192)
                draft_max_pos = getattr(self.draft_hf_config, 'max_position_embeddings', 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    self.draft_hf_config.max_position_embeddings = target_max_pos
        
        assert self.max_num_batched_tokens >= self.max_model_len