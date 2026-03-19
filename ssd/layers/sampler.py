import torch
from torch import nn

from ssd.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling

torch.manual_seed(0) 

class Sampler(nn.Module): 
    def __init__(self, sampler_x: float | None = None, async_fan_out: int = 3):
        super().__init__()
        self.sampler_x = sampler_x
        self.F = async_fan_out # will need to accomodate lists for hit/miss eventually

        # Adaptive phi vector for online logit shaping.
        # phi[j] is subtracted from the j-th ranked logit in S_F (the top-F set)
        # before softmax.  Starts as None (disabled); set via set_phi() each round
        # by DraftRunner after receiving the updated vector from the target process.
        self.phi: torch.Tensor | None = None

    def set_phi(self, phi: torch.Tensor):
        """Store updated phi.  Called by DraftRunner after recv-ing from target."""
        self.phi = phi.to(dtype=torch.float32)

    @torch.inference_mode() # what shape are logits during tree decode? MQ_LEN, 
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, is_tree: bool = False):
        # logits: [N, V] where N = B*MQ_LEN during tree decode, B during normal decode
        
        logits_cpy = logits.to(torch.float) 
        greedy_tokens = logits_cpy.argmax(dim=-1)

        # Fast path: any zero temperature rows are greedy
        temps = temperatures
        zero_mask = temps == 0

        # ------------------------------------------------------------------ #
        # Adaptive phi subtraction (new).                                     #
        # Applied to logits BEFORE temperature scaling so the effect is       #
        # consistent regardless of temperature.  phi[j] is subtracted from    #
        # the token at draft rank j for every row in the batch independently. #
        # Only active during tree decode (is_tree=True) when phi is set.      #
        # Completely skipped when phi is None — zero overhead on base path.   #
        # ------------------------------------------------------------------ #
        if self.phi is not None and is_tree and logits_cpy.dim() == 2:
            F = self.phi.shape[0]
            # top_f_idxs: [N, F] — indices of the F highest logits per row
            top_f_idxs = logits_cpy.topk(F, dim=-1).indices
            # phi_row: [N, F] — broadcast phi across all rows
            phi_row = self.phi.to(device=logits_cpy.device).unsqueeze(0).expand(
                logits_cpy.shape[0], -1
            )
            # scatter_add with -phi_row subtracts phi[j] from rank-j logit
            logits_cpy.scatter_add_(-1, top_f_idxs, -phi_row)
        
        # Note: keep inplace ops for speed
        logits_cpy.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits_cpy, dim=-1, dtype=torch.float)
        
        # Existing sampler_x rescaling path: completely unchanged
        if self.sampler_x is not None and is_tree:
            probs = apply_sampler_x_rescaling(probs, self.sampler_x, self.F)
        
        epsilon = 1e-10
        scores = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon)
        sample_tokens = scores.argmax(dim=-1)
        return torch.where(zero_mask, greedy_tokens, sample_tokens)


def profile_sampler():
    """Profile the sampler on [b, v] logits for b=128, v=150_000"""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nProfiling Sampler on {device}")
    
    # Test parameters
    b = 128
    v = 150_000
    
    # Create test data
    logits = torch.randn(b, v, device=device)
    temperatures = torch.rand(b, device=device) * 1.5  # temperatures in [0, 1.5]
    
    sampler = Sampler().to(device)
    
    print(f"Testing with batch_size={b}, vocab_size={v}")
    
    # Warm up
    print("Warming up sampler")
    for _ in range(10):
        _ = sampler(logits, temperatures)
    
    # Profile
    num_runs = 100
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        _ = sampler(logits, temperatures)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    sampler_time_ms = (end_time - start_time) * 1000 / num_runs
    
    print(f"Sampler time: {sampler_time_ms:.3f}ms")

# takes 0.5ms, negligible 
if __name__ == "__main__":
    profile_sampler()