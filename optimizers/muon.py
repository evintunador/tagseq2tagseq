"""
In-repo NorMuon optimizer combining four enhancements:

  - Cautious weight decay: WD fires only where the update and param are
    sign-aligned (mask = (update * p) >= 0), matching the modded-nanogpt
    reference for the Muon path.
  - NorMuon variance reduction: Adafactor-style low-rank second moment
    gives Muon per-row/column adaptive step sizes without a full O(n²)
    matrix.  Controlled by the muon_beta2 param-group key.
  - Polar Express orthogonalization: replaces Newton-Schulz with a faster-
    converging 5-iteration algorithm backed by three custom Triton kernels
    (XXT, XTX, ba_plus_cAA).  Fused with Nesterov momentum in one
    torch.compile-d call.
  - BF16 mantissa tracking: a uint16 buffer per Muon param stores the low
    16 precision bits so that late-training updates smaller than a BF16
    ULP are not silently dropped.

Compiled inner loops use @torch.compile(dynamic=False, fullgraph=True).
Scalar hyperparameters (lr, wd, momentum) that change during training are
passed as 0-D CPU tensors to avoid recompilation.

Deviation from modded-nanogpt reference:
  The reference uses lr² × wd for weight-decay magnitude, calibrated to their
  large LR range (0.04–0.6).  Our AdamW LR is ~1e-5–3e-4, where lr² × wd ≈ 0.
  We use lr × wd throughout, matching our existing optimizer behavior.
"""

import torch
import torch.distributed as dist

from kernels.polar_express import polar_express


# ---------------------------------------------------------------------------
# Compiled Muon inner loop: variance reduction + cautious WD + mantissa update
# ---------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def _apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
    """
    Adafactor-style low-rank second moment scaling (NorMuon variance reduction).

    Scales each row (or column) of v_chunk by the inverse sqrt of an EMA of
    its squared norm, then renormalizes globally to preserve Frobenius norm.

    beta2 and red_dim are Python scalars — specialized at compile time.
    """
    v_mean      = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm      = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size).sqrt_()
    second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta2)
    step_size   = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq   = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new  = scaled_sq.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))


@torch.compile(dynamic=False, fullgraph=True)
def _muon_cautious_update_inplace(p_u16, mantissa, grad, lr_t, wd_t):
    """
    Cautious WD + BF16 mantissa tracking for a Muon param.

    Reconstructs FP32 precision from the BF16 high bits (p_u16) and stored
    low bits (mantissa), applies sign-aligned weight decay and the gradient
    step, then writes both halves back.

    lr_t and wd_t are 0-D CPU tensors to avoid recompilation when lr changes.
    Only called for BF16 params; float32 params use _muon_cautious_update_fp.
    """
    lr = lr_t.to(torch.float32)
    wd = wd_t.to(torch.float32)
    grad = grad.float()
    p_raw = (p_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    p32   = p_raw.view(torch.float32)
    mask  = (grad * p32) >= 0
    p32.copy_(p32 - p32 * mask * (lr * wd) - grad * lr)
    p_u16.copy_((p_raw >> 16).to(torch.uint16))
    mantissa.copy_(p_raw.to(torch.uint16))


def _muon_cautious_update_fp(p, grad, lr, wd):
    """Cautious WD for non-BF16 Muon params (no mantissa tracking)."""
    grad = grad.to(p.dtype)
    mask = (grad * p) >= 0
    p.mul_(1.0 - lr * wd * mask.to(p.dtype))
    p.add_(grad, alpha=-lr)


# ---------------------------------------------------------------------------
# Compiled Adam inner loop: cautious WD
# ---------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def _adam_update_step(p, g, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
    """
    Adam update with cautious WD.

    step_size_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)  (bias-corrected lr)
    eff_wd_t    = lr * wd

    Both are 0-D CPU tensors to avoid recompilation.
    beta1, beta2, eps are Python scalars — specialized at compile time.
    """
    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)
    mask   = (update * p) > 0
    update.addcmul_(p, mask, value=eff_wd_t)
    p.add_(update, alpha=-1.0)


# ---------------------------------------------------------------------------
# Distributed optimizer
# ---------------------------------------------------------------------------

class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed NorMuon + AdamW optimizer.

    Muon groups: use_muon=True.  Required keys: lr, momentum, weight_decay,
    beta2 (NorMuon second-moment EMA coefficient).
    Adam groups: use_muon=False.  Required keys: lr, betas, eps, weight_decay.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"]       = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"]           = group.get("lr", 0.02)
                group["momentum"]     = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["beta2"]        = group.get("beta2", 0.95)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "beta2", "use_muon"}
            else:
                group["lr"]           = group.get("lr", 3e-4)
                group["betas"]        = group.get("betas", (0.9, 0.95))
                group["eps"]          = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        # 0-D CPU scalar buffers — filled before each compiled call to avoid
        # recompilation when lr/momentum change during training.
        self._momentum_t  = torch.zeros((), device="cpu")
        self._lr_t        = torch.zeros((), device="cpu")
        self._wd_t        = torch.zeros((), device="cpu")
        self._step_size_t = torch.zeros((), device="cpu")
        self._eff_wd_t    = torch.zeros((), device="cpu")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._momentum_t.fill_(group["momentum"])
                self._lr_t.fill_(group["lr"])
                self._wd_t.fill_(group["weight_decay"])
                beta2 = group["beta2"]

                params     = group["params"]
                world_size = dist.get_world_size()
                params_pad = params + [torch.empty_like(params[-1])] * (
                    world_size - len(params) % world_size
                )
                for base_i in range(len(params))[::world_size]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                            M, N = p.shape[-2], p.shape[-1]
                            batch = p.shape[:-2]
                            red_dim = -1 if M >= N else -2
                            buf_shape = (*batch, M, 1) if red_dim == -1 else (*batch, 1, N)
                            state["second_momentum_buffer"] = torch.zeros(
                                buf_shape, dtype=torch.float32, device=p.device
                            )
                            state["red_dim"] = red_dim
                            if p.dtype == torch.bfloat16:
                                state["mantissa"] = torch.zeros(
                                    p.shape, dtype=torch.uint16, device=p.device
                                )

                        split = p.shape[-2] > 1024
                        v = polar_express(
                            p.grad, state["momentum_buffer"],
                            self._momentum_t, split_baddbmm=split,
                        )
                        v = _apply_normuon_variance_reduction(
                            v.reshape(p.shape),
                            state["second_momentum_buffer"],
                            beta2,
                            state["red_dim"],
                        )
                        # Scale to match spectral-norm LR units
                        M, N = p.shape[-2], p.shape[-1]
                        v = v * max(1, M / N) ** 0.5

                        if p.dtype == torch.bfloat16:
                            _muon_cautious_update_inplace(
                                p.view(torch.uint16), state["mantissa"],
                                v, self._lr_t, self._wd_t,
                            )
                        else:
                            _muon_cautious_update_fp(p, v, group["lr"], group["weight_decay"])

                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"]       = 0
                    state["step"] += 1
                    t = state["step"]
                    self._step_size_t.fill_(group["lr"] * ((1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)))
                    self._eff_wd_t.fill_(group["lr"] * group["weight_decay"])
                    _adam_update_step(
                        p, p.grad,
                        state["exp_avg"], state["exp_avg_sq"],
                        beta1, beta2, group["eps"],
                        self._step_size_t, self._eff_wd_t,
                    )

        return loss


# ---------------------------------------------------------------------------
# Single-device optimizer (non-distributed)
# ---------------------------------------------------------------------------

class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Non-distributed variant of MuonWithAuxAdam.  Same update logic, no all_gather."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"]           = group.get("lr", 0.02)
                group["momentum"]     = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["beta2"]        = group.get("beta2", 0.95)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "beta2", "use_muon"}
            else:
                group["lr"]           = group.get("lr", 3e-4)
                group["betas"]        = group.get("betas", (0.9, 0.95))
                group["eps"]          = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        self._momentum_t  = torch.zeros((), device="cpu")
        self._lr_t        = torch.zeros((), device="cpu")
        self._wd_t        = torch.zeros((), device="cpu")
        self._step_size_t = torch.zeros((), device="cpu")
        self._eff_wd_t    = torch.zeros((), device="cpu")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._momentum_t.fill_(group["momentum"])
                self._lr_t.fill_(group["lr"])
                self._wd_t.fill_(group["weight_decay"])
                beta2 = group["beta2"]

                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                        M, N = p.shape[-2], p.shape[-1]
                        batch = p.shape[:-2]
                        red_dim = -1 if M >= N else -2
                        buf_shape = (*batch, M, 1) if red_dim == -1 else (*batch, 1, N)
                        state["second_momentum_buffer"] = torch.zeros(
                            buf_shape, dtype=torch.float32, device=p.device
                        )
                        state["red_dim"] = red_dim
                        if p.dtype == torch.bfloat16:
                            state["mantissa"] = torch.zeros(
                                p.shape, dtype=torch.uint16, device=p.device
                            )

                    split = p.shape[-2] > 1024
                    v = polar_express(
                        p.grad, state["momentum_buffer"],
                        self._momentum_t, split_baddbmm=split,
                    )
                    v = _apply_normuon_variance_reduction(
                        v.reshape(p.shape),
                        state["second_momentum_buffer"],
                        beta2,
                        state["red_dim"],
                    )
                    M, N = p.shape[-2], p.shape[-1]
                    v = v * max(1, M / N) ** 0.5

                    if p.dtype == torch.bfloat16:
                        _muon_cautious_update_inplace(
                            p.view(torch.uint16), state["mantissa"],
                            v, self._lr_t, self._wd_t,
                        )
                    else:
                        _muon_cautious_update_fp(p, v, group["lr"], group["weight_decay"])
            else:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"]       = 0
                    state["step"] += 1
                    t = state["step"]
                    self._step_size_t.fill_(group["lr"] * ((1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)))
                    self._eff_wd_t.fill_(group["lr"] * group["weight_decay"])
                    _adam_update_step(
                        p, p.grad,
                        state["exp_avg"], state["exp_avg_sq"],
                        beta1, beta2, group["eps"],
                        self._step_size_t, self._eff_wd_t,
                    )

        return loss
