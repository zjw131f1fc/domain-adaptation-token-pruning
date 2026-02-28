"""Subspace utilities for low-rank repair / alignment.

This module implements a minimal "gap-PCA" subspace estimator:
given token-wise gap vectors G = (teacher_h - student_h) of shape (N, D),
compute an orthonormal basis B ∈ R^{D×r} that spans the top-r principal
components of G.

The intended usage in this repo:
- Estimate B once at the beginning of Stage 2 (adapter fine-tune) when the pruner
  is frozen and deterministic.
- Constrain repair deltas to lie (mostly) in span(B) to reduce "orthogonal drift"
  in the full 4096-D space while still improving task metrics.
"""

from __future__ import annotations

from typing import Optional

import torch


def compute_gap_pca_basis(
    gap_tokens: torch.Tensor,
    rank: int,
    *,
    center: bool = True,
    niter: int = 2,
) -> torch.Tensor:
    """Compute an orthonormal PCA basis for gap tokens.

    Args:
        gap_tokens: (N, D) float tensor. Values should already be detached from graph.
        rank: target basis rank r.
        center: whether to mean-center gap_tokens before PCA.
        niter: iterations for randomized low-rank PCA (torch.pca_lowrank).

    Returns:
        basis: (D, r_eff) float32 tensor with orthonormal columns.
    """
    if gap_tokens is None:
        raise ValueError("gap_tokens is None")
    if gap_tokens.dim() != 2:
        raise ValueError(f"gap_tokens must be 2D (N,D), got {tuple(gap_tokens.shape)}")

    X = gap_tokens.float()
    n, d = int(X.shape[0]), int(X.shape[1])
    if n <= 1 or d <= 1:
        raise ValueError(f"gap_tokens too small for PCA: {(n, d)}")

    r = int(rank)
    if r <= 0:
        raise ValueError(f"rank must be > 0, got {rank}")

    # torch.pca_lowrank requires q <= min(n, d)
    q = min(r, n, d)
    if q <= 0:
        raise ValueError(f"effective rank is 0 after clamp: rank={rank}, shape={(n, d)}")

    if center:
        X = X - X.mean(dim=0, keepdim=True)

    # Randomized low-rank PCA; V has shape (D, q) with orthonormal columns.
    # Note: center=False because we already centered above.
    if hasattr(torch, "pca_lowrank"):
        _, _, V = torch.pca_lowrank(X, q=q, center=False, niter=int(niter))
        basis = V[:, :q].contiguous()
    else:
        # Fallback: full SVD (can be slower)
        # X = U S Vh, so right singular vectors are rows of Vh; take top-q.
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        basis = Vh[:q].transpose(0, 1).contiguous()
    # Explicitly normalize to reduce numerical drift when later cast to bf16.
    basis = torch.linalg.qr(basis, mode="reduced").Q.contiguous()
    return basis


def project_onto_basis(
    x: torch.Tensor,
    basis: Optional[torch.Tensor],
    *,
    orth_scale: float = 0.0,
) -> torch.Tensor:
    """Project x onto span(basis), optionally keeping a scaled orthogonal residual.

    Args:
        x: (..., D)
        basis: (D, r) with (approximately) orthonormal columns, or None.
        orth_scale: 0.0 means drop orthogonal component completely.
            1.0 means keep x unchanged. Values in (0,1) partially keep orth residual.

    Returns:
        x_proj: (..., D), same dtype/device as x.
    """
    if basis is None:
        return x
    if x.dim() < 1:
        raise ValueError(f"x must have at least 1 dim, got {tuple(x.shape)}")
    if basis.dim() != 2:
        raise ValueError(f"basis must be 2D (D,r), got {tuple(basis.shape)}")
    if int(basis.shape[0]) != int(x.shape[-1]):
        raise ValueError(f"basis D mismatch: basis={tuple(basis.shape)} vs x={tuple(x.shape)}")

    # Ensure numerically stable projection.
    x_f = x.float()
    B = basis.float()
    # (..., r)
    coeff = torch.matmul(x_f, B)
    # (..., D)
    x_par = torch.matmul(coeff, B.transpose(-2, -1))
    if orth_scale == 0.0:
        return x_par.to(dtype=x.dtype)
    x_orth = x_f - x_par
    out = x_par + float(orth_scale) * x_orth
    return out.to(dtype=x.dtype)
