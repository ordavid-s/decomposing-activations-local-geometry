import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal
import torch

@torch.no_grad()
def project_loader_to_subspace(
    model,
    loader,
    k: int,
    token_to_str,
    *,
    max_points: Optional[int] = 100_000,
    center: bool = False,                              # center around m_perp if True
    device: Optional[str] = None,
    assign: Literal["hard", "threshold"] = "hard",     # how to select points for concept k
    min_alpha: float = 0.2,                            # used when assign="threshold"
    tau: float = 1.0,                                  # temperature for responsibilities
):
    """
    Projects only points that belong to component k (by hard argmax or alpha-threshold)
    onto span(U_k) to get subspace coordinates. Uses m_perp for centering if center=True.

        Expects an MFA-like model exposing mu, loadings, and responsibilities(x, tau).

    Notes:
      - U_k are unit 'dir' columns (not guaranteed orthonormal).
      - Coordinates c solve least-squares: c = (U^T U)^{-1} U^T x.
      - Energy is ||U c||^2 (true captured energy for non-orthonormal bases).

    Returns:
        {
          "coords":   (N_kept, q) float32,   # least-squares coordinates in U_k
          "tokens":   list[str] length N_kept,
          "energy":   (N_kept,) float32,     # ||proj_U(x)||^2
          "ratio":    (N_kept,) float32,     # ||proj_U(x)||^2 / ||x||^2
          "axis_names": ["u0", ..., f"u{q-1}"],
          "k":        int,
          "centroid_coords": (q,) float32    # coords of mu_k in U_k if not centered; zeros if centered
        }
    """

    # Resolve loadings from whichever MFA interface is available.
    def get_dirs_and_mu(m):
        # Preferred: use public W (already rotated if rotation is enabled)
        if hasattr(m, "W"):
            W = m.W  # (K,D,q) — rotated view if apply_oblimin_rotation() was called
            # normalize columns to unit length over D to get 'directions'
            U = W / (W.norm(dim=1, keepdim=True).clamp_min(1e-8))
            return U, m.mu

        if hasattr(m, "_dir_hat"):
            return m._dir_hat(), m.mu

        raise AttributeError("Model must provide loadings via W or _dir_hat.")


    def chol_with_jitter(G, max_tries=6, eps0=1e-8, growth=10.0):
        # Try Cholesky; if it fails, add εI and retry.
        eps = eps0
        I = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
        for _ in range(max_tries):
            try:
                return torch.linalg.cholesky(G)
            except RuntimeError:
                G = G + eps * I
                eps *= growth
        # Final fallback: use LDLT-ish via svd->recompose to ensure PSD, then cholesky
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        S = S.clamp_min(1e-10)
        G_psd = (U * S) @ Vh
        return torch.linalg.cholesky(G_psd)

    # --- get subspace + mean from model ---
    U_all, mu_all = get_dirs_and_mu(model)     # (K,D,q), (K,D)
    U_k = U_all[k]                              # (D,q)
    mu_k = mu_all[k]                            # (D,)

    q = U_k.shape[-1]
    if device is None:
        device = U_k.device.type
    U_k = U_k.to(device)
    mu_k = mu_k.to(device)

    # Precompute Gram matrix G = U^T U and its Cholesky for solves
    G = (U_k.transpose(0, 1) @ U_k).to(device)           # (q,q)
    Lg = chol_with_jitter(G)                              # lower-triangular (q,q)

    def ls_coords(X: torch.Tensor) -> torch.Tensor:
        """
        Least-squares coords c = (U^T U)^{-1} U^T X for a batch X (B,D).
        Uses Cholesky solve for stability. Returns c (B,q).
        """
        XtU = X @ U_k                                     # (B,q)
        # Solve G c^T = (XtU)^T  => c = (XtU) @ G^{-1}
        cT = torch.cholesky_solve(XtU.transpose(0, 1), Lg, upper=False)  # (q,B)
        return cT.transpose(0, 1)                         # (B,q)

    def proj_onto_U(X: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection of X onto span(U): X_proj = U c."""
        C = ls_coords(X)                                  # (B,q)
        return (U_k @ C.transpose(0, 1)).transpose(0, 1)  # (B,D)

    # m_perp = mu_k - Proj_U(mu_k)
    mu_proj = proj_onto_U(mu_k.unsqueeze(0)).squeeze(0)   # (D,)
    m_perp = mu_k - mu_proj                               # (D,)

    coords_list, energy_list, ratio_list = [], [], []
    token_strs = []
    kept = 0

    for acts, toks in loader:
        # acts: (B,D), toks: length-B container
        acts = acts.to(device)

        # Responsibilities α = softmax((ll + log_pi)/tau)
        alpha = model.responsibilities(acts, tau=tau)  # (B,K)

        if assign == "hard":
            mask = (alpha.argmax(dim=1) == k)          # (B,)
        elif assign == "threshold":
            mask = (alpha[:, k] >= float(min_alpha))
        else:
            raise ValueError("assign must be 'hard' or 'threshold'")

        if not mask.any():
            continue

        # Keep only selected points
        acts_k = acts[mask]                             # (Bk,D)
        toks_k = [token_to_str(t) if callable(token_to_str) else str(t)
                  for t, m in zip(toks, mask.tolist()) if m]

        # center around m_perp if requested
        X = acts_k - m_perp.unsqueeze(0) if center else acts_k  # (Bk,D)

        # least-squares coordinates in U_k
        C = ls_coords(X)                                 # (Bk,q)

        # projection in D-space for energy/ratio
        X_proj = (U_k @ C.transpose(0, 1)).transpose(0, 1)  # (Bk,D)

        captured = (X_proj ** 2).sum(-1)                 # (Bk,)
        denom = (X ** 2).sum(-1).clamp_min(1e-6)         # (Bk,)
        ratio = captured / denom

        coords_list.append(C.detach().cpu().to(dtype=torch.float32))
        energy_list.append(captured.detach().cpu().to(dtype=torch.float32))
        ratio_list.append(ratio.detach().cpu().to(dtype=torch.float32))
        token_strs.extend(toks_k)

        kept += acts_k.size(0)
        if max_points is not None and kept >= max_points:
            break

    if len(coords_list) == 0:
        return {
            "coords": torch.empty(0, q, dtype=torch.float32).numpy(),
            "tokens": [],
            "energy": torch.empty(0, dtype=torch.float32).numpy(),
            "ratio": torch.empty(0, dtype=torch.float32).numpy(),
            "axis_names": [f"u{i}" for i in range(q)],
            "k": int(k),
            "centroid_coords": (torch.zeros(q, dtype=torch.float32)).numpy(),
        }

    coords = torch.cat(coords_list, 0)   # already CPU float32
    energy = torch.cat(energy_list, 0)
    ratio  = torch.cat(ratio_list, 0)

    if center:
        centroid_coords = torch.zeros(q, dtype=torch.float32)
    else:
        centroid_coords = ls_coords(mu_k.unsqueeze(0)).squeeze(0).detach().cpu().to(dtype=torch.float32)

    return {
        "coords": coords,                 # torch.Tensor (CPU)
        "tokens": token_strs,
        "energy": energy,                 # torch.Tensor (CPU)
        "ratio": ratio,                   # torch.Tensor (CPU)
        "axis_names": [f"u{i}" for i in range(q)],
        "k": int(k),
        "centroid_coords": centroid_coords,  # torch.Tensor (CPU)
    }


def plot_subspace_scatter(data, dims=(0, 1), max_labels=200, figsize=(9, 6)):
    coords = data["coords"]
    if isinstance(coords, torch.Tensor):
        coords = coords.tolist()  # safe: no numpy needed

    i, j = dims
    x = [c[i] for c in coords]
    y = [c[j] for c in coords]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=8, alpha=0.6)

    tokens = data.get("tokens", [])
    for idx in range(min(max_labels, len(tokens))):
        ax.text(x[idx], y[idx], tokens[idx], fontsize=8, alpha=0.9)

    axis_names = data.get("axis_names", [f"u{k}" for k in range(max(i, j) + 1)])
    ax.set_xlabel(axis_names[i])
    ax.set_ylabel(axis_names[j])
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_title(f"Subspace scatter (k={data.get('k','?')})")
    plt.tight_layout()
    return fig, ax
