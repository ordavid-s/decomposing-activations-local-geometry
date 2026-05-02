import os
import tempfile

import torch
from accelerate import Accelerator
from modeling.mfa import MFA

def _atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)  # atomic on POSIX
    except Exception:
        try: os.remove(tmp)
        except Exception:
            pass
        raise

def _cpu_clone_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Keep checkpoint tensors independent from model/device state.
    return {k: v.detach().cpu().clone() for k, v in sd.items()}

def save_mfa(model: MFA, path: str, *, accelerator: Accelerator):
    """Save an MFA checkpoint from a potentially wrapped Accelerate model."""
    sd_full = accelerator.get_state_dict(model)
    sd_cpu = _cpu_clone_state_dict(sd_full)

    mu = sd_cpu.get("mu", None)
    if mu is None or mu.ndim != 2:
        raise RuntimeError("State dict missing a 2D 'mu' tensor (K, D).")
    K, D = map(int, mu.shape)

    unwrapped = accelerator.unwrap_model(model)
    q = getattr(unwrapped, "q", None)
    if q is None:
        dir_raw = sd_cpu.get("dir_raw", None)
        if dir_raw is None or dir_raw.ndim != 3:
            raise RuntimeError("Could not infer latent rank 'q': set model.q or include 'dir_raw' (K,D,q).")
        q = int(dir_raw.shape[-1])

    psi_rho = sd_cpu.get("psi_rho", None)
    if psi_rho is None:
        raise RuntimeError("State dict missing 'psi_rho'.")
    psi_per_component = bool(psi_rho.ndim == 2 and psi_rho.shape[0] == K)

    eps_floor = float(getattr(unwrapped, "_eps", 1e-5))

    meta = {
        "cls": "MFA",
        "K": K,
        "D": D,
        "rank": int(q),
        "psi_per_component": psi_per_component,
        "eps_floor": eps_floor,
        "version": 1,
    }
    blob = {"meta": meta, "state_dict": sd_cpu}
    _atomic_torch_save(blob, path)

def load_mfa(path: str, map_location: str | torch.device = "cpu") -> MFA:
    """Load an MFA checkpoint saved with save_mfa and rebuild the module."""
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "state_dict" not in blob or "meta" not in blob:
        raise RuntimeError("Malformed MFA checkpoint: expected dict with 'state_dict' and 'meta'.")

    meta = blob["meta"]
    st = blob["state_dict"]

    mu = st.get("mu", None)
    if mu is None or mu.ndim != 2:
        raise RuntimeError("Checkpoint missing a 2D 'mu' tensor (K, D).")
    K, D = map(int, mu.shape)

    q = int(meta["rank"])
    psi_per_component = bool(meta.get("psi_per_component",
                                      st.get("psi_rho", torch.empty(())).ndim == 2))
    eps_floor = float(meta.get("eps_floor", 1e-5))

    m = MFA(
        centroids=mu,
        rank=q,
        psi_per_component=psi_per_component,
        eps_floor=eps_floor,
    )
    m.load_state_dict(st, strict=True)
    return m.to(map_location)