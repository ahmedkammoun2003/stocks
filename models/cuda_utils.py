"""CUDA lifecycle helpers for multiprocessing workers."""
import gc

import torch


def release_cuda_memory() -> None:
    """Drop cached GPU allocations so worker processes can exit cleanly."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
    except Exception:
        pass


def state_dict_to_cpu(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def model_to_cpu(model: torch.nn.Module) -> torch.nn.Module:
    return model.cpu()
