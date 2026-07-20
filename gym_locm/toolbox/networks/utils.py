import torch as th
import warnings

def safely_compile(model):
    """
    Safely compiles a PyTorch model if the environment supports it.
    Triton (the default backend for torch.compile) requires CUDA compute capability >= 7.0.
    """
    if th.cuda.is_available():
        major, _ = th.cuda.get_device_capability()
        if major < 7:
            warnings.warn(f"Skipping torch.compile: GPU compute capability is {major}.x (< 7.0)")
            return model
            
    try:
        return th.compile(model)
    except Exception as e:
        warnings.warn(f"Skipping torch.compile due to error: {e}")
        return model
