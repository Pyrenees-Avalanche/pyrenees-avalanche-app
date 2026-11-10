import numpy as np
import torch
import torch.nn.functional as F


def morph_close(mask, kernel_size: int = 3, iterations: int = 1, device: str | torch.device | None = None):
    """
    Morphological closing (dilate → erode) for *binary* masks.

    Parameters
    ----------
    mask : torch.Tensor | np.ndarray
        Supported shapes: (H, W), (1, H, W), (B, H, W), (B, 1, H, W).
        Values should be binary (0/1 or False/True). If you have probabilities,
        threshold before calling (e.g., (x > thr).astype(np.uint8)).
    kernel_size : int, optional
        Odd side length of the structuring element. Default = 3.
    iterations : int, optional
        How many times dilation and erosion are applied. Default = 1.
    device : str | torch.device | None, optional
        Device to run on. If None:
          - for torch inputs, use mask.device
          - for numpy inputs, use "cpu"

    Returns
    -------
    torch.Tensor | np.ndarray
        The morphologically closed mask, same type/shape/container as input.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    # --- Normalize to a torch tensor on the right device ---
    if isinstance(mask, np.ndarray):
        is_numpy = True
        np_dtype = mask.dtype
        mask_t = torch.from_numpy(mask)
        torch_target_dtype = mask_t.dtype
        dev = torch.device("cpu") if device is None else torch.device(device)
        if dev.type != "cpu":
            # from_numpy gives a CPU tensor; move if GPU requested
            mask_t = mask_t.to(dev)
    elif isinstance(mask, torch.Tensor):
        is_numpy = False
        np_dtype = None
        mask_t = mask
        torch_target_dtype = mask_t.dtype
        dev = mask_t.device if device is None else torch.device(device)
        if mask_t.device != dev:
            mask_t = mask_t.to(dev)
    else:
        raise TypeError("mask must be a torch.Tensor or numpy.ndarray")

    # --- Normalize shape to (B, C, H, W) ---
    added_b = added_c = False
    if mask_t.dim() == 2:             # (H, W)
        mask_t = mask_t.unsqueeze(0).unsqueeze(0); added_b = added_c = True
    elif mask_t.dim() == 3:           # (1,H,W) or (B,H,W)
        if mask_t.shape[0] == 1:      # (1,H,W)
            mask_t = mask_t.unsqueeze(0); added_b = True
        else:                         # (B,H,W)
            mask_t = mask_t.unsqueeze(1); added_c = True
    elif mask_t.dim() != 4:
        raise ValueError("mask must be 2D, 3D or 4D")

    # --- Closing: dilate then erode ---
    mask_f = mask_t.float()
    # Clamp to [0,1] to be safe (assumes binary inputs)
    mask_f = mask_f.clamp_(0.0, 1.0)

    pad = kernel_size // 2
    for _ in range(iterations):
        mask_f = F.max_pool2d(mask_f, kernel_size, stride=1, padding=pad)  # dilation
    for _ in range(iterations):
        inv = 1.0 - mask_f
        inv = F.max_pool2d(inv, kernel_size, stride=1, padding=pad)        # erosion
        mask_f = 1.0 - inv

    # --- Restore original shape ---
    if added_c:
        mask_f = mask_f.squeeze(1)
    if added_b:
        mask_f = mask_f.squeeze(0)

    # --- Cast back to original dtype & container ---
    mask_f = mask_f.to(torch_target_dtype)

    if is_numpy:
        out = mask_f.detach().cpu().numpy()
        # preserve original NumPy dtype
        if out.dtype != np_dtype:
            out = out.astype(np_dtype, copy=False)
        return out
    else:
        return mask_f
