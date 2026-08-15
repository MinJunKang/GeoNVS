from torch.utils.data._utils.collate import default_collate
import torch
import torch.nn.functional as F

def collate_with_gaussian_padding(batch):
    # here, we don't need mask since zero padded gaussian will have opacity of zeros
    # don't affect to rasterization process
    
    if 'gaussians' not in batch[0]:
        return default_collate(batch)
    
    # Pad gaussians to the same length
    gaussians = [item['gaussians'] for item in batch]  # List of (N_i, C)
    max_len = max(g.shape[0] for g in gaussians)
    padded_gaussians = [F.pad(g, (0, 0, 0, max_len - g.shape[0])) for g in gaussians]

    # Rebuild a clean batch dict
    batch_out = {key: default_collate([item[key] for item in batch]) for key in batch[0] if key != 'gaussians'}
    batch_out['gaussians'] = default_collate(padded_gaussians)  # shape: (B, max_len, C)

    return batch_out