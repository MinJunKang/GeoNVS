import torch
import torch.nn as nn




def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100, ratio: float = 0.1) -> torch.Tensor:
    """
    Convert 3D point grid + weight grid (HxWx4or8) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (B, H, W, 4 or 8) containing GS params
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (B, H, W, embed_dim) with positional embeddings
    """
    B, H, W, grid_dim = pos_grid.shape
    assert grid_dim == 4 or grid_dim == 8, "Input pos_grid must have last dimension of size 4 or 8"
    pos_flat = pos_grid.reshape(B, H*W, grid_dim)  # Flatten to (B, H*W, 4 or 8)

    # Process x and y coordinates separately
    if grid_dim == 4:
        emb_x = make_sincos_pos_embed(embed_dim // 4, pos_flat[..., 0], omega_0=omega_0)  # [1, H*W, D/4]
        emb_y = make_sincos_pos_embed(embed_dim // 4, pos_flat[..., 1], omega_0=omega_0)  # [1, H*W, D/4]
        emb_z = make_sincos_pos_embed(embed_dim // 4, pos_flat[..., 2], omega_0=omega_0)  # [1, H*W, D/4]
        emb_w = make_sincos_pos_embed(embed_dim // 4, pos_flat[..., 3], omega_0=omega_0)  # [1, H*W, D/4]
        emb = torch.cat([emb_x, emb_y, emb_z, emb_w], dim=-1)  # [B, H*W, D]
    else:
        emb_x = make_sincos_pos_embed(embed_dim // 8, pos_flat[..., 0], omega_0=omega_0)  # [1, H*W, D/8]
        emb_y = make_sincos_pos_embed(embed_dim // 8, pos_flat[..., 1], omega_0=omega_0)  # [1, H*W, D/8]
        emb_z = make_sincos_pos_embed(embed_dim // 8, pos_flat[..., 2], omega_0=omega_0)  # [1, H*W, D/8]
        emb_rx = make_sincos_pos_embed(embed_dim // 8, pos_flat[..., 3], omega_0=omega_0)  # [1, H*W, D/8]
        emb_ry = make_sincos_pos_embed(embed_dim // 8, pos_flat[..., 4], omega_0=omega_0)  # [1, H*W, D/8]
        emb_rz = make_sincos_pos_embed(embed_dim // 8, pos_flat[..., 5], omega_0=omega_0)  # [1, H*W, D/8]
        emb_rw = make_sincos_pos_embed(embed_dim // 8, pos_flat[..., 6], omega_0=omega_0)  # [1, H*W, D/8]
        emb_w = make_sincos_pos_embed(embed_dim // 8, pos_flat[..., 7], omega_0=omega_0)  # [1, H*W, D/8]
        emb = torch.cat([emb_x, emb_y, emb_z, emb_rx, emb_ry, emb_rz, emb_rw, emb_w], dim=-1)  # [B, H*W, D]

    return ratio * emb.view(B, H, W, embed_dim)  # [B, H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)
    out = torch.einsum("bm,d->bmd", pos, omega)  # (B, M, D/2), outer product

    emb_sin = torch.sin(out)  # (B, M, D/2)
    emb_cos = torch.cos(out)  # (B, M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (B, M, D)
    return emb.float()


class FreqEmbedder(nn.Module):
    """FreqEmbedder module. Embed inputs into higher dimensions.
    For example, x = sin(2**N * x) or sin(N * x) for N in range(0, 10)
    ref: https://github.com/ventusff/neurecon/blob/main/models/base.py
    """

    def __init__(
        self,
        input_dim,
        n_freqs,
        log_sampling=True,
        include_input=True,
        periodic_fns=(torch.sin, torch.cos),
        *args,
        **kwargs
    ):
        """
        Args:
            input_dim: dimension of input to be embedded. For example, xyz is dim=3
            n_freqs: number of frequency bands. If 0, will not encode the inputs.
            log_sampling: if True, use log factor sin(2**N * x). Else use scale factor sin(N * x).
                      By default is True
            include_input: if True, raw input is included in the embedding. Appear at beginning. By default is True
            periodic_fns: a list of periodic functions used to embed input. By default is (sin, cos)

        Returns:
            Embedded inputs with shape:
                (inputs_dim * len(periodic_fns) * N_freq + include_input * inputs_dim)
            For example, inputs_dim = 3, using (sin, cos) encoding, N_freq = 10, include_input, will results at
                3 * 2 * 10 + 3 = 63 output shape.
        """
        super(FreqEmbedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        # get output dim
        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim
        self.out_dim += self.input_dim * n_freqs * len(self.periodic_fns)

        if n_freqs == 0 and include_input:  # inputs only
            fb = torch.empty(0)
        else:
            if log_sampling:
                fb = 2.**torch.linspace(0., n_freqs - 1, n_freqs)
            else:
                fb = torch.linspace(2.**0., 2.**(n_freqs - 1), n_freqs)
        self.register_buffer("freq_bands", fb, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor of shape [B, input_dim]

        Returns:
            embed_x: tensor of shape [B, out_dim]
        """
        assert (x.shape[-1] == self.input_dim), 'Input shape should be (B, {})'.format(self.input_dim)

        embed_x = []
        if self.include_input:
            embed_x.append(x)

        for freq in self.freq_bands:
            for fn in self.periodic_fns:
                embed_x.append(fn(x * freq))

        if len(embed_x) > 1:
            embed_x = torch.cat(embed_x, dim=-1)
        else:
            embed_x = embed_x[0]

        return embed_x