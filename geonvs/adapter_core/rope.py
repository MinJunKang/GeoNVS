"""Thin wrapper around Transformer Engine's fused rotary position embedding.

Only `apply_rotary_pos_emb` is needed (by the RoPE-based fusion layers in
`gsadapter.py`). Kept in its own module so that `transformer_engine` is only
required when the `rattn` / `gattn` fusion methods are used.
"""

from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

__all__ = ["apply_rotary_pos_emb"]
