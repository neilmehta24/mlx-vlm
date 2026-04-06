# Re-export rope utilities from mlx-lm with ProportionalRoPE support.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _rotate_half(x: mx.array, traditional: bool) -> mx.array:
    if traditional:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return mx.stack((-x_odd, x_even), axis=-1).reshape(x.shape)

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)


class DefaultRoPE(nn.Module):
    """HF-style default RoPE using explicit rotate_half math."""

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        scaling_config: Optional[dict] = None,
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional

        scaling_config = scaling_config or {}
        partial_rotary_factor = scaling_config.get("partial_rotary_factor", 1.0)
        self.rotated_dims = min(dims, 2 * int(partial_rotary_factor * dims // 2))

        if self.rotated_dims > 0:
            exponents = mx.arange(0, self.rotated_dims, 2, dtype=mx.float32) / dims
            self._inv_freq = 1.0 / (base**exponents)
        else:
            self._inv_freq = None

    def __call__(self, x, offset=0):
        if self.rotated_dims <= 0:
            return x

        head = x[..., : self.rotated_dims]
        tail = x[..., self.rotated_dims :]
        seq_len = head.shape[-2]
        seq_positions = mx.arange(seq_len, dtype=mx.float32)

        if isinstance(offset, mx.array):
            offset = offset.astype(mx.float32)
            if offset.ndim == 0:
                positions = seq_positions + offset
                freqs = positions[:, None] * self._inv_freq[None, :]
                emb = mx.concatenate((freqs, freqs), axis=-1)
                cos = mx.reshape(mx.cos(emb), (1, 1, seq_len, self.rotated_dims))
                sin = mx.reshape(mx.sin(emb), (1, 1, seq_len, self.rotated_dims))
            else:
                positions = offset[:, None] + seq_positions[None, :]
                freqs = positions[..., None] * self._inv_freq[None, None, :]
                emb = mx.concatenate((freqs, freqs), axis=-1)
                cos = mx.expand_dims(mx.cos(emb), axis=1)
                sin = mx.expand_dims(mx.sin(emb), axis=1)
        else:
            positions = seq_positions + float(offset)
            freqs = positions[:, None] * self._inv_freq[None, :]
            emb = mx.concatenate((freqs, freqs), axis=-1)
            cos = mx.reshape(mx.cos(emb), (1, 1, seq_len, self.rotated_dims))
            sin = mx.reshape(mx.sin(emb), (1, 1, seq_len, self.rotated_dims))

        cos = cos.astype(head.dtype)
        sin = sin.astype(head.dtype)
        rotated = _rotate_half(head, traditional=self.traditional)
        head = (head * cos) + (rotated * sin)

        if tail.shape[-1] == 0:
            return head
        return mx.concatenate((head, tail), axis=-1)


class ProportionalRoPE(nn.Module):
    """Proportional RoPE for Gemma 4 full-attention layers.

    Frequencies are computed relative to the full head dimension (not just the
    rotated portion), and rotation is applied to the first rotated_dims//2
    elements of each half of the head — matching HF's rotate_half convention.
    """

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        scaling_config: Optional[dict] = None,
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional

        scaling_config = scaling_config or {}
        factor = scaling_config.get("factor", 1.0)
        partial_rotary_factor = scaling_config.get("partial_rotary_factor", 1.0)

        rope_angles = int(partial_rotary_factor * dims // 2)
        self.rotated_dims = 2 * rope_angles

        if self.rotated_dims > 0:
            exponents = mx.arange(0, self.rotated_dims, 2, dtype=mx.float32) / dims
            self._freqs = factor * (base**exponents)
        else:
            self._freqs = None

    def __call__(self, x, offset=0):
        if self.rotated_dims <= 0:
            return x

        head = x[..., : self.dims]
        tail = x[..., self.dims :]
        half = self.dims // 2

        left = head[..., :half]
        right = head[..., half:]
        rotated = mx.concatenate(
            [left[..., : self.rotated_dims // 2], right[..., : self.rotated_dims // 2]],
            axis=-1,
        )
        rotated = mx.fast.rope(
            rotated,
            self.rotated_dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )

        left = mx.concatenate(
            [
                rotated[..., : self.rotated_dims // 2],
                left[..., self.rotated_dims // 2 :],
            ],
            axis=-1,
        )
        right = mx.concatenate(
            [
                rotated[..., self.rotated_dims // 2 :],
                right[..., self.rotated_dims // 2 :],
            ],
            axis=-1,
        )
        head = mx.concatenate([left, right], axis=-1)

        if tail.shape[-1] == 0:
            return head
        return mx.concatenate([head, tail], axis=-1)


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: Optional[dict] = None,
    max_position_embeddings: Optional[int] = None,
):
    """Initialize the appropriate RoPE variant based on scaling_config."""
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get(
            "rope_type", "default"
        )
    else:
        rope_type = "default"

    if rope_type == "proportional":
        return ProportionalRoPE(
            dims=dims,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )

    # Default: explicit HF-style rotate_half RoPE for tighter text parity.
    return DefaultRoPE(
        dims=dims,
        traditional=traditional,
        base=base,
        scaling_config=scaling_config,
    )
