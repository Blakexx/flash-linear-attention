# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""RoLA in FLA's mixer contract: a thin wrapper over the standalone ``rola`` package's layer.

``rola`` owns the model -- the route producer, the recurrence, the op and its envelope.
This file adds exactly what FLA's mixer contract needs and nothing that computes RoLA:

1. a serializable construction surface: :class:`LevelRoutingConfig` names one routing level
   in plain fields so an HF config can carry it, and ``decay`` is a plain dict naming one of
   ``rola``'s decay sources;
2. FLA's call convention: ``forward(hidden_states, attention_mask, past_key_values,
   use_cache, ...)`` returning ``(o, None, past_key_values)``, the layer's continuation
   carried in the cache's ``recurrent_state``;
3. FLA's state accounting: ``get_stats`` / ``state_size``, read off the wrapped layer.

What ``rola`` refuses, this refuses, by passing the call through: a configuration outside
the built envelope, training before the native backward, prefill while the kernel is
rebuilt. There is no second implementation to fall back on.

``rola`` is an optional dependency: ``fla`` imports without it, and constructing :class:`RoLA` without it raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from fla.layers.utils import get_layer_cache, update_layer_cache

try:
    import rola
except ImportError:
    rola = None

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


@dataclass(frozen=True)
class LevelRoutingConfig:
    """One routing level in plain fields: its width, each duty's activation, and whether the
    duties share a projection.

    ``read``/``write`` are ``'dense'`` (softmax) or ``'sparse'`` (entmax at ``alpha``). An
    untied sparse/sparse level is ``rola``'s union routing (one support, the mass split
    between the duties); every other combination is its independent routing.
    """

    width: int
    read: str = "dense"
    write: str = "dense"
    tied: bool = False
    alpha: float = 1.5

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 1:
            raise ValueError(f"LevelRoutingConfig.width must be a positive int, got {self.width!r}")
        if type(self.tied) is not bool:
            raise TypeError(f"LevelRoutingConfig.tied must be bool, got {self.tied!r}")
        for side, value in (("read", self.read), ("write", self.write)):
            if value not in ("sparse", "dense"):
                raise ValueError(f"LevelRoutingConfig.{side} must be 'sparse' or 'dense', got {value!r}")

    @classmethod
    def of(cls, spec: LevelRoutingConfig | dict) -> LevelRoutingConfig:
        """A config, or one built from its dict."""
        return spec if isinstance(spec, cls) else cls(**spec)

    def level(self):
        """This level as ``rola``'s own level config."""
        if not self.tied and self.read == "sparse" and self.write == "sparse":
            return rola.UnionRouting(width=self.width, alpha=self.alpha)

        def activation(side: str):
            return rola.entmax(self.alpha) if side == "sparse" else rola.softmax()

        return rola.IndependentRouting(width=self.width, read=activation(self.read),
                                       write=activation(self.write), tied=self.tied)


#: the decay sources a config can name, each ``rola``'s own class by name
DECAY_SOURCES = {"constant": "ConstantDecay", "learned": "LearnedDecay"}


def _require_rola() -> None:
    if rola is None:
        raise ImportError("fla.layers.RoLA needs the `rola` package (https://github.com/Blakexx/rola)")


def decay_source(decay: dict | None, widths: tuple[int, ...], num_heads: int):
    """``None``, or ``{"source": "constant" | "learned", **its arguments}`` as that decay source."""
    if decay is None:
        return None
    arguments = dict(decay)
    source = arguments.pop("source", None)
    if source not in DECAY_SOURCES:
        raise ValueError(f"decay['source'] must be one of {sorted(DECAY_SOURCES)}, got {source!r}")
    return getattr(rola, DECAY_SOURCES[source])(**arguments, widths=widths, num_heads=num_heads)


class RoLA(nn.Module):
    """Routed linear attention as an FLA mixer.

    Args:
        hidden_size: model width.
        num_heads: heads.
        head_v_dim: per-head value width (``rola.RoLA``'s ``d_v``).
        levels: the routing levels, outermost first, as :class:`LevelRoutingConfig` or its dict.
        decay: ``None``, or a dict naming a decay source (:data:`DECAY_SOURCES`) and its arguments.
        router_bias: bias on the route projection.
        gain_bias_init: the side gain's bias initialization (``rola.RouteProducer``).
        layer_idx: this layer's index in the cache.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        head_v_dim: int = 64,
        levels: tuple[LevelRoutingConfig | dict, ...] = (LevelRoutingConfig(width=16),),
        decay: dict | None = None,
        router_bias: bool = False,
        gain_bias_init: float | None = None,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        _require_rola()
        self.levels = tuple(LevelRoutingConfig.of(level) for level in levels)
        routes = rola.RouteProducer([level.level() for level in self.levels], hidden_size=hidden_size,
                                    num_heads=num_heads, bias=router_bias, gain_bias_init=gain_bias_init)
        self.layer = rola.RoLA(routes, d_v=head_v_dim, layer_idx=layer_idx,
                               decay=decay_source(decay, routes.topology.widths, num_heads))
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_v_dim = head_v_dim
        self.layer_idx = layer_idx

    @property
    def routes(self):
        return self.layer.routes

    @property
    def topology(self):
        return self.layer.topology

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, None, Cache | None]:
        if output_attentions:
            raise NotImplementedError("RoLA has no attention matrix to return")
        cached = get_layer_cache(self, past_key_values)
        if cached is not None and cached["recurrent_state"] is not None:
            continuation = rola.LayerContinuation(state=cached["recurrent_state"])
        else:
            continuation = rola.LayerContinuation(state=rola.state()) if use_cache else None
        o, continuation = self.layer(hidden_states, attention_mask, continuation, **kwargs)
        if use_cache:
            update_layer_cache(self, past_key_values, recurrent_state=continuation.state,
                               offset=hidden_states.shape[1])
        return o, None, past_key_values

    def get_stats(self) -> dict:
        return self.layer.get_stats()

    def state_size(self, sequence_length: int | None = None, **kwargs) -> int:
        return self.layer.state_size(sequence_length)


__all__ = ["DECAY_SOURCES", "LevelRoutingConfig", "RoLA", "decay_source"]
