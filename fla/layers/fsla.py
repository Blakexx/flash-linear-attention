# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""FSLA in FLA's mixer contract: a thin wrapper over the standalone ``fsla`` package's layer.

``fsla`` owns the model -- the route producer, the recurrence, the op and its envelope.
This file adds exactly what FLA's mixer contract needs and nothing that computes FSLA:

1. construction from ``fsla``'s own objects: the routing levels are ``fsla``'s level types
   and the decay is one of its decay sources, each also accepted as its JSON (:func:`encode` /
   :func:`decode`: the ``fsla`` class by name and its own fields), so an HF config carries
   exactly ``fsla``'s vocabulary and nothing translates it;
2. FLA's call convention: ``forward(hidden_states, attention_mask, past_key_values,
   use_cache, ...)`` returning ``(o, None, past_key_values)``, the layer's continuation
   carried in the cache's ``recurrent_state``;
3. FLA's state accounting: ``get_stats`` / ``state_size``, read off the wrapped layer.

What ``fsla`` refuses, this refuses, by passing the call through: a configuration outside
the built envelope, training before the native backward, prefill while the kernel is
rebuilt. There is no second implementation to fall back on.

``fsla`` is an optional dependency: ``fla`` imports without it, and constructing :class:`FSLA` without it raises.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from fla.layers.utils import get_layer_cache, update_layer_cache

try:
    import fsla
except ImportError:
    fsla = None

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def _require_fsla() -> None:
    if fsla is None:
        raise ImportError("fla.layers.FSLA needs the `fsla` package (https://github.com/Blakexx/fsla)")


def encode(value: Any) -> Any:
    """A ``fsla`` routing value as JSON: ``{"type": <its fsla class>, **its constructor fields}``, recursively.

    Plain values and lists pass through. A decay source is written the same way by hand (its class and its own
    arguments), since its ``widths`` and ``num_heads`` are derived from the levels when it is built.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__name__,
                **{f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def decode(blob: Any, **derived: Any) -> Any:
    """The ``fsla`` object a JSON blob names: ``fsla.<type>`` built from its decoded fields, plus whichever of the
    ``derived`` arguments its constructor takes. ``fsla`` objects, plain values and lists pass through."""
    if isinstance(blob, (list, tuple)):
        return [decode(item) for item in blob]
    if not isinstance(blob, dict) or "type" not in blob:
        return blob
    _require_fsla()
    name = blob["type"]
    if name not in fsla.__all__:
        raise ValueError(f"{name!r} is not a type the fsla package exports")
    cls = getattr(fsla, name)
    arguments = {key: decode(value) for key, value in blob.items() if key != "type"}
    accepted = inspect.signature(cls).parameters
    arguments.update({key: value for key, value in derived.items() if key in accepted and key not in arguments})
    return cls(**arguments)


class FSLA(nn.Module):
    """Routed linear attention as an FLA mixer, over ``fsla.FSLA`` built from ``fsla.RouteProducer``.

    Args:
        hidden_size: model width (``fsla.RouteProducer``'s ``hidden_size``).
        num_heads: heads (``fsla.RouteProducer``'s ``num_heads``).
        levels: the routing levels, outermost first: ``fsla.IndependentRouting`` / ``fsla.TiedRouting`` /
            ``fsla.UnionRouting``, or their JSON.
        d_v: per-head value width (``fsla.FSLA``'s ``d_v``).
        decay: ``None``, a ``fsla`` decay source, or its JSON (``{"type": "LearnedDecay", "target_leaf_rate": ...}``);
            its ``widths`` and ``num_heads`` come from the levels.
        bias: bias on the route projection (``fsla.RouteProducer``'s ``bias``).
        gain_bias_init: the side gain's bias initialization (``fsla.RouteProducer``'s ``gain_bias_init``).
        layer_idx: this layer's index in the cache.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        levels: list | tuple = ({"type": "IndependentRouting", "width": 16, "read": {"type": "SoftmaxActivation"},
                                 "write": {"type": "SoftmaxActivation"}},),
        d_v: int = 64,
        decay: Any = None,
        bias: bool = False,
        gain_bias_init: float | None = None,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        _require_fsla()
        routes = fsla.RouteProducer(decode(list(levels)), hidden_size=hidden_size, num_heads=num_heads, bias=bias,
                                    gain_bias_init=gain_bias_init)
        decay = decode(decay, widths=routes.topology.widths, num_heads=num_heads)
        self.layer = fsla.FSLA(routes, d_v=d_v, decay=decay, layer_idx=layer_idx)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_v = d_v
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
            raise NotImplementedError("FSLA has no attention matrix to return")
        cached = get_layer_cache(self, past_key_values)
        if cached is not None and cached["recurrent_state"] is not None:
            continuation = fsla.LayerContinuation(state=cached["recurrent_state"])
        else:
            continuation = fsla.LayerContinuation(state=fsla.state()) if use_cache else None
        o, continuation = self.layer(hidden_states, attention_mask, continuation, **kwargs)
        if use_cache:
            update_layer_cache(self, past_key_values, recurrent_state=continuation.state,
                               offset=hidden_states.shape[1])
        return o, None, past_key_values

    def get_stats(self) -> dict:
        return self.layer.get_stats()

    def state_size(self, sequence_length: int | None = None, **kwargs) -> int:
        return self.layer.state_size(sequence_length)


__all__ = ["FSLA", "decode", "encode"]
