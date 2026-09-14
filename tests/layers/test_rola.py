# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""fla's RoLA is a thin wrapper: it builds rola's own layer from rola's own objects (or their JSON) and passes every
call through."""
import json

import pytest
import torch

rola = pytest.importorskip("rola")

from fla.layers import rola as rola_layer  # noqa: E402
from fla.layers.rola import RoLA, decode, encode  # noqa: E402

LEVELS = [rola.IndependentRouting(width=8, read=rola.softmax(), write=rola.entmax(1.5)),
          rola.UnionRouting(width=8, alpha=1.5)]


def _built(levels, decay):
    torch.manual_seed(0)
    wrapped = RoLA(hidden_size=128, num_heads=2, levels=levels, d_v=64, decay=decay, layer_idx=0)
    return wrapped, dict(wrapped.layer.named_parameters())


def test_the_wrapper_holds_exactly_rolas_layer_built_from_the_same_objects():
    wrapped, params = _built(LEVELS, {"type": "LearnedDecay", "target_leaf_rate": 2 ** -8})
    torch.manual_seed(0)
    routes = rola.RouteProducer(LEVELS, hidden_size=128, num_heads=2)
    reference = rola.RoLA(routes, d_v=64, layer_idx=0,
                          decay=rola.LearnedDecay(2 ** -8, widths=routes.topology.widths, num_heads=2))
    expected = dict(reference.named_parameters())
    assert type(wrapped.layer) is rola.RoLA
    assert sorted(params) == sorted(expected)
    assert all(torch.equal(params[name], expected[name]) for name in params)
    assert wrapped.get_stats() == reference.get_stats()


def test_every_rola_level_type_survives_json():
    levels = [rola.IndependentRouting(width=8, read=rola.softmax(), write=rola.entmax(2.0)),
              rola.TiedRouting(width=16, op=rola.entmax(1.5)), rola.TiedRouting(width=4, op=rola.softmax()),
              rola.UnionRouting(width=8, alpha=1.5)]
    blob = json.loads(json.dumps(encode(levels)))
    assert blob[1] == {"type": "TiedRouting", "width": 16, "op": {"type": "EntmaxActivation", "alpha": 1.5}}
    assert decode(blob) == levels


def test_objects_and_their_json_build_the_same_layer():
    _, from_objects = _built(LEVELS, None)
    _, from_json = _built(json.loads(json.dumps(encode(LEVELS))), None)
    assert all(torch.equal(from_objects[name], from_json[name]) for name in from_objects)


def test_a_decay_source_takes_its_widths_and_heads_from_the_levels():
    wrapped, _ = _built(LEVELS, {"type": "ConstantDecay", "target_leaf_rate": 2 ** -6})
    assert type(wrapped.layer.decay) is rola.ConstantDecay
    assert wrapped.layer.decay.widths == (8, 8) and wrapped.layer.decay.num_heads == 2


def test_a_name_rola_does_not_export_is_refused():
    with pytest.raises(ValueError, match="rola package exports"):
        decode({"type": "os", "width": 8})


def test_what_rola_refuses_it_refuses_with_rolas_own_error():
    with pytest.raises(TypeError):
        decode({"type": "IndependentRouting", "width": 8, "read": {"type": "SoftmaxActivation"},
                "write": {"type": "SoftmaxActivation"}, "tied": True})


def test_what_rola_refuses_the_wrapper_refuses_with_rolas_own_error():
    layer = RoLA(hidden_size=128, num_heads=2, levels=LEVELS[:1], layer_idx=0).eval()
    with pytest.raises(Exception) as through_wrapper:
        layer(torch.randn(1, 4, 128))
    with pytest.raises(Exception) as direct:
        layer.layer(torch.randn(1, 4, 128))
    assert type(through_wrapper.value) is type(direct.value)
    assert str(through_wrapper.value) == str(direct.value)


def test_without_rola_the_layer_refuses_at_construction(monkeypatch):
    monkeypatch.setattr(rola_layer, "rola", None)
    with pytest.raises(ImportError, match="rola"):
        RoLA(hidden_size=128, num_heads=2, levels=LEVELS[:1])
