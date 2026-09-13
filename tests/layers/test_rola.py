# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""fla's RoLA is a thin wrapper: it builds rola's own layer from plain data and passes every call through."""
import pytest
import torch

rola = pytest.importorskip("rola")

from fla.layers import rola as rola_layer  # noqa: E402
from fla.layers.rola import LevelRoutingConfig, RoLA  # noqa: E402


def _pair(levels, decay):
    torch.manual_seed(0)
    wrapped = RoLA(hidden_size=128, num_heads=2, head_v_dim=64, levels=levels, decay=decay, layer_idx=0)
    return wrapped, dict(wrapped.layer.named_parameters())


def test_the_wrapper_holds_exactly_rolas_layer_built_from_the_same_arguments():
    wrapped, params = _pair([{"width": 8, "write": "sparse"}, LevelRoutingConfig(width=8, read="sparse", write="sparse")],
                            {"source": "learned", "target_leaf_rate": 2 ** -8})
    torch.manual_seed(0)
    routes = rola.RouteProducer([rola.IndependentRouting(width=8, read=rola.softmax(), write=rola.entmax(1.5)),
                                 rola.UnionRouting(width=8, alpha=1.5)], hidden_size=128, num_heads=2)
    reference = rola.RoLA(routes, d_v=64, layer_idx=0,
                          decay=rola.LearnedDecay(2 ** -8, widths=routes.topology.widths, num_heads=2))
    expected = dict(reference.named_parameters())
    assert type(wrapped.layer) is rola.RoLA
    assert sorted(params) == sorted(expected)
    assert all(torch.equal(params[name], expected[name]) for name in params)
    assert wrapped.get_stats() == reference.get_stats()


def test_a_level_dict_and_its_dataclass_build_the_same_layer():
    _, from_dict = _pair([{"width": 16, "read": "sparse", "tied": True, "alpha": 2.0}], None)
    _, from_config = _pair([LevelRoutingConfig(width=16, read="sparse", tied=True, alpha=2.0)], None)
    assert all(torch.equal(from_dict[name], from_config[name]) for name in from_dict)


@pytest.mark.parametrize("bad", [{"width": 0}, {"width": 8, "read": "soft"}, {"width": 8, "tied": 1}])
def test_a_malformed_level_is_refused_at_construction(bad):
    with pytest.raises((TypeError, ValueError)):
        LevelRoutingConfig.of(bad)


def test_an_unknown_decay_source_is_refused():
    with pytest.raises(ValueError, match="decay"):
        RoLA(hidden_size=128, num_heads=2, levels=[{"width": 8}], decay={"source": "inclusion_gated"})


def test_what_rola_refuses_the_wrapper_refuses_with_rolas_own_error():
    layer = RoLA(hidden_size=128, num_heads=2, levels=[{"width": 8}], layer_idx=0).eval()
    with pytest.raises(Exception) as through_wrapper:
        layer(torch.randn(1, 4, 128))
    with pytest.raises(Exception) as direct:
        layer.layer(torch.randn(1, 4, 128))
    assert type(through_wrapper.value) is type(direct.value)
    assert str(through_wrapper.value) == str(direct.value)


def test_without_rola_the_layer_refuses_at_construction(monkeypatch):
    monkeypatch.setattr(rola_layer, "rola", None)
    with pytest.raises(ImportError, match="rola"):
        RoLA(hidden_size=128, num_heads=2, levels=[{"width": 8}])
