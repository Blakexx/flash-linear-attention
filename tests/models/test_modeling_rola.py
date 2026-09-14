# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""RoLA's HF model: built from its config, every block's mixer rola's own layer, the config round-tripping."""
import pytest

rola = pytest.importorskip("rola")

from fla.layers.rola import encode  # noqa: E402
from fla.models import RoLAConfig, RoLAForCausalLM  # noqa: E402


def test_the_hf_model_is_built_from_the_config_and_the_config_round_trips():
    config = RoLAConfig(levels=encode([rola.IndependentRouting(width=8, read=rola.softmax(), write=rola.softmax()),
                                       rola.TiedRouting(width=8, op=rola.entmax(1.5))]), d_v=64, num_heads=2,
                        hidden_size=128, num_hidden_layers=2, vocab_size=100, fuse_cross_entropy=False,
                        fuse_norm=False, fuse_swiglu=False)
    model = RoLAForCausalLM(config)
    assert [type(block.attn.layer) for block in model.model.layers] == [rola.RoLA, rola.RoLA]
    assert RoLAConfig.from_dict(config.to_dict()).levels == config.levels


def test_the_model_saves_and_loads_through_the_auto_classes(tmp_path):
    from transformers import AutoModelForCausalLM

    config = RoLAConfig(levels=encode([rola.UnionRouting(width=8)]), d_v=64, num_heads=2, hidden_size=128, num_hidden_layers=1,
                        vocab_size=100, fuse_cross_entropy=False, fuse_norm=False, fuse_swiglu=False)
    model = RoLAForCausalLM(config)
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert type(loaded) is RoLAForCausalLM
    saved, restored = model.state_dict(), loaded.state_dict()
    assert sorted(saved) == sorted(restored) and all(saved[k].equal(restored[k]) for k in saved)
