# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""FSLAConfig -- HF config for the FSLA causal LM. Mirrors GLAConfig (same backbone, MLP, norm and fuse
fields); the sequence mixer is `fla.layers.FSLA`, described by its construction arguments, which are fsla's own:
the routing `levels` and the `decay` as the JSON of fsla's objects (`fla.layers.fsla.encode`), `d_v`, `num_heads`,
`bias` and `gain_bias_init`."""
from transformers.configuration_utils import PretrainedConfig


class FSLAConfig(PretrainedConfig):

    model_type = 'fsla'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        # --- FSLA mixer (fla.layers.FSLA's arguments) ---
        levels: list | None = None,                          # fsla level types as JSON, outermost first
        d_v: int = 64,
        num_heads: int = 8,
        decay: dict | None = None,                           # None or a fsla decay source as JSON
        bias: bool = False,
        gain_bias_init: float | None = None,
        # --- backbone (mirrors GLAConfig) ---
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,   # FLA convention (matches GLAConfig)
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.levels = levels if levels is not None else [
            {"type": "IndependentRouting", "width": 16, "read": {"type": "SoftmaxActivation"},
             "write": {"type": "SoftmaxActivation"}}]
        self.d_v = d_v
        self.num_heads = num_heads
        self.decay = decay
        self.bias = bias
        self.gain_bias_init = gain_bias_init

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot both be True.")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
