# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rola.configuration_rola import RoLAConfig
from fla.models.rola.modeling_rola import RoLAForCausalLM, RoLAModel

AutoConfig.register(RoLAConfig.model_type, RoLAConfig, exist_ok=True)
AutoModel.register(RoLAConfig, RoLAModel, exist_ok=True)
AutoModelForCausalLM.register(RoLAConfig, RoLAForCausalLM, exist_ok=True)


__all__ = ['RoLAConfig', 'RoLAForCausalLM', 'RoLAModel']
