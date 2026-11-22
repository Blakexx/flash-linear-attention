# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.fsla.configuration_fsla import FSLAConfig
from fla.models.fsla.modeling_fsla import FSLAForCausalLM, FSLAModel

AutoConfig.register(FSLAConfig.model_type, FSLAConfig, exist_ok=True)
AutoModel.register(FSLAConfig, FSLAModel, exist_ok=True)
AutoModelForCausalLM.register(FSLAConfig, FSLAForCausalLM, exist_ok=True)


__all__ = ['FSLAConfig', 'FSLAForCausalLM', 'FSLAModel']
