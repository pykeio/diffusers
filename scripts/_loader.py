# Copyright 2022-2023 pyke.io
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Type, TypeVar

import accelerate
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file as safetensors_load
import torch

T = TypeVar('T')

@torch.inference_mode()
def load_accelerate(cls: Type[T], root: Path, checkpoint_name = 'diffusion_pytorch_model.bin', device=None, dtype=None) -> T:
	with accelerate.init_empty_weights():
		model = cls.from_config( # type: ignore
			config_path=root / 'config.json',
			pretrained_model_name_or_path=root
		)

	accelerate.load_checkpoint_and_dispatch(model, str(root / checkpoint_name), device_map='auto')

	model = model.to(dtype=dtype, device=device)
	model.eval()
	return model

def load_standard(cls: Type[T], root: Path, checkpoint_name = 'diffusion_pytorch_model.bin', device=None, dtype=None) -> T:
	model = cls.from_pretrained(root) # type: ignore
	model = model.to(dtype=dtype, device=device)
	model.eval()
	return model

def load_safetensors(cls: Type[T], root: Path, checkpoint_name = 'diffusion_pytorch_model.safetensors', device: torch.device = torch.device('cpu'), dtype=None) -> T:
	with accelerate.init_empty_weights():
		model = cls.from_config( # type: ignore
			config_path=root / 'config.json',
			pretrained_model_name_or_path=root
		)

	state_dict = safetensors_load(root / checkpoint_name, device='cpu')
	for name, param in state_dict.items():
		set_module_tensor_to_device(model, name, device, value=param, dtype=dtype)

	return model
