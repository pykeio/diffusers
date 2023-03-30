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

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import warnings
import shutil
import sys

import coloredlogs
from imohash import hashfile
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
#from onnxruntime.transformers.optimizer import optimize_by_fusion
from yaspin import yaspin

from _export import onnx_simplify
from _utils import SPINNER, collect_garbage

#warnings.filterwarnings('ignore')

import logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.DEBUG)

sys.path.append(os.path.join(os.path.dirname(__file__), '_optimizer'))
from fusion_options import FusionOptions
from onnx_model_clip import ClipOnnxModel
from onnx_model_unet import UnetOnnxModel
from onnx_model_vae import VaeOnnxModel
from optimizer import optimize_by_onnxruntime, optimize_model

coloredlogs.install(fmt="%(funcName)20s: %(message)s")

parser = ArgumentParser(prog='optimize', description='Optimize pyke Diffusers models.')
parser.add_argument('in_path', type=Path, help='Path to the model to optimize.')
parser.add_argument('-H', '--fp16', action='store_true', help='Convert all models to float16. Saves disk space, memory, and boosts speed on GPUs with little quality loss.')
parser.add_argument('--fp16-unet', action='store_true', help='Only convert the UNet to float16. Can be beneficial when only the UNet is placed on GPU.')
args = parser.parse_args()

in_path: Path = args.in_path
if not os.path.exists(args.in_path):
	print('model doesn\'t exist!')

#model_index = json.load(open(in_path / 'diffusers.json'))
#if model_index['framework'] != 'onnx':
#	print('model is not ONNX based; cannot optimize')
#	sys.exit(1)

#with yaspin(text='Optimizing UNet (this takes a long time!)', spinner=SPINNER):
fusion_options = FusionOptions('unet')

fusion_options.enable_gemm_fast_gelu = True
fusion_options.enable_gelu_approximation = True
fusion_options.use_multi_head_attention = False

fusion_options.enable_packed_kv = True
fusion_options.enable_packed_qkv = False
fusion_options.enable_bias_add = False

m = optimize_model(
	str(in_path / 'unet.onnx'),
	model_type='unet',
	num_heads=0,
	hidden_size=0,
	opt_level=0,
	optimization_options=fusion_options,
	use_gpu=True
)

m.convert_float_to_float16(
	keep_io_types=True,
	op_block_list=['RandomNormalLike'],
)
m.get_fused_operator_statistics()
m.save_model_to_file(in_path / 'unet.opti.onnx', use_external_data_format=False)

raise Exception()

with yaspin(text='Simplifying text encoder', spinner=SPINNER):
	onnx_simplify(in_path / 'text_encoder.onnx')
with yaspin(text='Simplifying VAE', spinner=SPINNER):
	onnx_simplify(in_path / 'vae_encoder.onnx')
	onnx_simplify(in_path / 'vae_decoder.onnx')

model_index['hashes'] = {
	"text-encoder": hashfile(in_path / 'text_encoder.onnx', hexdigest=True),
	"unet": hashfile(in_path / 'unet.onnx', hexdigest=True),
	"vae-encoder": hashfile(in_path / 'vae_encoder.onnx', hexdigest=True),
	"vae-decoder": hashfile(in_path / 'vae_decoder.onnx', hexdigest=True),
	"safety-checker": None
}

if os.path.exists(in_path / 'safety_checker.onnx'):
	model_index['hashes']['safety-checker'] = hashfile(in_path / 'safety_checker.onnx', hexdigest=True)

with open(in_path / 'diffusers.json', 'w') as f:
	json.dump(model_index, f)
