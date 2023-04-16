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
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import List

import coloredlogs
from imohash import hashfile
import onnx
from onnxruntime.tools.convert_onnx_models_to_ort import convert_onnx_models_to_ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from termcolor import cprint
import tomli as toml
import tomli_w as tomlw
from yaspin import yaspin

from _export import onnx_simplify
from _utils import SPINNER, collect_garbage

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
#coloredlogs.set_level('debug')

parser = ArgumentParser(prog='optimize', description='Optimize pyke Diffusers models.')
parser.add_argument('in_path', type=Path, help='Path to the model to optimize.')
parser.add_argument(
	'--preset',
	type=str,
	choices=[
		'cpu', # Optimize for CPU inference.
		'cpu-q8', # Optimize for CPU inference with 8-bit quantization.
		'gpu', # Optimize for generic GPU inference.
		'gpu-nv', # Optimize for GPU inference on NVIDIA GPUs with Tensor Cores.
		'mixed', # Optimize for 'mixed' GPU(UNet)/CPU(Rest) inference.
		'mixed-nv', # Optimize for 'mixed' GPU(UNet)/CPU(Rest) inference for NVIDIA GPUs with Tensor Cores.
		'mixed-nv-nightly', # Optimize for 'mixed' GPU(UNet)/CPU(Rest) inference for NVIDIA GPUs with Tensor Cores using a nightly version of ORT.
	],
	help='Optimization preset. See the docs for specific information.'
)
parser.add_argument('--fp16', type=str, help='List models to convert to float16. Saves disk space, memory, and boosts speed on GPUs with little quality loss.')
parser.add_argument('--group-norm', type=str, help='Use fused GroupNorm kernel for NHWC tensor layout.')
parser.add_argument('--skip-layer-norm', type=str, help='Fuses LayerNormalization with Add bias and residual inputs.')
parser.add_argument('--attention', type=str, help='Fuse Attention nodes.')
parser.add_argument('--multihead-attention', type=str, help='Fuse attention nodes into MultiHeadAttention. CUDA only.')
parser.add_argument('--approximate-gelu', type=str, help='Use approximate formula for GELU.')
parser.add_argument('--nhwc', type=str, help='Use NHWC format for convolutions.')
parser.add_argument('--bias-split-gelu', type=str, help='Fuse Add bias with SplitGelu activation.')
parser.add_argument('--bias-add', type=str, help='Fuse BiasAdd nodes. Nightly only.')
parser.add_argument('--packed-qkv', type=str, help='Use packed QKV for MultiHeadAttention. Nightly only.')
parser.add_argument('--packed-kv', type=str, help='Use packed KV for MultiHeadAttention.')
args = parser.parse_args()

PRESETS = {
	'cpu': {
		'text-encoder': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'unet': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'vae': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'safety-checker': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
	},
	'cpu-q8': {
		'text-encoder': ['skip_layer_norm', 'attention', 'multihead_attention', 'packed_kv'],
		'unet': ['skip_layer_norm', 'attention', 'multihead_attention', 'packed_kv'],
		'vae': ['skip_layer_norm', 'attention', 'multihead_attention', 'packed_kv'],
		'safety-checker': ['skip_layer_norm', 'attention', 'multihead_attention', 'packed_kv'],
	},
	'gpu': {
		'text-encoder': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'unet': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'vae': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'safety-checker': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
	},
	'gpu-nv': {
		'text-encoder': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'nhwc', 'packed_kv'],
		'unet': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'nhwc', 'bias_split_gelu', 'packed_kv'],
		'vae': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'nhwc', 'bias_split_gelu', 'packed_kv'],
		'safety-checker': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'nhwc', 'packed_kv'],
	},
	'mixed': {
		'text-encoder': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'unet': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'vae': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'safety-checker': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
	},
	'mixed-nv': {
		'text-encoder': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'unet': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'nhwc', 'bias_split_gelu', 'packed_kv'],
		'vae': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'safety-checker': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
	},
	'mixed-nv-nightly': {
		'text-encoder': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'unet': ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'nhwc', 'bias_split_gelu', 'bias_add', 'packed_qkv', 'packed_kv'],
		'vae': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
		'safety-checker': ['skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'packed_kv'],
	}
}

in_path: Path = args.in_path
if not os.path.exists(args.in_path):
	cprint('model doesn\'t exist!', color='red')
	sys.exit(1)

model_index = toml.load(open(in_path / 'pyke-diffusers.toml', mode='rb'))
if model_index['framework']['type'] != 'orte':
	cprint('model is not ONNX based; cannot optimize', color='red')
	sys.exit(1)

def get_config(name: str) -> List[str]:
	v_args = vars(args)
	options = []
	if args.preset and args.preset in PRESETS:
		for opt in PRESETS[args.preset][name]:
			options.append(opt)
	for opt in ['fp16', 'group_norm', 'skip_layer_norm', 'attention', 'multihead_attention', 'approximate_gelu', 'nhwc', 'bias_add', 'packed_qkv', 'packed_kv']:
		if name in (v_args[opt] or '') or v_args[opt] == '*':
			options.append(opt)
	return options

def create_fusion_options(model_type: str) -> FusionOptions:
	fusion_options = FusionOptions(model_type)
	fusion_options.enable_gelu = True
	fusion_options.enable_layer_norm = True
	return fusion_options

def get_fusion_options(model_type: str, onnx_model_type: str) -> FusionOptions:
	fusion_options = create_fusion_options(onnx_model_type)
	config = get_config(model_type)
	fusion_options.enable_attention = 'attention' in config
	fusion_options.use_multi_head_attention = 'multihead_attention' in config
	fusion_options.enable_skip_layer_norm = 'skip_layer_norm' in config
	fusion_options.enable_embed_layer_norm = True
	fusion_options.enable_bias_skip_layer_norm = 'skip_layer_norm' in config
	fusion_options.enable_bias_gelu = True
	fusion_options.enable_gelu_approximation = 'approximate_gelu' in config
	fusion_options.enable_qordered_matmul = True
	fusion_options.enable_shape_inference = model_type != 'unet'
	fusion_options.enable_gemm_fast_gelu = True
	fusion_options.enable_nhwc_conv = 'nhwc' in config
	fusion_options.enable_group_norm = 'group_norm' in config
	fusion_options.enable_bias_splitgelu = 'bias_split_gelu' in config
	fusion_options.enable_packed_qkv = 'packed_qkv' in config
	fusion_options.enable_packed_kv = 'fp16' in config or 'packed_kv' in config
	fusion_options.enable_bias_add = 'bias_add' in config
	return fusion_options

def fuse_vae():
	fusion_options = get_fusion_options('vae', 'vae')
	model = optimize_model(
		str(in_path / model_index['vae']['decoder']),
		model_type='vae',
		num_heads=0,
		hidden_size=0,
		opt_level=0,
		optimization_options=fusion_options,
		use_gpu=True
	)
	fp16 = 'fp16' in get_config('vae')
	if fp16:
		model.convert_float_to_float16(keep_io_types=True)

	model.get_fused_operator_statistics()
	model.save_model_to_file(str(in_path / 'vae_decoder.onnx'), use_external_data_format=False)
	cprint('🟢 VAE successfully optimized', color='green')
	del model

def fuse_unet():
	fusion_options = get_fusion_options('unet', 'unet')
	model = optimize_model(
		str(in_path / model_index['unet']['path']),
		model_type='unet',
		num_heads=0,
		hidden_size=0,
		opt_level=0,
		optimization_options=fusion_options,
		use_gpu=True
	)
	fp16 = 'fp16' in get_config('unet')
	if fp16:
		# note: MultiHeadAttention should be blocked for Stable Diffusion v2.1
		model.convert_float_to_float16(keep_io_types=True, op_block_list=['RandomNormalLike'])

	model.get_fused_operator_statistics()
	model.save_model_to_file(str(in_path / 'unet.onnx'), use_external_data_format=not fp16)
	cprint('🟢 UNet successfully optimized', color='green')
	del model

def fuse_text_encoder():
	fusion_options = get_fusion_options('text-encoder', 'clip')
	model = optimize_model(
		str(in_path / model_index['text-encoder']['path']),
		model_type='clip',
		num_heads=0,
		hidden_size=0,
		opt_level=0,
		optimization_options=fusion_options,
		use_gpu=True
	)
	fp16 = 'fp16' in get_config('text-encoder')
	if fp16:
		model.convert_float_to_float16(keep_io_types=True)

	model.get_fused_operator_statistics()
	model.save_model_to_file(str(in_path / 'text_encoder.onnx'), use_external_data_format=False)
	cprint('🟢 Text encoder successfully optimized', color='green')
	del model

fuse_vae()
fuse_text_encoder()
fuse_unet()

cprint('🔵 All models successfully optimized', color='blue')

model_index['hashes'] = {
	"text-encoder": hashfile(in_path / 'text_encoder.onnx', hexdigest=True),
	"unet": hashfile(in_path / 'unet.onnx', hexdigest=True),
	"vae-encoder": hashfile(in_path / 'vae_encoder.onnx', hexdigest=True),
	"vae-decoder": hashfile(in_path / 'vae_decoder.onnx', hexdigest=True)
}

if os.path.exists(in_path / 'safety_checker.onnx'):
	model_index['hashes']['safety-checker'] = hashfile(in_path / 'safety_checker.onnx', hexdigest=True)

with open(in_path / 'pyke-diffusers.toml', 'wb') as f:
	tomlw.dump(model_index, f)
