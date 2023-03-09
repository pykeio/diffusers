from argparse import ArgumentParser
import json
import os
from pathlib import Path
import warnings
import sys

from imohash import hashfile
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.optimizer import optimize_by_fusion
from yaspin import yaspin

from _export import onnx_simplify
from _utils import SPINNER, collect_garbage

warnings.filterwarnings('ignore')

parser = ArgumentParser(prog='optimize', description='Optimize pyke Diffusers models.')
parser.add_argument('in_path', type=Path, help='Path to the model to optimize.')
parser.add_argument('-H', '--fp16', action='store_true', help='Convert all models to float16. Saves disk space, memory, and boosts speed on GPUs with little quality loss.')
parser.add_argument('--fp16-unet', action='store_true', help='Only convert the UNet to float16. Can be beneficial when only the UNet is placed on GPU.')
args = parser.parse_args()

in_path: Path = args.in_path
if not os.path.exists(args.in_path):
	print('model doesn\'t exist!')

model_index = json.load(open(in_path / 'diffusers.json'))
if model_index['framework'] != 'onnx':
	print('model is not ONNX based; cannot optimize')
	sys.exit(1)

with yaspin(text='Optimizing UNet (this takes a long time!)', spinner=SPINNER):
	model = onnx.load_model(str(in_path / 'unet.onnx'))
	if 'onnxruntime' in model.producer_name:
		print('model is already optimized!')
		sys.exit(1)

	optimizer = optimize_by_fusion(model, 'unet')
	if args.fp16:
		optimizer.convert_float_to_float16(keep_io_types=True)
	optimizer.save_model_to_file(str(in_path / 'unet.opt.onnx'), False)

	del model
	del optimizer
	collect_garbage()

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
