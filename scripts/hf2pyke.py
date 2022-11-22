from argparse import ArgumentParser
from datetime import date
import errno
import gc
import json
import os
from pathlib import Path
import posixpath
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union
import warnings

import accelerate
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.vae import AutoencoderKL, AutoencoderKLOutput, DecoderOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import snapshot_download # type: ignore
import onnx
from termcolor import cprint
import torch
from transformers import CLIPTextModel, CLIPTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPooling
from yaspin import yaspin, spinners

warnings.filterwarnings('ignore')

spinner = spinners.Spinners._asdict()['clock' if date.today().month == 3 and date.today().day == 14 else 'dots12']
root = Path(os.path.dirname(os.path.realpath(__file__))).parent

parser = ArgumentParser(prog='hf2pyke', description='Converts HuggingFace Diffusers models to pyke Diffusers models')
parser.add_argument('hf_path', type=Path, help='Path to the HuggingFace model to convert.')
parser.add_argument('out_path', type=Path, help='Output path.')
parser.add_argument('-f16', '--fp16', action='store_true', help='Whether or not the input model is in float16 format.')
parser.add_argument('--no-collate', action='store_true', help='Do not collate UNet weights into a single file.')
args = parser.parse_args()

def collect_garbage():
	torch.cuda.empty_cache()
	gc.collect()

def mkdirp(path: Path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise e

out_path: Path = args.out_path.resolve()

if not os.path.exists(out_path):
	mkdirp(out_path)

hf_path: Path = args.hf_path
if not os.path.exists(args.hf_path):
	revision: Optional[str] = None
	if len(str(hf_path).split('@')) == 2:
		repo, revision = str(hf_path).split('@')
	else:
		repo = str(hf_path)

	if revision == 'fp16' and not args.fp16:
		args.fp16 = True

	repo = repo.replace(os.sep, posixpath.sep)

	hf_cache_path = os.path.expanduser(os.getenv('HF_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'huggingface')))
	diffusers_cache_path = os.path.join(hf_cache_path, 'diffusers')

	hf_path = Path(snapshot_download(
		repo,
		revision=revision,
		cache_dir=diffusers_cache_path,
		token=True,
		user_agent='pykeDiffusers/1.0',
		resume_download=True,
		allow_patterns=[
			"feature_extractor/*",
			"safety_checker/*",
			"text_encoder/*",
			"tokenizer/*",
			"unet/*",
			"vae/*",
			"model_index.json"
		]
	))

model_index = json.load(open(hf_path / 'model_index.json'))
if not model_index['_diffusers_version']:
	print('repo is not a HuggingFace diffusers model')
	exit(1)
if model_index['_class_name'] != 'StableDiffusionPipeline':
	print('repo is not a Stable Diffusion model; only Stable Diffusion models are supported')
	exit(1)

OPSET = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DTYPE = torch.float16 if args.fp16 else torch.float32
IO_DTYPE = torch.float32

tokenizer = CLIPTokenizerFast.from_pretrained(hf_path / 'tokenizer')

@yaspin(text='Reformatting tokenizer', spinner=spinner)
def reformat_tokenizer(out_path: Path):
	with TemporaryDirectory() as tmp:
		tokenizer.save_pretrained(tmp)

		res = subprocess.run(
			[
				'cargo', 'run', '--bin', 'diffusers-reserialize-clip',
				'--features=stable-diffusion,serde,serde_json',
				'--color', 'always',
				'--',
				Path(tmp) / 'tokenizer.json',
				out_path,
				str(tokenizer.model_max_length),
				str(tokenizer.bos_token_id),
				str(tokenizer.eos_token_id)
			],
			cwd=root,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			universal_newlines=True
		)
		if res.returncode != 0:
			sys.stderr.write(res.stdout)
			raise RuntimeError(f'`diffusers-reserialize-clip` exited with code {res.returncode}')

@torch.inference_mode()
def onnx_export(
	model: torch.nn.Module,
	model_args: tuple,
	output_path: Path,
	ordered_input_names: List[str],
	output_names: List[str],
	dynamic_axes: Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
):
	torch.onnx.export(
		model,
		model_args,
		f=output_path.as_posix(),
		input_names=ordered_input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
		do_constant_folding=True,
		opset_version=OPSET
	)

class UNet2DConditionModelIOWrapper(UNet2DConditionModel):
	def forward(
		self,
		sample: torch.Tensor,
		timestep: torch.Tensor,
		encoder_hidden_states: torch.Tensor
	) -> Tuple:
		sample = sample.to(dtype=MODEL_DTYPE)
		timestep = timestep.to(dtype=torch.long)
		encoder_hidden_states = encoder_hidden_states.to(dtype=MODEL_DTYPE)

		sample = UNet2DConditionModel.forward(self, sample, timestep, encoder_hidden_states, return_dict=True).sample # type: ignore
		return (sample.to(dtype=IO_DTYPE),)

class CLIPTextModelIOWrapper(CLIPTextModel):
	def forward(self, input_ids: torch.IntTensor) -> Tuple:
		outputs: BaseModelOutputWithPooling = CLIPTextModel.forward(self, input_ids=input_ids, return_dict=True) # type: ignore
		return (outputs.last_hidden_state.to(dtype=IO_DTYPE), outputs.pooler_output.to(dtype=IO_DTYPE))

class AutoencoderKLIOWrapper(AutoencoderKL):
	def encode(self, x: torch.Tensor) -> Tuple:
		x = x.to(dtype=MODEL_DTYPE)

		outputs: AutoencoderKLOutput = AutoencoderKL.encode(self, x, True) # type: ignore
		return (outputs.latent_dist.sample().to(dtype=IO_DTYPE),)

	def decode(self, z: torch.Tensor) -> Tuple:
		z = z.to(dtype=MODEL_DTYPE)

		outputs: DecoderOutput = AutoencoderKL.decode(self, z, True) # type: ignore
		return (outputs.sample.to(dtype=IO_DTYPE),)

T = TypeVar('T')

@torch.inference_mode()
def load_efficient(cls: Type[T], root: Path, checkpoint_name = 'diffusion_pytorch_model.bin') -> T:
	with accelerate.init_empty_weights():
		model = cls.from_config( # type: ignore
			config_path=root / 'config.json',
			pretrained_model_name_or_path=root
		)

	accelerate.load_checkpoint_and_dispatch(model, root / checkpoint_name, device_map='auto')

	model = model.to(dtype=MODEL_DTYPE, device=DEVICE)
	model.eval()
	return model

@yaspin(text='Converting text encoder', spinner=spinner)
def convert_text_encoder() -> Tuple[int, int]:
	text_encoder: CLIPTextModelIOWrapper = CLIPTextModelIOWrapper.from_pretrained(hf_path / 'text_encoder') # type: ignore
	text_encoder = text_encoder.to(dtype=MODEL_DTYPE, device=DEVICE)
	text_encoder.eval()

	num_tokens = text_encoder.config.max_position_embeddings
	text_hidden_size = text_encoder.config.hidden_size
	max_length = tokenizer.model_max_length

	text_input: torch.IntTensor = tokenizer(
		"It's ocean law! If it's in the ocean long enough, it's yours!",
		padding='max_length',
		max_length=max_length,
		truncation=True,
		return_tensors='pt'
	).input_ids.to(device=DEVICE, dtype=torch.int32)

	onnx_export(
		text_encoder,
		model_args=(text_input,),
		output_path=out_path / 'text_encoder.onnx',
		ordered_input_names=['input_ids'],
		output_names=['last_hidden_state', 'pooler_output'],
		dynamic_axes={
			"input_ids": {0: "batch", 1: "sequence"}
		}
	)

	del text_encoder
	collect_garbage()

	return num_tokens, text_hidden_size

@yaspin(text='Converting UNet', spinner=spinner)
def convert_unet(num_tokens: int, text_hidden_size: int):
	unet = load_efficient(UNet2DConditionModelIOWrapper, hf_path / 'unet')

	unet_model_size = 0
	for param in unet.parameters():
		unet_model_size += param.nelement() * param.element_size()
	for buffer in unet.buffers():
		unet_model_size += buffer.nelement() * buffer.element_size()

	unet_model_size_mb = unet_model_size / 1024**2
	needs_collate = unet_model_size_mb > 2000

	in_channels = unet.config['in_channels']
	sample_size = unet.config['sample_size']

	unet_out_path = out_path
	if needs_collate:
		unet_out_path = out_path / 'unet_data'
		mkdirp(unet_out_path)

	onnx_export(
		unet,
		model_args=(
			torch.randn(2, in_channels, sample_size, sample_size).to(device=DEVICE, dtype=IO_DTYPE),
			torch.randn(2).to(device=DEVICE, dtype=IO_DTYPE),
			torch.randn(2, num_tokens, text_hidden_size).to(device=DEVICE, dtype=IO_DTYPE)
		),
		output_path=unet_out_path / 'unet.onnx',
		ordered_input_names=['sample', 'timestep', 'encoder_hidden_states', 'return_dict'],
		output_names=['out_sample'],
		dynamic_axes={
			"sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
			"timestep": {0: "batch"},
			"encoder_hidden_states": {0: "batch", 1: "sequence"}
		}
	)

	del unet
	collect_garbage()

	if needs_collate and not args.no_collate:
		unet = onnx.load(str((unet_out_path / 'unet.onnx').absolute().as_posix()))
		onnx.save_model(
			unet,
			str((out_path / 'unet.onnx').absolute().as_posix()),
			save_as_external_data=True,
			all_tensors_to_one_file=True,
			location="unet.pb",
			convert_attribute=False
		)

		del unet
		collect_garbage()

		shutil.rmtree(unet_out_path)

	return sample_size

@yaspin(text='Converting VAE', spinner=spinner)
def convert_vae(unet_sample_size: int):
	vae = load_efficient(AutoencoderKLIOWrapper, hf_path / 'vae')

	vae_in_channels = vae.config['in_channels']
	vae_sample_size = vae.config['sample_size']
	vae_latent_channels = vae.config['latent_channels']
	vae_out_channels = vae.config['out_channels']

	vae.forward = lambda sample: vae.encode(sample)[0] # type: ignore
	onnx_export(
		vae,
		model_args=(torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(device=DEVICE, dtype=IO_DTYPE),),
		output_path=out_path / 'vae_encoder.onnx',
		ordered_input_names=['sample'],
		output_names=['latent_sample'],
		dynamic_axes={
			"sample": {0: "batch", 1: "channels", 2: "height", 3: "width"}
		}
	)

	vae.forward = vae.decode # type: ignore
	onnx_export(
		vae,
		model_args=(torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=DEVICE, dtype=IO_DTYPE),),
		output_path=out_path / 'vae_decoder.onnx',
		ordered_input_names=['latent_sample'],
		output_names=['sample'],
		dynamic_axes={
			"latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"}
		}
	)

	del vae
	collect_garbage()

with torch.no_grad():
	reformat_tokenizer(out_path / 'tokenizer.bin')
	
	num_tokens, text_hidden_size = convert_text_encoder()
	unet_sample_size = convert_unet(num_tokens, text_hidden_size)
	convert_vae(unet_sample_size)

cprint(f'✨ Your model is ready! {str(out_path)}')
