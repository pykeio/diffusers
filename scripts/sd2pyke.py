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
import contextlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, Dict, Optional, Tuple
import warnings

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
from imohash import hashfile
import onnx
from termcolor import cprint
import tomli_w as toml
import torch
from transformers import CLIPTextModel, CLIPTokenizerFast
from yaspin import yaspin

from _export import onnx_export
from _loader import load_safetensors
from _utils import SPINNER, collect_garbage, mkdirp

warnings.filterwarnings('ignore')

parser = ArgumentParser(prog='sd2pyke', description='Converts original Stable Diffusion checkpoints to pyke Diffusers models')
parser.add_argument('ckpt_path', type=Path, help='Path to the Stable Diffusion checkpoint to convert.')
parser.add_argument('out_path', type=Path, help='Output path.')
# !wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
parser.add_argument('-C', '--config-file', default='v1-inference.yaml', type=str, help='The YAML config file corresponding to the original architecture.')
parser.add_argument('--override-unet-sample-size', type=int, help='Override the sample size when converting the UNet.')
parser.add_argument('--prediction-type', default=None, type=str, help="The prediction type that the model was trained on. Use 'epsilon' for Stable Diffusion v1.x and Stable Diffusion v2 Base. Use 'v-prediction' for Stable Diffusion v2.")
parser.add_argument('-E', '--ema', action="store_true", help='Extract EMA weights from the checkpoint. EMA weights may yield higher quality images for inference.')
parser.add_argument('--upcast-attention', action='store_true', help='Whether the attention computation should always be upcasted. This is necessary when running Stable Diffusion 2.1 & derivatives.')
parser.add_argument('-H', '--fp16', action='store_true', help='Convert all models to float16. Saves disk space, memory, and boosts speed on GPUs with little quality loss.')
parser.add_argument('--fp16-unet', action='store_true', help='Only convert the UNet to float16. Can be beneficial when only the UNet is placed on GPU.')
parser.add_argument('--no-collate', action='store_true', help='Do not collate UNet weights into a single file.')
parser.add_argument('--skip-safety-checker', action='store_true', help='Skips converting the safety checker.')
parser.add_argument('-S', '--simplify-small-models', action='store_true', help='Run onnx-simplifier on the VAE and text encoder for a slight speed boost. Requires `pip install onnxsim` and ~6 GB of free RAM.')
parser.add_argument('--simplify-unet', action='store_true', help='Run onnx-simplifier on the UNet. Requires `pip install onnxsim` and an unholy amount of free RAM (>24 GB), probably not worth it.')
parser.add_argument('-O', '--opset', type=int, default=15, help='The ONNX opset version models will be output with.')
parser.add_argument('-q', '--quantize', type=str, help='Quantize models. See the documentation for more information.')
args = parser.parse_args()

out_path: Path = args.out_path.resolve()
if not os.path.exists(out_path):
	mkdirp(out_path)

model_config: Dict[str, Any] = {
	'v': 2,
	'pipeline': 'stable-diffusion',
	'framework': {
		'type': 'orte',
		'opset': args.opset
	}
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DTYPE = torch.float16 if args.fp16 else torch.float32
UNET_DTYPE = torch.float16 if args.fp16_unet else MODEL_DTYPE
IO_DTYPE = torch.float32

@contextlib.contextmanager
def cd(newdir, cleanup=lambda: None):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()

@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()
    def cleanup():
        shutil.rmtree(dirpath)
    with cd(dirpath, cleanup):
        yield dirpath

class UNet2DConditionModelIOWrapper(UNet2DConditionModel):
	def forward(
		self,
		sample: torch.Tensor,
		timestep: torch.Tensor,
		encoder_hidden_states: torch.Tensor
	) -> Tuple:
		sample = sample.to(dtype=UNET_DTYPE)
		timestep = timestep.to(dtype=torch.long)
		encoder_hidden_states = encoder_hidden_states.to(dtype=UNET_DTYPE)

		sample = UNet2DConditionModel.forward(self, sample, timestep, encoder_hidden_states, return_dict=True).sample # type: ignore
		return (sample.to(dtype=IO_DTYPE),)

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class CLIPTextModelIOWrapper(CLIPTextModel):
	def forward(self, input_ids: torch.IntTensor) -> Tuple:
		outputs: BaseModelOutputWithPooling = CLIPTextModel.forward(self, input_ids=input_ids, return_dict=True) # type: ignore
		return (outputs.last_hidden_state.to(dtype=IO_DTYPE), outputs.pooler_output.to(dtype=IO_DTYPE))

class CLIPPreembeddedTextModelIOWrapper(CLIPTextModel):
	def forward(self, token_embeddings: torch.Tensor, attention_mask: Optional[torch.BoolTensor] = None, position_ids: Optional[torch.Tensor] = None) -> Tuple:
		token_embeddings = token_embeddings.to(dtype=MODEL_DTYPE)

		batch, seq_length, _ = token_embeddings.shape
		if position_ids is None:
			position_ids = self.text_model.embeddings.position_ids[:, :seq_length] # type: ignore
		position_embeddings = self.text_model.embeddings.position_embedding(position_ids)
		hidden_states = token_embeddings + position_embeddings

		causal_attention_mask = self.text_model._build_causal_attention_mask(batch, seq_length, hidden_states.dtype).to(hidden_states.device)
		if attention_mask is not None:
			attention_mask = _expand_mask(attention_mask, hidden_states.dtype) # type: ignore

		encoder_outputs = self.text_model.encoder(
			inputs_embeds=hidden_states,
			attention_mask=attention_mask,
			causal_attention_mask=causal_attention_mask,
			return_dict=True
		)

		last_hidden_state = encoder_outputs[0]
		last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

		return last_hidden_state.to(dtype=IO_DTYPE)

	def encode_token_embeddings(self, input_ids: torch.IntTensor) -> torch.FloatTensor:
		return self.text_model.embeddings.token_embedding(input_ids.view(-1, input_ids.size(-1))).to(dtype=IO_DTYPE)

class AutoencoderKLIOWrapper(AutoencoderKL):
	def encode(self, x: torch.Tensor) -> Tuple:
		x = x.to(dtype=MODEL_DTYPE)

		outputs: AutoencoderKLOutput = AutoencoderKL.encode(self, x, True) # type: ignore
		return (outputs.latent_dist.sample().to(dtype=IO_DTYPE),)

	def decode(self, z: torch.Tensor) -> Tuple:
		z = z.to(dtype=MODEL_DTYPE)

		outputs: DecoderOutput = AutoencoderKL.decode(self, z, True) # type: ignore
		return (outputs.sample.to(dtype=IO_DTYPE),)

class SafetyCheckerIOWrapper(StableDiffusionSafetyChecker):
	def forward(self, clip_input: torch.Tensor, images: torch.Tensor) -> Tuple:
		clip_input = clip_input.to(dtype=MODEL_DTYPE)
		images = images.to(dtype=MODEL_DTYPE)

		images, has_nsfw_concepts = StableDiffusionSafetyChecker.forward_onnx(self, clip_input, images) # type: ignore
		return (images.to(dtype=IO_DTYPE), has_nsfw_concepts)
	
@yaspin(text='Converting text encoder', spinner=SPINNER)
def convert_text_encoder(hf_root: Path) -> Tuple[Path, Path, int, int]:
	text_encoder: CLIPPreembeddedTextModelIOWrapper = CLIPPreembeddedTextModelIOWrapper.from_pretrained(hf_root / 'text_encoder') # type: ignore
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

	embed_input = text_encoder.encode_token_embeddings(text_input)
	onnx_export(
		text_encoder,
		model_args=(embed_input,),
		output_path=out_path / 'text_encoder.onnx',
		ordered_input_names=['input_embeddings'],
		output_names=['last_hidden_state'],
		dynamic_axes={
			"input_embeddings": {0: "batch", 1: "sequence"}
		}
	)

	embeddings_path = out_path / 'text_embeddings.bin'
	embeddings_file = open(embeddings_path, 'wb')
	embeddings = text_encoder.text_model.embeddings.token_embedding.weight
	assert embeddings.size(0) == num_tokens
	assert embeddings.size(1) == text_hidden_size

	embeddings_file.write(struct.pack('L', num_tokens))
	embeddings_file.write(struct.pack('L', text_hidden_size))

	for token in embeddings.detach().to(dtype=torch.float32):
		for value in token:
			embeddings_file.write(struct.pack('f', value.item()))

	del text_encoder
	collect_garbage()

	return out_path / 'text_encoder.onnx', embeddings_path, num_tokens, text_hidden_size

@yaspin(text='Converting UNet', spinner=SPINNER)
def convert_unet(hf_path: Path, num_tokens: int, text_hidden_size: int) -> Tuple[Path, int]:
	unet = load_safetensors(UNet2DConditionModelIOWrapper, hf_path / 'unet', device=DEVICE, dtype=UNET_DTYPE)

	#if isinstance(unet.config.attention_head_dim, int): # type: ignore
	#	slice_size = unet.config.attention_head_dim // 2 # type: ignore
	#else:
	#	slice_size = min(unet.config.attention_head_dim) # type: ignore
	#unet.set_attention_slice(slice_size)

	unet_model_size = 0
	for param in unet.parameters():
		unet_model_size += param.nelement() * param.element_size()
	for buffer in unet.buffers():
		unet_model_size += buffer.nelement() * buffer.element_size()

	unet_model_size_mb = unet_model_size / 1024**2
	needs_collate = unet_model_size_mb > 2000

	in_channels = unet.config['in_channels']
	sample_size = args.override_unet_sample_size or unet.config['sample_size']

	unet_out_path = out_path
	if needs_collate:
		unet_out_path = out_path / 'unet_data'
		mkdirp(unet_out_path)

	onnx_export(
		unet,
		model_args=(
			# sample_size + 1 so non-default resolutions work - ONNX throws an error with just sample_size
			torch.randn(2, in_channels, sample_size, sample_size + 1).to(device=DEVICE, dtype=IO_DTYPE),
			torch.randn(2).to(device=DEVICE, dtype=IO_DTYPE),
			# (num_tokens * 3) - 2 to make LPW work.
			torch.randn(2, (num_tokens * 3) - 2, text_hidden_size).to(device=DEVICE, dtype=IO_DTYPE)
		),
		output_path=unet_out_path / 'unet.onnx',
		ordered_input_names=['sample', 'timestep', 'encoder_hidden_states'],
		output_names=['out_sample'],
		dynamic_axes={
			'sample': {0: 'batch', 1: 'channels', 2: 'height', 3: 'width'},
			'timestep': {0: 'batch'},
			'encoder_hidden_states': {0: 'batch', 1: 'sequence'}
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
			location='unet_weights.bin',
			convert_attribute=False
		)

		del unet
		collect_garbage()

		shutil.rmtree(unet_out_path)
		unet_out_path = out_path

	return unet_out_path / 'unet.onnx', sample_size

@yaspin(text='Converting VAE', spinner=SPINNER)
def convert_vae(hf_path: Path, unet_sample_size: int) -> Tuple[Path, Path, int, int, float]:
	vae = load_safetensors(AutoencoderKLIOWrapper, hf_path / 'vae', device=DEVICE, dtype=MODEL_DTYPE)

	vae_in_channels = vae.config['in_channels']
	vae_sample_size = vae.config['sample_size']
	vae_latent_channels = vae.config['latent_channels']
	vae_out_channels = vae.config['out_channels']
	vae_scale_factor = vae.config['scaling_factor']

	vae.forward = lambda sample: vae.encode(sample)[0] # type: ignore
	onnx_export(
		vae,
		model_args=(torch.randn(2, vae_in_channels, vae_sample_size, vae_sample_size).to(device=DEVICE, dtype=IO_DTYPE),),
		output_path=out_path / 'vae_encoder.onnx',
		ordered_input_names=['sample'],
		output_names=['latent_sample'],
		dynamic_axes={
			'sample': {0: 'batch', 1: 'channels', 2: 'height', 3: 'width'}
		}
	)

	vae.forward = vae.decode # type: ignore
	onnx_export(
		vae,
		model_args=(torch.randn(2, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=DEVICE, dtype=IO_DTYPE),),
		output_path=out_path / 'vae_decoder.onnx',
		ordered_input_names=['latent_sample'],
		output_names=['sample'],
		dynamic_axes={
			'latent_sample': {0: 'batch', 1: 'channels', 2: 'height', 3: 'width'}
		}
	)

	del vae
	collect_garbage()

	return out_path / 'vae_encoder.onnx', out_path / 'vae_decoder.onnx', vae_sample_size, vae_out_channels, vae_scale_factor

with torch.no_grad():
	pipe = load_pipeline_from_original_stable_diffusion_ckpt(
		checkpoint_path=args.ckpt_path,
		from_safetensors=args.ckpt_path.suffix == '.safetensors',
		image_size=((args.override_unet_sample_size or 0) * 8) or 512,
		prediction_type=args.prediction_type,
		extract_ema=args.ema,
		scheduler_type='pndm',
		upcast_attention=args.upcast_attention,
		original_config_file=args.config_file,
		device=DEVICE # type: ignore
	)
	pipe = pipe.to(DEVICE)

	with tempdir() as tmp:
		print()
		print()
		print('You can ignore all of the above messages.')
		print('The model will first be converted to a Hugging Face model due to memory constraints.')
		with yaspin(text='Saving Hugging Face model', spinner=SPINNER):
			pipe.save_pretrained(tmp, safe_serialization=True)

		del pipe

		tmp = Path(tmp)

		tokenizer = CLIPTokenizerFast.from_pretrained(tmp / 'tokenizer')
		tokenizer.backend_tokenizer.save(str(out_path / 'tokenizer.json'))
		model_config['tokenizer'] = {
			'type': 'CLIPTokenizer',
			'path': 'tokenizer.json',
			'model-max-length': tokenizer.model_max_length,
			'bos-token': tokenizer.bos_token_id,
			'eos-token': tokenizer.eos_token_id
		}

		if os.path.exists(tmp / 'feature_extractor'):
			feature_extractor = json.load(open(tmp / 'feature_extractor' / 'preprocessor_config.json'))
			size = feature_extractor['size']
			crop = feature_extractor['crop_size']
			model_config['feature-extractor'] = {
				'resample': feature_extractor['resample'],
				'size': size['shortest_edge'] if isinstance(size, dict) else size,
				'crop': [ crop['width'], crop['height'] ] if isinstance(crop, dict) else [ crop, crop ],
				'crop-center': feature_extractor['do_center_crop'],
				'rgb': feature_extractor['do_convert_rgb'],
				'normalize': feature_extractor['do_normalize'],
				'resize': feature_extractor['do_resize'],
				'image-mean': feature_extractor['image_mean'],
				'image-std': feature_extractor['image_std']
			}

		text_encoder_path, text_embeddings_path, num_tokens, text_hidden_size = convert_text_encoder(tmp)
		unet_path, unet_sample_size = convert_unet(tmp, num_tokens, text_hidden_size)
		vae_encoder_path, vae_decoder_path, vae_sample_size, vae_out_channels, vae_scale_factor = convert_vae(tmp, unet_sample_size)

		model_config['text-encoder'] = {
			'path': text_encoder_path.relative_to(out_path).as_posix(),
			'text-embeddings': {
				'path': text_embeddings_path.relative_to(out_path).as_posix()
			}
		}
		model_config['unet'] = { 'path': unet_path.relative_to(out_path).as_posix() }
		model_config['vae'] = {
			'encoder': vae_encoder_path.relative_to(out_path).as_posix(),
			'decoder': vae_decoder_path.relative_to(out_path).as_posix(),
			'scale_factor': vae_scale_factor
		}

		model_config['hashes'] = {
			'text-encoder': hashfile(text_encoder_path, hexdigest=True),
			'text-embeddings': hashfile(text_embeddings_path, hexdigest=True),
			'unet': hashfile(unet_path, hexdigest=True),
			'vae-encoder': hashfile(vae_encoder_path, hexdigest=True),
			'vae-decoder': hashfile(vae_decoder_path, hexdigest=True)
		}

		# TODO: safety checker

		with open(out_path / 'pyke-diffusers.toml', 'wb') as f:
			toml.dump(model_config, f)

cprint(f'✨ Your model is ready! {str(out_path)}')
