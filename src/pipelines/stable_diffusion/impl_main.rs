// Copyright 2022-2023 pyke.io
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::path::Path;
use std::{fs, path::PathBuf, sync::Arc};

use image::{DynamicImage, Rgb32FImage};
use ndarray::{concatenate, Array2, Array4, ArrayD, ArrayView4, Axis, IxDyn};
use ndarray_einsum_beta::einsum;
use ort::{
	tensor::{FromArray, InputTensor, OrtOwnedTensor},
	Environment, OrtResult, Session, SessionBuilder
};

use super::{StableDiffusionOptions, StableDiffusionTxt2ImgOptions};
use crate::text_embeddings::TextEmbeddings;
use crate::{
	clip::CLIPStandardTokenizer,
	config::{DiffusionFramework, DiffusionPipeline, StableDiffusionConfig, TokenizerConfig},
	schedulers::DiffusionScheduler,
	Prompt
};

/// A [Stable Diffusion](https://github.com/CompVis/stable-diffusion) pipeline.
///
/// ```
/// # fn main() -> anyhow::Result<()> {
/// use pyke_diffusers::{
/// 	EulerDiscreteScheduler, OrtEnvironment, SchedulerOptimizedDefaults, StableDiffusionOptions,
/// 	StableDiffusionPipeline, StableDiffusionTxt2ImgOptions
/// };
///
/// let environment = OrtEnvironment::default().into_arc();
/// let mut scheduler = EulerDiscreteScheduler::stable_diffusion_v1_optimized_default()?;
/// let pipeline =
/// 	StableDiffusionPipeline::new(&environment, "tests/stable-diffusion", StableDiffusionOptions::default())?;
///
/// let imgs = StableDiffusionTxt2ImgOptions::default()
/// 	.with_prompts("photo of a red fox", None)
/// 	.run(&pipeline, &mut scheduler)?;
/// # Ok(())
/// # }
/// ```
pub struct StableDiffusionPipeline {
	environment: Arc<Environment>,
	options: StableDiffusionOptions,
	config: StableDiffusionConfig,
	vae_encoder: Option<Session>,
	vae_decoder: Session,
	text_encoder: Session,
	text_embeddings: TextEmbeddings,
	pub(crate) unet: Session,
	safety_checker: Option<Session>,
	#[allow(dead_code)]
	feature_extractor: Option<()>
}

impl StableDiffusionPipeline {
	/// Creates a new Stable Diffusion pipeline, loading models from `root`.
	///
	/// ```
	/// # fn main() -> anyhow::Result<()> {
	/// # use pyke_diffusers::{StableDiffusionPipeline, StableDiffusionOptions, OrtEnvironment};
	/// # let environment = OrtEnvironment::default().into_arc();
	/// let pipeline =
	/// 	StableDiffusionPipeline::new(&environment, "tests/stable-diffusion", StableDiffusionOptions::default())?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new(environment: &Arc<Environment>, root: impl Into<PathBuf>, options: StableDiffusionOptions) -> anyhow::Result<Self> {
		let root: PathBuf = root.into();
		let config: DiffusionPipeline = toml::from_str(&fs::read_to_string(root.join("pyke-diffusers.toml"))?)?;
		let config: StableDiffusionConfig = match config {
			DiffusionPipeline::StableDiffusion { framework, inner } => {
				match framework {
					DiffusionFramework::Orte { .. } => (),
					_ => panic!("bad framework")
				}
				inner
			}
			#[allow(unreachable_patterns)]
			_ => anyhow::bail!("not a stable diffusion pipeline")
		};

		let tokenizer = match &config.tokenizer {
			TokenizerConfig::CLIPTokenizer {
				path,
				model_max_length,
				bos_token,
				eos_token
			} => CLIPStandardTokenizer::new(root.join(path.clone()), !options.lpw, *model_max_length, *bos_token, *eos_token)?,
			#[allow(unreachable_patterns)]
			_ => anyhow::bail!("not a clip tokenizer")
		};
		let text_embeddings = TextEmbeddings::from_file(root.join(&config.text_encoder.text_embeddings.as_ref().unwrap().path), tokenizer)?;

		let text_encoder = SessionBuilder::new(environment)?
			.with_execution_providers([options.devices.text_encoder.clone().into()])?
			.with_model_from_file(root.join(config.text_encoder.path.clone()))?;

		let vae_encoder = config
			.vae
			.encoder
			.as_ref()
			.map(|path| -> OrtResult<Session> {
				SessionBuilder::new(environment)?
					.with_execution_providers([options.devices.vae_encoder.clone().into()])?
					.with_model_from_file(root.join(path))
			})
			.transpose()?;

		let vae_decoder = SessionBuilder::new(environment)?
			.with_execution_providers([options.devices.vae_decoder.clone().into()])?
			.with_model_from_file(root.join(config.vae.decoder.clone()))?;

		let unet = SessionBuilder::new(environment)?
			.with_execution_providers([options.devices.unet.clone().into()])?
			.with_model_from_file(root.join(config.unet.path.clone()))?;

		let safety_checker = config
			.safety_checker
			.as_ref()
			.map(|safety_checker| -> OrtResult<Session> {
				SessionBuilder::new(environment)?
					.with_execution_providers([options.devices.safety_checker.clone().into()])?
					.with_model_from_file(root.join(safety_checker.path.clone()))
			})
			.transpose()?;

		Ok(Self {
			environment: Arc::clone(environment),
			options,
			config,
			vae_encoder,
			vae_decoder,
			text_encoder,
			text_embeddings,
			unet,
			safety_checker,
			feature_extractor: None
		})
	}

	/// Replace some or all models in this pipeline. This function will only replace models that are different to the
	/// models currently loaded, which can save a good amount of time on slower hardware.
	///
	/// An additional [`StableDiffusionOptions`] parameter can be used to move models to another device.
	///
	/// ```no_run
	/// # fn main() -> anyhow::Result<()> {
	/// # use pyke_diffusers::{StableDiffusionPipeline, StableDiffusionOptions, OrtEnvironment};
	/// # let environment = OrtEnvironment::default().into_arc();
	/// let mut pipeline =
	/// 	StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;
	/// pipeline = pipeline.replace("./waifu-diffusion-v1-3/", None)?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn replace(mut self, new_root: impl Into<PathBuf>, options: Option<StableDiffusionOptions>) -> anyhow::Result<Self> {
		let new_root: PathBuf = new_root.into();
		let new_config: DiffusionPipeline = toml::from_str(&fs::read_to_string(new_root.join("pyke-diffusers.toml"))?)?;
		let new_config: StableDiffusionConfig = match new_config {
			DiffusionPipeline::StableDiffusion { framework, inner } => {
				match framework {
					DiffusionFramework::Orte { .. } => (),
					_ => panic!("bad framework")
				}
				inner
			}
			#[allow(unreachable_patterns)]
			_ => anyhow::bail!("not a stable diffusion pipeline!")
		};

		let options = options.unwrap_or_else(|| self.options.clone());

		if self.config.hashes.unet != new_config.hashes.unet {
			let path = new_root.join(new_config.unet.path.clone());
			self.replace_unet(path)?
		}
		if self.config.hashes.text_encoder != new_config.hashes.text_encoder {
			let path = new_root.join(new_config.text_encoder.path.clone());
			self.replace_text_encoder(path)?
		}
		if self.config.hashes.vae_decoder != new_config.hashes.vae_decoder || self.config.hashes.vae_encoder != new_config.hashes.vae_encoder {
			let decoder = new_root.join(new_config.vae.decoder.clone());
			let encoder = new_config.vae.encoder.as_ref().map(|s| new_root.join(s));
			self.replace_vae(decoder, encoder)?
		}
		if self.config.hashes.safety_checker != new_config.hashes.safety_checker {
			let path = new_config.safety_checker.as_ref().map(|s| new_root.join(&s.path));
			self.replace_safety_checker(path)?
		}

		let tokenizer = match &new_config.tokenizer {
			TokenizerConfig::CLIPTokenizer {
				path,
				model_max_length,
				bos_token,
				eos_token
			} => CLIPStandardTokenizer::new(new_root.join(path.clone()), !options.lpw, *model_max_length, *bos_token, *eos_token)?,
			#[allow(unreachable_patterns)]
			_ => anyhow::bail!("not a clip tokenizer")
		};
		self.text_embeddings = TextEmbeddings::from_file(new_root.join(&new_config.text_encoder.text_embeddings.as_ref().unwrap().path), tokenizer)?;

		self.options.clone_from(&options);
		self.config = new_config;

		Ok(self)
	}

	/// Replace unet model at runtime, ensuring that the model is using the same config as before.
	///
	/// # Arguments
	///
	/// * `path`: Path to the new unet model
	///
	/// # Examples
	///
	/// load raw stable diffusion pipeline and replace anything unet model
	///
	/// ```no_run
	/// # fn main() -> anyhow::Result<()> {
	/// # use pyke_diffusers::{OrtEnvironment, StableDiffusionOptions, StableDiffusionPipeline};
	/// let environment = OrtEnvironment::default().into_arc();
	/// let mut pipeline =
	/// 	StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;
	/// pipeline.replace_unet("./anything/unet.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn replace_unet<P: AsRef<Path>>(&mut self, path: P) -> OrtResult<()> {
		self.unet = SessionBuilder::new(&self.environment)?
			.with_execution_providers([self.options.devices.unet.clone().into()])?
			.with_model_from_file(path)?;
		Ok(())
	}
	/// Replace text encode model at runtime, ensuring that the model is using the same config as before.
	pub fn replace_text_encoder<P: AsRef<Path>>(&mut self, path: P) -> OrtResult<()> {
		self.text_encoder = SessionBuilder::new(&self.environment)?
			.with_execution_providers([self.options.devices.text_encoder.clone().into()])?
			.with_model_from_file(path)?;
		Ok(())
	}

	/// Replace vae model at runtime, ensuring that the model is using the same config as before.
	///
	/// # Arguments
	///
	/// * `decoder`: Path to the new vae decoder model, this is required
	/// * `encoder`: Path to the new vae encoder model, this is optional
	///
	/// # Examples
	///
	/// load raw stable diffusion pipeline and replace with anything vae model.
	///
	/// ```no_run
	/// # fn main() -> anyhow::Result<()> {
	/// # use pyke_diffusers::{StableDiffusionOptions, StableDiffusionPipeline, OrtEnvironment};
	/// let environment = OrtEnvironment::default().into_arc();
	/// let mut pipeline =
	/// 	StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;
	/// pipeline.replace_vae("./anything/vae-decoder.onnx", Some("./anything/vae-encoder.onnx"))?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn replace_vae<D, E>(&mut self, decoder: D, encoder: Option<E>) -> OrtResult<()>
	where
		E: AsRef<Path>,
		D: AsRef<Path>
	{
		self.vae_decoder = SessionBuilder::new(&self.environment)?
			.with_execution_providers([self.options.devices.vae_decoder.clone().into()])?
			.with_model_from_file(decoder.as_ref())?;
		// unable to use ? in map, so use match here
		self.vae_encoder = match encoder {
			Some(s) => Some(
				SessionBuilder::new(&self.environment)?
					.with_execution_providers([self.options.devices.vae_encoder.clone().into()])?
					.with_model_from_file(s)?
			),
			None => None
		};
		Ok(())
	}
	/// Replace safety checker at runtime, ensuring that the model is using the same config as before.
	pub fn replace_safety_checker<P: AsRef<Path>>(&mut self, path: Option<P>) -> OrtResult<()> {
		self.safety_checker = match path {
			Some(s) => Some(
				SessionBuilder::new(&self.environment)?
					.with_execution_providers([self.options.devices.safety_checker.clone().into()])?
					.with_model_from_file(s)?
			),
			None => None
		};
		Ok(())
	}

	/// Encodes the given prompt(s) into an array of text embeddings to be used as input to the UNet.
	pub fn encode_prompt(&self, prompt: Prompt, do_classifier_free_guidance: bool, negative_prompt: Option<&Prompt>) -> anyhow::Result<ArrayD<f32>> {
		let batch_size = prompt.len();
		let negative_prompt = if let Some(negative_prompt) = negative_prompt {
			if batch_size > 1 && negative_prompt.len() == 1 {
				Some(Prompt::from(vec![negative_prompt[0].clone(); batch_size]))
			} else {
				assert_eq!(batch_size, negative_prompt.len());
				Some(negative_prompt.to_owned())
			}
		} else {
			None
		};

		let text_embeddings = {
			let embeddings = crate::pipelines::lpw::get_weighted_text_embeddings(
				&self.text_embeddings,
				&self.text_encoder,
				prompt,
				if do_classifier_free_guidance {
					negative_prompt.or_else(|| Some(Prompt::default_batched(batch_size)))
				} else {
					negative_prompt
				},
				3,
				true
			)?;
			let mut text_embeddings = embeddings.0;
			if do_classifier_free_guidance {
				if let Some(uncond_embeddings) = embeddings.1 {
					text_embeddings = concatenate![Axis(0), uncond_embeddings, text_embeddings];
				}
			}
			text_embeddings.into_dyn()
		};

		Ok(text_embeddings)
	}

	fn to_image(&self, width: u32, height: u32, arr: &Array4<f32>) -> anyhow::Result<DynamicImage> {
		Ok(DynamicImage::ImageRgb32F(
			Rgb32FImage::from_raw(width, height, arr.map(|f| f.clamp(0.0, 1.0)).into_iter().collect::<Vec<_>>())
				.ok_or_else(|| anyhow::anyhow!("failed to construct image"))?
		))
	}

	/// Decodes UNet latents via a cheap approximation into an array of [`image::DynamicImage`]s.
	pub fn approximate_decode_latents(&self, latents: ArrayView4<'_, f32>) -> anyhow::Result<Vec<DynamicImage>> {
		let coefs = Array2::from_shape_vec((4, 3), vec![0.298, 0.207, 0.208, 0.187, 0.286, 0.173, -0.158, 0.189, 0.264, -0.184, -0.271, -0.473])?;
		let approx = einsum("blxy,lr->bxyr", &[&latents, &coefs]).expect("einsum error");
		let mut images = Vec::new();
		for approx_chunk in approx.axis_iter(Axis(0)) {
			let approx_chunk = approx_chunk.insert_axis(Axis(0)).into_dimensionality()?.to_owned();
			let image = self.to_image(approx_chunk.shape()[1] as _, approx_chunk.shape()[2] as _, &approx_chunk)?;
			images.push(image);
		}
		Ok(images)
	}

	/// Decodes UNet latents via the variational autoencoder into an array of [`image::DynamicImage`]s.
	pub fn decode_latents(&self, latents: ArrayView4<'_, f32>) -> anyhow::Result<Vec<DynamicImage>> {
		let latents = 1.0 / 0.18215 * &latents;

		let mut images = Vec::new();
		for latent_chunk in latents.axis_iter(Axis(0)) {
			let latent_chunk = latent_chunk.into_dyn().insert_axis(Axis(0));
			let image = self.vae_decoder.run(vec![InputTensor::from_array(latent_chunk.to_owned())])?;
			let image: OrtOwnedTensor<'_, f32, IxDyn> = image[0].try_extract()?;
			let f_image: Array4<f32> = image.view().to_owned().into_dimensionality()?;
			let f_image = f_image.permuted_axes([0, 2, 3, 1]) / 2.0 + 0.5;

			let image = self.to_image(f_image.shape()[1] as _, f_image.shape()[2] as _, &f_image)?;
			images.push(image);
		}

		Ok(images)
	}

	/// > **Note**: this is deprecated, use `run()` in [`StableDiffusionTxt2ImgOptions`] instead.
	///
	/// Generates images from given text prompt(s). Returns a vector of [`image::DynamicImage`]s, using float32 buffers.
	/// In most cases, you'll want to convert the images into RGB8 via `img.into_rgb8().`
	///
	/// `scheduler` must be a Stable Diffusion-compatible scheduler.
	///
	/// See [`StableDiffusionTxt2ImgOptions`] for additional configuration.
	///
	/// # Examples
	///
	/// Simple text-to-image:
	/// ```
	/// # fn main() -> anyhow::Result<()> {
	/// # use pyke_diffusers::{StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionOptions, StableDiffusionTxt2ImgOptions, OrtEnvironment};
	/// # let environment = OrtEnvironment::default().into_arc();
	/// # let mut scheduler = EulerDiscreteScheduler::default();
	/// let pipeline =
	/// 	StableDiffusionPipeline::new(&environment, "tests/stable-diffusion", StableDiffusionOptions::default())?;
	///
	/// let mut imgs = StableDiffusionTxt2ImgOptions::default()
	/// 	.with_prompts("photo of a red fox", None)
	/// 	.run(&pipeline, &mut scheduler)?;
	/// imgs.remove(0).into_rgb8().save("result.png")?;
	/// # Ok(())
	/// # }
	/// ```
	#[deprecated(note = "use builder pattern with `StableDiffusionTxt2ImgOptions::run` instead")]
	pub fn txt2img<S: DiffusionScheduler>(
		&self,
		prompt: impl Into<Prompt>,
		scheduler: &mut S,
		options: StableDiffusionTxt2ImgOptions
	) -> anyhow::Result<Vec<DynamicImage>> {
		let mut new_options = options;
		new_options.positive_prompt = prompt.into();
		new_options.run(self, scheduler)
	}
}
