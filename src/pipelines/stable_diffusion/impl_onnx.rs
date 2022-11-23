#![allow(dead_code)]

use std::{cell::RefCell, path::PathBuf, sync::Arc};

use bitflags::bitflags;
use image::{DynamicImage, Rgb32FImage};
use ml2::onnx::{
	tensor::{FromArray, InputTensor, OrtOwnedTensor},
	Environment, Session, SessionBuilder
};
use ndarray::{concatenate, Array1, Array2, Array4, ArrayD, Axis, IxDyn};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::{StableDiffusionOptions, StableDiffusionTxt2ImgOptions};
use crate::{clip::CLIPStandardTokenizer, schedulers::DiffusionScheduler, Prompt, StableDiffusionCallback};

bitflags! {
	/// Helper to select which models to replace when using [`StableDiffusionPipeline::replace`].
	pub struct StableDiffusionReplaceFlags: u32 {
		/// Replace the UNet.
		const REPLACE_UNET = 1 << 0;
		/// Replace the variational autoencoder decoder.
		const REPLACE_VAE_DECODER = 1 << 1;
		/// Replace the text encoder.
		const REPLACE_TEXT_ENCODER = 1 << 2;
		/// Replace the tokenizer.
		const REPLACE_TOKENIZER = 1 << 3;
		/// Replace the variational autoencoder.
		const REPLACE_VAE_ENCODER = 1 << 4;
		/// Replace the safety checker.
		const REPLACE_SAFETY_CHECKER = 1 << 5;

		/// Replace all models.
		const REPLACE_ALL =
			Self::REPLACE_UNET.bits |
			Self::REPLACE_VAE_DECODER.bits |
			Self::REPLACE_TEXT_ENCODER.bits |
			Self::REPLACE_TOKENIZER.bits |
			Self::REPLACE_VAE_ENCODER.bits |
			Self::REPLACE_SAFETY_CHECKER.bits;
	}
}

/// A [Stable Diffusion](https://github.com/CompVis/stable-diffusion) pipeline.
pub struct StableDiffusionPipeline {
	environment: Arc<Environment>,
	options: StableDiffusionOptions,
	vae_decoder: RefCell<Session>,
	text_encoder: RefCell<Session>,
	tokenizer: CLIPStandardTokenizer,
	unet: RefCell<Session>,
	safety_checker: Option<RefCell<Session>>,
	feature_extractor: Option<RefCell<Session>>
}

impl StableDiffusionPipeline {
	/// Creates a new Stable Diffusion pipeline, loading models from `root`.
	///
	/// `environment` must be an [`ml2::onnx::Environment`]. Only one environment can be created per process.
	pub fn new(environment: &Arc<Environment>, root: impl Into<PathBuf>, options: &StableDiffusionOptions) -> anyhow::Result<Self> {
		let root: PathBuf = root.into();
		let vae_decoder = SessionBuilder::new(environment)?
			.with_execution_providers([options.devices.vae_decoder.clone().into()])?
			.with_model_from_file(root.join("vae_decoder.onnx"))?;
		let text_encoder = SessionBuilder::new(environment)?
			.with_execution_providers([options.devices.text_encoder.clone().into()])?
			.with_model_from_file(root.join("text_encoder.onnx"))?;
		let tokenizer = CLIPStandardTokenizer::new(root.join("tokenizer.json"))?;
		let unet = SessionBuilder::new(environment)?
			.with_execution_providers([options.devices.unet.clone().into()])?
			.with_model_from_file(root.join("unet.onnx"))?;
		let safety_checker = SessionBuilder::new(environment)?
			.with_execution_providers([options.devices.safety_checker.clone().into()])?
			.with_model_from_file(root.join("safety_checker.onnx"))
			.map(|s| Some(RefCell::new(s)))
			.unwrap_or(None);
		let feature_extractor = None;
		Ok(Self {
			environment: Arc::clone(environment),
			options: options.clone(),
			vae_decoder: RefCell::new(vae_decoder),
			text_encoder: RefCell::new(text_encoder),
			tokenizer,
			unet: RefCell::new(unet),
			safety_checker,
			feature_extractor
		})
	}

	/// Replace some or all models in this pipeline.
	///
	/// Most Stable Diffusion variants use the same VAE, text encoder, and safety checker as Stable Diffusion v1.4 &
	/// v1.5. If you know beforehand what models are the same as the currently loaded pipeline, you can prevent deleting
	/// and recreating an entire pipeline by modifying this pipeline in-place.
	///
	/// The VAE, text encoder, and safety checker for Stable Diffusion v1.5 and Waifu Diffusion v1.3 are confirmed to be
	/// identical to Stable Diffusion v1.4's models. If using models converted from HuggingFace, you can conveniently
	/// compare the SHA256 file hashes online by viewing the files in the model cards.
	///
	/// See [`StableDiffusionReplaceFlags`] for info on choosing which models to replace. In 99% of cases you'll just
	/// want to replace the unet only: `StableDiffusionReplaceFlags::REPLACE_UNET`
	///
	/// ```ignore
	/// let mut pipeline = StableDiffusionPipeline::new(&environment, "./sd1.4/", &StableDiffusionOptions::default())?;
	/// pipeline = pipeline.replace("./sd1.5/", StableDiffusionReplaceFlags::REPLACE_UNET)?;
	/// ```
	pub fn replace(mut self, new_root: impl Into<PathBuf>, flags: StableDiffusionReplaceFlags) -> anyhow::Result<Self> {
		let new_root = new_root.into();
		if flags.contains(StableDiffusionReplaceFlags::REPLACE_UNET) {
			// we need to drop the old model before allocating the new one so we have enough memory
			std::mem::drop(self.unet);
			self.unet = RefCell::new(
				SessionBuilder::new(&self.environment)?
					.with_execution_providers([self.options.devices.unet.clone().into()])?
					.with_model_from_file(new_root.join("unet.onnx"))?
			);
		}
		if flags.contains(StableDiffusionReplaceFlags::REPLACE_VAE_DECODER) {
			// we need to drop the old model before allocating the new one so we have enough memory
			std::mem::drop(self.vae_decoder);
			self.vae_decoder = RefCell::new(
				SessionBuilder::new(&self.environment)?
					.with_execution_providers([self.options.devices.vae_decoder.clone().into()])?
					.with_model_from_file(new_root.join("vae_decoder.onnx"))?
			);
		}
		if flags.contains(StableDiffusionReplaceFlags::REPLACE_TEXT_ENCODER) {
			// we need to drop the old model before allocating the new one so we have enough memory
			std::mem::drop(self.text_encoder);
			self.text_encoder = RefCell::new(
				SessionBuilder::new(&self.environment)?
					.with_execution_providers([self.options.devices.text_encoder.clone().into()])?
					.with_model_from_file(new_root.join("text_encoder.onnx"))?
			);
		}
		if flags.contains(StableDiffusionReplaceFlags::REPLACE_TOKENIZER) {
			std::mem::drop(self.tokenizer);
			self.tokenizer = CLIPStandardTokenizer::new(new_root.join("tokenizer.json"))?;
		}
		if flags.contains(StableDiffusionReplaceFlags::REPLACE_SAFETY_CHECKER) {
			// we need to drop the old model before allocating the new one so we have enough memory
			std::mem::drop(self.safety_checker);
			self.safety_checker = SessionBuilder::new(&self.environment)?
				.with_execution_providers([self.options.devices.safety_checker.clone().into()])?
				.with_model_from_file(new_root.join("safety_checker.onnx"))
				.map(|s| Some(RefCell::new(s)))
				.unwrap_or(None);
		}
		Ok(self)
	}

	/// Encodes the given prompt(s) into an array of text embeddings to be used as input to the UNet.
	pub fn encode_prompt(&self, prompt: Prompt, do_classifier_free_guidance: bool, negative_prompt: Option<&Prompt>) -> anyhow::Result<ArrayD<f32>> {
		if let Some(neg_prompt) = negative_prompt {
			assert_eq!(prompt.len(), neg_prompt.len());
		}

		let batch_size = prompt.len();
		let text_input_ids: Vec<Vec<i32>> = self
			.tokenizer
			.encode(prompt.0)?
			.iter()
			.map(|v| v.iter().map(|tok| *tok as i32).collect::<Vec<i32>>())
			.collect();
		for batch in &text_input_ids {
			if batch.len() > self.tokenizer.len() {
				anyhow::bail!("prompts over 77 tokens is not currently implemented");
			}
		}

		let mut text_encoder = self.text_encoder.borrow_mut();
		let text_input_ids: Vec<i32> = text_input_ids.into_iter().flatten().collect();
		let text_input_ids = Array2::from_shape_vec((batch_size, self.tokenizer.len()), text_input_ids)?.into_dyn();
		let text_embeddings = text_encoder.run(vec![InputTensor::from_array(text_input_ids)])?;

		let mut text_embeddings: ArrayD<f32> = text_embeddings[0].try_extract()?.view().to_owned();

		if do_classifier_free_guidance {
			let uncond_input: Vec<i32> = self
				.tokenizer
				.encode(negative_prompt.cloned().unwrap_or_else(|| vec![""; batch_size].into()).0)?
				.iter()
				.flat_map(|v| v.iter().map(|tok| *tok as i32).collect::<Vec<i32>>())
				.collect();
			let uncond_embeddings =
				text_encoder.run(vec![InputTensor::from_array(Array2::from_shape_vec((batch_size, self.tokenizer.len()), uncond_input)?.into_dyn())])?;
			let uncond_embeddings: ArrayD<f32> = uncond_embeddings[0].try_extract()?.view().to_owned();
			text_embeddings = concatenate![Axis(0), uncond_embeddings, text_embeddings];
		}

		Ok(text_embeddings)
	}

	/// Decodes UNet latents via the variational autoencoder into an array of [`image::DynamicImage`]s.
	pub fn decode_latents(&self, mut latents: Array4<f32>, options: &StableDiffusionTxt2ImgOptions) -> anyhow::Result<Vec<DynamicImage>> {
		latents = 1.0 / 0.18215 * latents;

		let mut vae_decoder = self.vae_decoder.borrow_mut();
		let latent_vae_input: ArrayD<f32> = latents.into_dyn();
		let mut images = Vec::new();
		for latent_chunk in latent_vae_input.axis_iter(Axis(0)) {
			let latent_chunk = latent_chunk.to_owned().insert_axis(Axis(0));
			let image = vae_decoder.run(vec![InputTensor::from_array(latent_chunk)])?;
			let image: OrtOwnedTensor<'_, f32, IxDyn> = image[0].try_extract()?;
			let f_image: Array4<f32> = image.view().to_owned().into_dimensionality()?;
			let f_image = f_image.permuted_axes([0, 2, 3, 1]).map(|f| (f / 2.0 + 0.5).clamp(0.0, 1.0));

			images.push(DynamicImage::ImageRgb32F(
				Rgb32FImage::from_raw(options.width, options.height, f_image.map(|f| f.clamp(0.0, 1.0)).into_iter().collect::<Vec<_>>())
					.ok_or_else(|| anyhow::anyhow!("failed to construct image"))?
			));
		}

		Ok(images)
	}

	/// Generates images from given text prompt(s). Returns a vector of [`image::DynamicImage`]s, using float32 buffers.
	/// In most cases, you'll want to convert it the images into RGB8 via `img.into_rgb8().`
	///
	/// `scheduler` must be a Stable Diffusion-compatible scheduler.
	///
	/// See [`StableDiffusionTxt2ImgOptions`] for configuration.
	pub fn txt2img<S: DiffusionScheduler>(
		&self,
		prompt: impl Into<Prompt>,
		scheduler: &mut S,
		options: &StableDiffusionTxt2ImgOptions
	) -> anyhow::Result<Vec<DynamicImage>> {
		let steps = options.steps;

		let seed = options.seed.unwrap_or_else(|| rand::thread_rng().gen::<u64>());
		let mut rng = StdRng::seed_from_u64(seed);

		if options.height % 8 != 0 || options.width % 8 != 0 {
			anyhow::bail!("`width` ({}) and `height` ({}) must be divisible by 8 for Stable Diffusion", options.width, options.height);
		}

		let prompt: Prompt = prompt.into();
		let batch_size = prompt.len();

		let do_classifier_free_guidance = options.guidance_scale > 1.0;
		let text_embeddings = self.encode_prompt(prompt, do_classifier_free_guidance, options.negative_prompt.as_ref())?;

		let latents_shape = (batch_size, 4_usize, (options.height / 8) as usize, (options.width / 8) as usize);
		let mut latents = Array4::<f32>::random_using(latents_shape, StandardNormal, &mut rng);

		scheduler.set_timesteps(steps);
		latents *= scheduler.init_noise_sigma();

		let mut unet = self.unet.borrow_mut();

		for (i, t) in scheduler.timesteps().to_owned().indexed_iter() {
			let latent_model_input = if do_classifier_free_guidance {
				concatenate![Axis(0), latents, latents]
			} else {
				latents.to_owned()
			};
			let latent_model_input = scheduler.scale_model_input(latent_model_input.view(), *t);

			let latent_model_input: ArrayD<f32> = latent_model_input.into_dyn();
			let timestep: ArrayD<f32> = Array1::from_iter([*t]).into_dyn();
			let encoder_hidden_states: ArrayD<f32> = text_embeddings.clone().into_dyn();

			let noise_pred = unet.run(vec![
				InputTensor::from_array(latent_model_input),
				InputTensor::from_array(timestep),
				InputTensor::from_array(encoder_hidden_states),
			])?;
			let noise_pred: OrtOwnedTensor<'_, f32, IxDyn> = noise_pred[0].try_extract()?;
			let noise_pred: Array4<f32> = noise_pred.view().to_owned().into_dimensionality()?;

			let mut noise_pred: Array4<f32> = noise_pred.clone();
			if do_classifier_free_guidance {
				let mut noise_pred_chunks = noise_pred.axis_iter(Axis(0));
				let (noise_pred_uncond, noise_pred_text) = (noise_pred_chunks.next().unwrap(), noise_pred_chunks.next().unwrap());
				let (noise_pred_uncond, noise_pred_text) = (noise_pred_uncond.insert_axis(Axis(0)).to_owned(), noise_pred_text.insert_axis(Axis(0)).to_owned());
				noise_pred = &noise_pred_uncond + options.guidance_scale * (noise_pred_text - &noise_pred_uncond);
			}

			let scheduler_output = scheduler.step(noise_pred.view(), *t, Some(i), latents.view(), &mut rng);
			latents = scheduler_output.prev_sample().to_owned();

			if let Some(callback) = options.callback.as_ref() {
				let keep_going = match callback {
					StableDiffusionCallback::Progress(every, callback) if i % every == 0 => callback(i, *t),
					StableDiffusionCallback::Latents(every, callback) if i % every == 0 => callback(i, *t, latents.clone()),
					StableDiffusionCallback::Decoded(every, callback) if i % every == 0 => callback(i, *t, self.decode_latents(latents.clone(), options)?),
					_ => true
				};
				if !keep_going {
					break;
				}
			}
		}

		self.decode_latents(latents, options)
	}
}
