use std::{fs, path::PathBuf, sync::Arc};

use image::{DynamicImage, Rgb32FImage};
use ndarray::{concatenate, Array1, Array2, Array4, ArrayD, Axis, IxDyn};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use num_traits::ToPrimitive;
use ort::{
	tensor::{FromArray, InputTensor, OrtOwnedTensor},
	Environment, OrtResult, Session, SessionBuilder
};
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::{StableDiffusionOptions, StableDiffusionTxt2ImgOptions};
use crate::{
	clip::CLIPStandardTokenizer,
	config::{DiffusionFramework, DiffusionPipeline, StableDiffusionConfig, TokenizerConfig},
	schedulers::DiffusionScheduler,
	Prompt, StableDiffusionCallback
};

/// A [Stable Diffusion](https://github.com/CompVis/stable-diffusion) pipeline.
///
/// ```ignore
/// use std::sync::Arc;
///
/// use pyke_diffusers::{
/// 	EulerDiscreteScheduler, OrtEnvironment, SchedulerOptimizedDefaults, StableDiffusionOptions, StableDiffusionPipeline,
/// 	StableDiffusionTxt2ImgOptions
/// };
///
/// let environment = Arc::new(OrtEnvironment::builder().build()?);
/// let mut scheduler = EulerDiscreteScheduler::stable_diffusion_v1_optimized_default()?;
/// let pipeline =
/// 	StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;
///
/// let imgs = pipeline.txt2img("photo of a red fox", &mut scheduler, StableDiffusionTxt2ImgOptions::default())?;
/// ```
pub struct StableDiffusionPipeline {
	environment: Arc<Environment>,
	options: StableDiffusionOptions,
	config: StableDiffusionConfig,
	vae_encoder: Option<Session>,
	vae_decoder: Session,
	text_encoder: Option<Session>,
	tokenizer: Option<CLIPStandardTokenizer>,
	unet: Session,
	safety_checker: Option<Session>,
	#[allow(dead_code)]
	feature_extractor: Option<()>
}

impl StableDiffusionPipeline {
	/// Creates a new Stable Diffusion pipeline, loading models from `root`.
	///
	/// ```ignore
	/// let pipeline =
	/// 	StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;
	/// ```
	pub fn new(environment: &Arc<Environment>, root: impl Into<PathBuf>, options: StableDiffusionOptions) -> anyhow::Result<Self> {
		let root: PathBuf = root.into();
		let config: DiffusionPipeline = serde_json::from_reader(fs::read(root.join("diffusers.json"))?.as_slice())?;
		let config: StableDiffusionConfig = match config {
			DiffusionPipeline::StableDiffusion { framework, inner } => {
				assert_eq!(framework, DiffusionFramework::Onnx);
				inner
			}
			#[allow(unreachable_patterns)]
			_ => anyhow::bail!("not a stable diffusion pipeline")
		};

		let tokenizer = config
			.tokenizer
			.as_ref()
			.map(|tokenizer| match tokenizer {
				TokenizerConfig::CLIPTokenizer {
					path,
					model_max_length,
					bos_token,
					eos_token
				} => CLIPStandardTokenizer::new(root.join(path.clone()), !options.lpw, *model_max_length, *bos_token, *eos_token),
				#[allow(unreachable_patterns)]
				_ => anyhow::bail!("not a clip tokenizer")
			})
			.transpose()?;

		let text_encoder = config
			.text_encoder
			.as_ref()
			.map(|text_encoder| -> OrtResult<Session> {
				SessionBuilder::new(environment)?
					.with_execution_providers([options.devices.text_encoder.clone().into()])?
					.with_model_from_file(root.join(text_encoder.path.clone()))
			})
			.transpose()?;

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
			tokenizer,
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
	/// ```ignore
	/// let mut pipeline = StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;
	/// pipeline = pipeline.replace("./waifu-diffusion-v1-3/", None)?;
	/// ```
	pub fn replace(mut self, new_root: impl Into<PathBuf>, options: Option<StableDiffusionOptions>) -> anyhow::Result<Self> {
		let new_root: PathBuf = new_root.into();
		let new_config: DiffusionPipeline = serde_json::from_reader(fs::read(new_root.join("diffusers.json"))?.as_slice())?;
		let new_config: StableDiffusionConfig = match new_config {
			DiffusionPipeline::StableDiffusion { framework, inner } => {
				assert_eq!(framework, DiffusionFramework::Onnx);
				inner
			}
			#[allow(unreachable_patterns)]
			_ => anyhow::bail!("not a stable diffusion pipeline!")
		};

		let options = options.unwrap_or_else(|| self.options.clone());

		if self.config.hashes.unet != new_config.hashes.unet {
			std::mem::drop(self.unet);
			self.unet = SessionBuilder::new(&self.environment)?
				.with_execution_providers([self.options.devices.unet.clone().into()])?
				.with_model_from_file(new_root.join(new_config.unet.path.clone()))?;
		}
		if self.config.hashes.text_encoder != new_config.hashes.text_encoder {
			std::mem::drop(self.text_encoder);
			self.text_encoder = new_config
				.text_encoder
				.as_ref()
				.map(|text_encoder| -> OrtResult<Session> {
					SessionBuilder::new(&self.environment)?
						.with_execution_providers([options.devices.text_encoder.clone().into()])?
						.with_model_from_file(new_root.join(text_encoder.path.clone()))
				})
				.transpose()?
		}
		if self.config.hashes.vae_decoder != new_config.hashes.vae_decoder {
			std::mem::drop(self.vae_decoder);
			self.vae_decoder = SessionBuilder::new(&self.environment)?
				.with_execution_providers([options.devices.vae_decoder.clone().into()])?
				.with_model_from_file(new_root.join(new_config.vae.decoder.clone()))?;
		}
		if self.config.hashes.vae_encoder != new_config.hashes.vae_encoder {
			std::mem::drop(self.vae_encoder);
			self.vae_encoder = new_config
				.vae
				.encoder
				.as_ref()
				.map(|path| -> OrtResult<Session> {
					SessionBuilder::new(&self.environment)?
						.with_execution_providers([options.devices.vae_encoder.clone().into()])?
						.with_model_from_file(new_root.join(path))
				})
				.transpose()?;
		}
		if self.config.hashes.safety_checker != new_config.hashes.safety_checker {
			std::mem::drop(self.safety_checker);
			self.safety_checker = new_config
				.safety_checker
				.as_ref()
				.map(|safety_checker| -> OrtResult<Session> {
					SessionBuilder::new(&self.environment)?
						.with_execution_providers([options.devices.safety_checker.clone().into()])?
						.with_model_from_file(new_root.join(safety_checker.path.clone()))
				})
				.transpose()?;
		}

		self.options.clone_from(&options);
		self.config = new_config;

		Ok(self)
	}

	/// Encodes the given prompt(s) into an array of text embeddings to be used as input to the UNet.
	pub fn encode_prompt(&self, prompt: Prompt, do_classifier_free_guidance: bool, negative_prompt: Option<&Prompt>) -> anyhow::Result<ArrayD<f32>> {
		let tokenizer = self
			.tokenizer
			.as_ref()
			.ok_or_else(|| anyhow::anyhow!("tokenizer required for text-based generation"))?;

		if let Some(neg_prompt) = negative_prompt {
			assert_eq!(prompt.len(), neg_prompt.len());
		}

		let text_encoder = self
			.text_encoder
			.as_ref()
			.ok_or_else(|| anyhow::anyhow!("text encoder required for text-based generation"))?;

		let batch_size = prompt.len();
		let text_embeddings = if self.options.lpw {
			let embeddings = crate::pipelines::lpw::get_weighted_text_embeddings(
				tokenizer,
				text_encoder,
				prompt,
				if do_classifier_free_guidance {
					negative_prompt.cloned().or_else(|| Some(vec![""; batch_size].into()))
				} else {
					negative_prompt.cloned()
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
		} else {
			let text_input_ids: Vec<Vec<i32>> = tokenizer
				.encode(prompt.0)?
				.iter()
				.map(|v| v.iter().map(|tok| *tok as i32).collect::<Vec<i32>>())
				.collect();

			let text_input_ids: Vec<i32> = text_input_ids.into_iter().flatten().collect();
			let text_input_ids = Array2::from_shape_vec((batch_size, tokenizer.len()), text_input_ids)?.into_dyn();
			let text_embeddings = text_encoder.run(vec![InputTensor::from_array(text_input_ids)])?;
			let mut text_embeddings: ArrayD<f32> = text_embeddings[0].try_extract()?.view().to_owned();

			if do_classifier_free_guidance {
				let uncond_input: Vec<i32> = tokenizer
					.encode(negative_prompt.cloned().unwrap_or_else(|| vec![""; batch_size].into()).0)?
					.iter()
					.flat_map(|v| v.iter().map(|tok| *tok as i32).collect::<Vec<i32>>())
					.collect();
				let uncond_embeddings =
					text_encoder.run(vec![InputTensor::from_array(Array2::from_shape_vec((batch_size, tokenizer.len()), uncond_input)?.into_dyn())])?;
				let uncond_embeddings: ArrayD<f32> = uncond_embeddings[0].try_extract()?.view().to_owned();
				text_embeddings = concatenate![Axis(0), uncond_embeddings, text_embeddings];
			}

			text_embeddings
		};

		Ok(text_embeddings)
	}

	fn to_image(&self, width: u32, height: u32, arr: &Array4<f32>) -> anyhow::Result<DynamicImage> {
		Ok(DynamicImage::ImageRgb32F(
			Rgb32FImage::from_raw(width, height, arr.map(|f| f.clamp(0.0, 1.0)).into_iter().collect::<Vec<_>>())
				.ok_or_else(|| anyhow::anyhow!("failed to construct image"))?
		))
	}

	/// Decodes UNet latents via the variational autoencoder into an array of [`image::DynamicImage`]s.
	pub fn decode_latents(&self, mut latents: Array4<f32>, options: &StableDiffusionTxt2ImgOptions) -> anyhow::Result<Vec<DynamicImage>> {
		latents = 1.0 / 0.18215 * latents;

		let latent_vae_input: ArrayD<f32> = latents.into_dyn();
		let mut images = Vec::new();
		for latent_chunk in latent_vae_input.axis_iter(Axis(0)) {
			let latent_chunk = latent_chunk.to_owned().insert_axis(Axis(0));
			let image = self.vae_decoder.run(vec![InputTensor::from_array(latent_chunk)])?;
			let image: OrtOwnedTensor<'_, f32, IxDyn> = image[0].try_extract()?;
			let f_image: Array4<f32> = image.view().to_owned().into_dimensionality()?;
			let f_image = f_image.permuted_axes([0, 2, 3, 1]).map(|f| (f / 2.0 + 0.5).clamp(0.0, 1.0));

			let image = self.to_image(options.width, options.height, &f_image)?;
			images.push(image);
		}

		Ok(images)
	}

	/// Generates images from given text prompt(s). Returns a vector of [`image::DynamicImage`]s, using float32 buffers.
	/// In most cases, you'll want to convert the images into RGB8 via `img.clone().into_rgb8().`
	///
	/// `scheduler` must be a Stable Diffusion-compatible scheduler.
	///
	/// See [`StableDiffusionTxt2ImgOptions`] for additional configuration.
	///
	/// # Examples
	///
	/// Simple text-to-image:
	/// ```ignore
	/// let pipeline =
	/// 	StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", &StableDiffusionOptions::default())?;
	///
	/// let imgs = pipeline.txt2img("photo of a red fox", &mut scheduler, &StableDiffusionTxt2ImgOptions::default())?;
	/// imgs[0].clone().into_rgb8().save("result.png")?;
	/// ```
	pub fn txt2img<S: DiffusionScheduler>(
		&self,
		prompt: impl Into<Prompt>,
		scheduler: &mut S,
		options: StableDiffusionTxt2ImgOptions
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

		let timesteps = scheduler.timesteps().to_owned();
		let num_warmup_steps = timesteps.len() - options.steps * S::order();

		for (i, t) in timesteps.to_owned().indexed_iter() {
			let latent_model_input = if do_classifier_free_guidance {
				concatenate![Axis(0), latents, latents]
			} else {
				latents.to_owned()
			};
			let latent_model_input = scheduler.scale_model_input(latent_model_input.view(), *t);

			let latent_model_input: ArrayD<f32> = latent_model_input.into_dyn();
			let timestep: ArrayD<f32> = Array1::from_iter([t.to_f32().unwrap()]).into_dyn();
			let encoder_hidden_states: ArrayD<f32> = text_embeddings.clone().into_dyn();

			let noise_pred = self.unet.run(vec![
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

			let scheduler_output = scheduler.step(noise_pred.view(), *t, latents.view(), &mut rng);
			latents = scheduler_output.prev_sample().to_owned();

			if let Some(callback) = options.callback.as_ref() {
				if i == timesteps.len() - 1 || ((i + 1) > num_warmup_steps && (i + 1) % S::order() == 0) {
					let keep_going = match callback {
						StableDiffusionCallback::Progress { frequency, cb } if i % frequency == 0 => cb(i, t.to_f32().unwrap()),
						StableDiffusionCallback::Latents { frequency, cb } if i % frequency == 0 => cb(i, t.to_f32().unwrap(), latents.clone()),
						StableDiffusionCallback::Decoded { frequency, cb } if i % frequency == 0 => {
							cb(i, t.to_f32().unwrap(), self.decode_latents(latents.clone(), &options)?)
						}
						_ => true
					};
					if !keep_going {
						break;
					}
				}
			}
		}

		self.decode_latents(latents, &options)
	}
}
