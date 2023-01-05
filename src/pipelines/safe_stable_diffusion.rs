use std::{path::PathBuf, sync::Arc};

use image::DynamicImage;
use ndarray::{Array4, ArrayD};
use ort::Environment;

use super::stable_diffusion::StableDiffusionPipeline;
pub use super::stable_diffusion::{StableDiffusionOptions, StableDiffusionTxt2ImgOptions};
use crate::{schedulers::DiffusionScheduler, Prompt};

/// A [Stable Diffusion](https://github.com/CompVis/stable-diffusion) pipeline designed to guide generation away from
/// harmful or unsafe imagery.
///
/// ```ignore
/// use std::sync::Arc;
///
/// use pyke_diffusers::{
/// 	EulerDiscreteScheduler, OrtEnvironment, SafeStableDiffusionPipeline, SchedulerOptimizedDefaults, StableDiffusionOptions,
/// 	StableDiffusionTxt2ImgOptions
/// };
///
/// let environment = Arc::new(OrtEnvironment::builder().build()?);
/// let mut scheduler = EulerDiscreteScheduler::stable_diffusion_v1_optimized_default()?;
/// let pipeline =
/// 	SafeStableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", &StableDiffusionOptions::default())?;
///
/// let imgs = pipeline.txt2img("photo of a red fox", &mut scheduler, StableDiffusionTxt2ImgOptions::default())?;
/// ```
pub struct SafeStableDiffusionPipeline {
	pipeline: StableDiffusionPipeline,
	safety_concept: Prompt
}

impl SafeStableDiffusionPipeline {
	/// Creates a new Stable Diffusion pipeline, loading models from `root`.
	///
	/// ```ignore
	/// let pipeline =
	/// 	SafeStableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", &StableDiffusionOptions::default())?;
	/// ```
	pub fn new(environment: &Arc<Environment>, root: impl Into<PathBuf>, options: StableDiffusionOptions) -> anyhow::Result<Self> {
		let safety_concept = "an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty".into();
		Ok(Self {
			pipeline: StableDiffusionPipeline::new(environment, root, options)?,
			safety_concept
		})
	}

	/// Replace some or all models in this pipeline. This function will only replace models that are different to the
	/// models currently loaded, which can save a good amount of time on slower hardware.
	///
	/// An additional [`StableDiffusionOptions`] parameter can be used to move models to another device.
	///
	/// ```ignore
	/// let mut pipeline = SafeStableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", &StableDiffusionOptions::default())?;
	/// pipeline = pipeline.replace("./waifu-diffusion-v1-3/", None)?;
	/// ```
	pub fn replace(mut self, new_root: impl Into<PathBuf>, options: Option<StableDiffusionOptions>) -> anyhow::Result<Self> {
		self.pipeline = self.pipeline.replace(new_root, options)?;
		Ok(self)
	}

	/// Encodes the given prompt(s) into an array of text embeddings to be used as input to the UNet.
	pub fn encode_prompt(&self, prompt: Prompt, do_classifier_free_guidance: bool, negative_prompt: Option<&Prompt>) -> anyhow::Result<ArrayD<f32>> {
		self.pipeline.encode_prompt(prompt, do_classifier_free_guidance, negative_prompt)
	}

	/// Decodes UNet latents via the variational autoencoder into an array of [`image::DynamicImage`]s.
	pub fn decode_latents(&self, latents: Array4<f32>, options: &StableDiffusionTxt2ImgOptions) -> anyhow::Result<Vec<DynamicImage>> {
		self.pipeline.decode_latents(latents, options)
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
	/// 	SafeStableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;
	///
	/// let imgs = pipeline.txt2img("photo of a red fox", &mut scheduler, StableDiffusionTxt2ImgOptions::default())?;
	/// imgs[0].clone().into_rgb8().save("result.png")?;
	/// ```
	pub fn txt2img<S: DiffusionScheduler>(
		&self,
		prompt: impl Into<Prompt>,
		scheduler: &mut S,
		mut options: StableDiffusionTxt2ImgOptions
	) -> anyhow::Result<Vec<DynamicImage>> {
		options.negative_prompt = Some(self.safety_concept.clone());
		self.pipeline.txt2img(prompt, scheduler, options)
	}
}
