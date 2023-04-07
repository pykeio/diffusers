use image::DynamicImage;
use ndarray::{concatenate, s, Array1, Array4, ArrayD, Axis, IxDyn};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use num_traits::ToPrimitive;
use ort::tensor::{FromArray, InputTensor, OrtOwnedTensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::{DiffusionScheduler, Prompt, StableDiffusionCallback, StableDiffusionPipeline, StableDiffusionTxt2ImgOptions};

impl Default for StableDiffusionTxt2ImgOptions {
	fn default() -> Self {
		Self {
			height: 512,
			width: 512,
			guidance_scale: 7.5,
			steps: 25,
			seed: None,
			positive_prompt: Prompt::default(),
			negative_prompt: None,
			callback: None
		}
	}
}

// builder for options
impl StableDiffusionTxt2ImgOptions {
	/// Set the size of the image. **Size will be rounded to a multiple of 8.**
	pub fn with_size(self, width: u32, height: u32) -> Self {
		self.with_width(width).with_height(height)
	}
	/// Set the width of the image. **Width will be rounded to a multiple of 8.**
	#[inline]
	pub fn with_width(mut self, width: u32) -> Self {
		self.width = (width / 8).max(1) * 8;
		self
	}
	/// Set the height of the image. **Height will be rounded to a multiple of 8.**
	#[inline]
	pub fn with_height(mut self, height: u32) -> Self {
		self.height = (height / 8).max(1) * 8;
		self
	}
	/// The number of steps to take to generate the image. More steps typically yields higher quality images.
	pub fn with_steps(mut self, steps: usize) -> Self {
		self.steps = steps;
		self
	}
	/// Set the prompt(s) to use when generating the image.
	pub fn with_prompts<P>(mut self, positive_prompt: P, negative_prompt: Option<P>) -> Self
	where
		P: Into<Prompt>
	{
		self.positive_prompt = positive_prompt.into();
		self.negative_prompt = negative_prompt.map(|p| p.into());
		self
	}
	/// Set the seed to use when first generating noise.
	pub fn with_seed(mut self, seed: u64) -> Self {
		self.seed = Some(seed);
		self
	}
	/// Use a random seed, so that each run generates a different image.
	pub fn with_random_seed(mut self) -> Self {
		self.seed = None;
		self
	}
	/// The 'guidance scale' for classifier-free guidance. A lower guidance scale gives the model more freedom, but the
	/// output may not match the prompt. A higher guidance scale mean the model will match the prompt(s) more strictly,
	/// but may introduce artifacts; `7.5` is a good balance.
	pub fn with_guidance_scale(mut self, guidance_scale: f32) -> Self {
		self.guidance_scale = guidance_scale;
		self
	}
}

// builder for callbacks
impl StableDiffusionTxt2ImgOptions {
	#[doc = include_str!("_doc/callback-progress.md")]
	pub fn callback_progress<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32) -> bool + 'static
	{
		self.callback = Some(StableDiffusionCallback::Progress { frequency, cb: Box::new(callback) });
		self
	}
	#[doc = include_str!("_doc/callback-latents.md")]
	pub fn callback_latents<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Array4<f32>) -> bool + 'static
	{
		self.callback = Some(StableDiffusionCallback::Latents { frequency, cb: Box::new(callback) });
		self
	}
	#[doc = include_str!("_doc/callback-decode-image.md")]
	pub fn callback_decoded<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Vec<DynamicImage>) -> bool + 'static
	{
		self.callback = Some(StableDiffusionCallback::Decoded { frequency, cb: Box::new(callback) });
		self
	}
	#[doc = include_str!("_doc/callback-approximate-image.md")]
	pub fn callback_approximate<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Vec<DynamicImage>) -> bool + 'static
	{
		self.callback = Some(StableDiffusionCallback::ApproximateDecoded { frequency, cb: Box::new(callback) });
		self
	}
}

impl StableDiffusionTxt2ImgOptions {
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
	///
	/// ```no_run
	/// # fn main() -> anyhow::Result<()> {
	/// # use pyke_diffusers::{StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionOptions, StableDiffusionTxt2ImgOptions, OrtEnvironment};
	/// # let environment = OrtEnvironment::default().into_arc();
	/// # let mut scheduler = EulerDiscreteScheduler::default();
	/// let pipeline =
	/// 	StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;
	///
	/// let mut imgs = StableDiffusionTxt2ImgOptions::default().with_prompts("photo of a red fox", None).run(&pipeline, &mut scheduler)?;
	/// imgs.remove(0).into_rgb8().save("result.png")?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn run<S: DiffusionScheduler>(&self, session: &StableDiffusionPipeline, scheduler: &mut S) -> anyhow::Result<Vec<DynamicImage>> {
		let steps = self.steps;
		let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen::<u64>());
		let mut rng = StdRng::seed_from_u64(seed);

		if self.height % 8 != 0 || self.width % 8 != 0 {
			anyhow::bail!("`width` ({}) and `height` ({}) must be divisible by 8 for Stable Diffusion", self.width, self.height);
		}

		let prompt = self.positive_prompt.clone();
		let batch_size = prompt.len();

		let do_classifier_free_guidance = self.guidance_scale > 1.0;
		let text_embeddings = session.encode_prompt(prompt, do_classifier_free_guidance, self.negative_prompt.as_ref())?;

		let latents_shape = (batch_size, 4_usize, (self.height / 8) as usize, (self.width / 8) as usize);
		let mut latents = Array4::<f32>::random_using(latents_shape, StandardNormal, &mut rng);

		scheduler.set_timesteps(steps);
		latents *= scheduler.init_noise_sigma();

		let timesteps = scheduler.timesteps().to_owned();
		let num_warmup_steps = timesteps.len() - self.steps * S::order();

		for (i, t) in timesteps.indexed_iter() {
			let latent_model_input = if do_classifier_free_guidance {
				concatenate![Axis(0), latents, latents]
			} else {
				latents.clone()
			};
			let latent_model_input = scheduler.scale_model_input(latent_model_input.view(), *t);
			let latent_model_input: ArrayD<f32> = latent_model_input.into_dyn();
			let timestep: ArrayD<f32> = Array1::from_iter([t.to_f32().unwrap()]).into_dyn();
			let encoder_hidden_states: ArrayD<f32> = text_embeddings.clone().into_dyn();

			let noise_pred = session.unet.run(vec![
				InputTensor::from_array(latent_model_input),
				InputTensor::from_array(timestep),
				InputTensor::from_array(encoder_hidden_states),
			])?;
			let noise_pred: OrtOwnedTensor<'_, f32, IxDyn> = noise_pred[0].try_extract()?;
			let noise_pred: Array4<f32> = noise_pred.view().to_owned().into_dimensionality()?;

			let mut noise_pred: Array4<f32> = noise_pred.clone();
			if do_classifier_free_guidance {
				assert!(noise_pred.shape()[0] % 2 == 0);
				let split_len = (noise_pred.shape()[0] / 2) as isize;
				let noise_pred_uncond = noise_pred.slice(s![..split_len, .., .., ..]);
				let noise_pred_text = noise_pred.slice(s![split_len.., .., .., ..]);
				noise_pred = &noise_pred_uncond + self.guidance_scale * (&noise_pred_text - &noise_pred_uncond);
			}

			let scheduler_output = scheduler.step(noise_pred.view(), *t, latents.view(), &mut rng);
			latents = scheduler_output.prev_sample;

			if let Some(callback) = self.callback.as_ref() {
				if i == timesteps.len() - 1 || ((i + 1) > num_warmup_steps && (i + 1) % S::order() == 0) {
					let keep_going = match callback {
						StableDiffusionCallback::Progress { frequency, cb } if i % frequency == 0 => cb(i, t.to_f32().unwrap()),
						StableDiffusionCallback::Latents { frequency, cb } if i % frequency == 0 => cb(i, t.to_f32().unwrap(), latents.clone()),
						StableDiffusionCallback::Decoded { frequency, cb } if i != 0 && i % frequency == 0 => {
							cb(i, t.to_f32().unwrap(), session.decode_latents(latents.view())?)
						}
						StableDiffusionCallback::ApproximateDecoded { frequency, cb } if i != 0 && i % frequency == 0 => {
							cb(i, t.to_f32().unwrap(), session.approximate_decode_latents(latents.view())?)
						}
						_ => true
					};
					if !keep_going {
						break;
					}
				}
			}
		}

		session.decode_latents(latents.view())
	}
}
