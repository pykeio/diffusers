use image::DynamicImage;
use ndarray::{concatenate, s, Array1, Array4, ArrayD, Axis, IxDyn};
use ndarray_rand::{
	rand::{self, rngs::StdRng, Rng, SeedableRng},
	rand_distr::StandardNormal,
	RandomExt
};
use num_traits::ToPrimitive;
use ort::tensor::OrtOwnedTensor;

use crate::{DiffusionScheduler, Prompt, StableDiffusionCallback, StableDiffusionPipeline};

/// Options for the Stable Diffusion text-to-image pipeline.
#[derive(Debug)]
pub struct StableDiffusionTxt2ImgOptions {
	/// The height of the image. **Must be divisible by 8.**
	/// Note that higher resolution images require more VRAM.
	pub height: u32,
	/// The width of the image. **Must be divisible by 8.**
	/// Note that higher resolution images require more VRAM.
	pub width: u32,
	/// The 'guidance scale' for classifier-free guidance. A lower guidance scale gives the model more freedom, but the
	/// output may not match the prompt. A higher guidance scale mean the model will match the prompt(s) more strictly,
	/// but may introduce artifacts; `7.5` is a good balance.
	pub guidance_scale: f32,
	/// The number of steps to take to generate the image. More steps typically yields higher quality images.
	pub steps: usize,
	/// An optional seed to use when first generating noise. The same seed with the same scheduler, prompt, & guidance
	/// scale will produce the same image. If `None`, a random seed will be generated.
	///
	/// Seeds are not interchangable between schedulers, and **a seed from Hugging Face diffusers or AUTOMATIC1111's
	/// web UI will *not* generate the same image** in pyke Diffusers.
	pub seed: Option<u64>,
	/// ETA noise seed delta (ENSD). The scheduler will be given an RNG seeded with `seed + ensd`.
	pub ensd: u64,
	/// Prompt(s) describing what the model should generate in classifier-free guidance.
	pub positive_prompt: Prompt,
	/// Optional prompt(s) describing what the model should **not** generate in classifier-free guidance. Typically used
	/// to produce safe outputs, e.g. `negative_prompt: Some("gore, violence, blood".into())`. Must have the same
	/// number of prompts as the 'positive' prompt input.
	pub negative_prompt: Option<Prompt>,
	/// An optional callback to call every `n` steps in the generation process. Can be used to log or display progress,
	/// see [`StableDiffusionCallback`] for more details.
	pub callback: Option<StableDiffusionCallback>
}

impl Default for StableDiffusionTxt2ImgOptions {
	fn default() -> Self {
		Self {
			height: 512,
			width: 512,
			guidance_scale: 7.5,
			steps: 25,
			seed: None,
			ensd: 0,
			positive_prompt: Prompt::default(),
			negative_prompt: None,
			callback: None
		}
	}
}

impl StableDiffusionTxt2ImgOptions {
	/// Set the size of the image. **Size will be rounded to a multiple of 8**. Note that higher resolution images
	/// require more (V)RAM to generate.
	pub fn with_size(self, width: u32, height: u32) -> Self {
		self.with_width(width).with_height(height)
	}

	/// Set the width of the image. **Width will be rounded to a multiple of 8**. Note that higher resolution images
	/// require more (V)RAM to generate.
	#[inline]
	pub fn with_width(mut self, width: u32) -> Self {
		self.width = (width / 8).max(1) * 8;
		self
	}

	/// Set the height of the image. **Height will be rounded to a multiple of 8**. Note that higher resolution images
	/// require more (V)RAM to generate.
	#[inline]
	pub fn with_height(mut self, height: u32) -> Self {
		self.height = (height / 8).max(1) * 8;
		self
	}

	/// The number of steps to take to generate the image. Typically, more steps yields higher quality images up to a
	/// certain point, depending on the scheduler. For instance, the quality of images generated using
	/// [`DPMSolverMultistepScheduler`](crate::schedulers::DPMSolverMultistepScheduler) does not meaningfully improve
	/// past about 40 steps.
	pub fn with_steps(mut self, steps: usize) -> Self {
		self.steps = steps;
		self
	}

	/// Set the prompt(s) describing what the model should generate in classifier-free guidance.
	///
	/// Prompts support long prompt weighting (LPW). LPW enables prompts beyond the typical 77 token limit of the CLIP
	/// tokenizer, and allows for emphasizing or de-emphasizing certain parts of the prompt. The weighting syntax and
	/// values are the same as AUTOMATIC1111's implementation:
	///
	/// - `a (word)` - *increase* attention to `word` by a factor of 1.1
	/// - `a ((word))` - *increase* attention to `word` by a factor of 1.21 (= 1.1 * 1.1)
	/// - `a [word]` - *decrease* attention to `word` by a factor of 1.1
	/// - `a (word:1.5)` - *increase* attention to `word` by a factor of 1.5
	/// - `a (word:0.25)` - *decrease* attention to `word` by a factor of 4 (= 1 / 0.25)
	/// - `a \(word\)` - use literal `()` characters in prompt; `word` would not be emphasized in this case
	///
	/// With `()`, a weight can be specified, like `(text:1.4)` - this will increase attention to `text` by a factor of
	/// 1.4. If the weight is not specified, it is assumed to be 1.1. Specifying weight only works with `()`, but not
	/// with `[]`. Emphasis weights should be between 0 and 1.4; going too far above 1.4 typically seems to generate
	/// Lovecraftian horrors. Additionally, a weight of 0 will not act the same as if the concept were in the negative
	/// prompt (it will just introduce a token with 0 embeddings, which will likely just confuse the model).
	///
	/// If you want to use any of the literal `()[]` characters in the prompt, use the backslash to escape them:
	/// `anime_\(character\)`.
	///
	/// pyke Diffusers' method of weighting is slightly different to NovelAI's; NAI uses ~1.05 as the default emphasis
	/// multiplier and `{}` instead of `()`:
	/// - NAI `{word}` = pyke `(word:1.05)`
	/// - NAI `{{word}}` = pyke `(word:1.1025)`
	/// - NAI `[word]` = pyke `(word:0.952)`
	/// - NAI `[[word]]` = pyke `(word:0.907)`
	pub fn with_prompt<P>(mut self, positive_prompt: P) -> Self
	where
		P: Into<Prompt>
	{
		self.positive_prompt = positive_prompt.into();
		self
	}

	/// Set the prompt(s) describing what the model should **not** generate in classifier-free guidance. The model will
	/// be guided *away* from the concepts described in the negative prompt. The negative prompt must have the same
	/// number of prompts as the 'positive' prompt input.
	///
	/// Negative prompts are typically used to produce non-harmful outputs, e.g. `.with_negative_prompt("gore, violence,
	/// blood")`. For this case, we provide a "safety concept" prompt that you can use in your negative prompt to
	/// produce safe outputs for base (non-finetuned) models; see [`StableDiffusionPipeline::SAFETY_CONCEPT`].
	///
	/// Negative prompts support long prompt weighting (LPW). LPW enables prompts beyond the typical 77 token limit of
	/// the CLIP tokenizer, and allows for emphasizing or de-emphasizing certain parts of the prompt. See
	/// [`StableDiffusionTxt2ImgOptions::with_prompt`] for more detailed documentation on LPW.
	///
	/// Note that using LPW to emphasize certain parts in the *negative* prompt will have the opposite effect. For
	/// example, `[word]` means the model will only be guided slightly away from `word`.
	pub fn with_negative_prompt<P>(mut self, negative_prompt: P) -> Self
	where
		P: Into<Prompt>
	{
		self.negative_prompt = Some(negative_prompt.into());
		self
	}

	/// Set a seed to use when first generating noise. The same seed with the same prompt and parameters will produce
	/// the same image. If `None`, a random seed will be generated.
	///
	/// Seeds are not interchangable between schedulers, and **a seed from Hugging Face diffusers or AUTOMATIC1111's
	/// web UI will *not* generate the same image** in pyke Diffusers.
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

	/// ETA noise seed delta (ENSD). The scheduler will be given an RNG seeded with `seed + ensd`.
	pub fn with_eta_noise_seed_delta(mut self, ensd: u64) -> Self {
		self.ensd = ensd;
		self
	}

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
	/// let mut imgs = StableDiffusionTxt2ImgOptions::default().with_prompt("photo of a red fox").run(&pipeline, &mut scheduler)?;
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

		let mut scheduler_rng = StdRng::seed_from_u64(seed + 31337);

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

			let noise_pred = session
				.unet
				.run(&[latent_model_input.into(), timestep.into(), encoder_hidden_states.into()])?;
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

			let scheduler_output = scheduler.step(noise_pred.view(), *t, latents.view(), &mut scheduler_rng);
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
