use image::imageops::FilterType;
use image::{DynamicImage, Rgb32FImage};
use ndarray::{Array4, Ix};

use crate::{pipelines::stable_diffusion::StableDiffusionTxt2ImgOptions, Prompt, StableDiffusionCallback};

/// The image preprocessing method to on images that mismatch size.
#[derive(Debug)]
pub enum ImagePreprocessing {
	/// The image is resized to the target size.
	Resize,
	/// Resize and crop the image to fit in the target size.
	CropFill
}

/// Options for the Stable Diffusion image-to-image pipeline.
#[derive(Debug)]
pub struct StableDiffusionImg2ImgOptions {
	pub reference_image: Array4<f32>,
	pub noise_strength: f32,
	pub preprocessing: ImagePreprocessing,
	pub text_config: StableDiffusionTxt2ImgOptions
}

impl Default for StableDiffusionImg2ImgOptions {
	fn default() -> Self {
		Self {
			reference_image: Array4::default((1, 1, 1, 1)),
			noise_strength: 0.6,
			preprocessing: ImagePreprocessing::CropFill,
			text_config: StableDiffusionTxt2ImgOptions::default()
		}
	}
}

impl StableDiffusionImg2ImgOptions {
	/// Get the size of each reference image.
	pub fn get_size(&self) -> (u32, u32) {
		(self.text_config.width, self.text_config.height)
	}

	/// Get the dimensions of reference images.
	pub fn get_dimensions(&self) -> (Ix, Ix, Ix, Ix) {
		self.reference_image.dim()
	}

	/// Set the size of the image. **Size will be rounded to a multiple of 8.**
	pub fn with_size(self, width: u32, height: u32) -> Self {
		self.with_width(width).with_height(height)
	}

	/// Set the width of the image. **Width will be rounded to a multiple of 8.**
	#[inline]
	pub fn with_width(mut self, width: u32) -> Self {
		self.text_config.width = (width / 8).max(1) * 8;
		self.drop_reference_image();
		self
	}

	/// Set the height of the image. **Height will be rounded to a multiple of 8.**
	#[inline]
	pub fn with_height(mut self, height: u32) -> Self {
		self.text_config.height = (height / 8).max(1) * 8;
		self.drop_reference_image();
		self
	}

	#[inline(always)]
	fn drop_reference_image(&mut self) {
		self.reference_image = Array4::default((1, 1, 1, 1));
	}

	/// The number of steps to take to generate the image. Typically, more steps yields higher quality images up to a
	/// certain point, depending on the scheduler. For instance, the quality of images generated using
	/// [`DPMSolverMultistepScheduler`](crate::schedulers::DPMSolverMultistepScheduler) does not meaningfully improve
	/// past about 40 steps.
	pub fn with_steps(mut self, steps: usize) -> Self {
		self.text_config.steps = steps;
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
		self.text_config.positive_prompt = positive_prompt.into();
		self
	}

	/// Set the prompt(s) describing what the model should **not** generate in classifier-free guidance. The model will
	/// be guided *away* from the concepts described in the negative prompt. The negative prompt must have the same
	/// number of prompts as the 'positive' prompt input.
	///
	/// Negative prompts support long prompt weighting (LPW). LPW enables prompts beyond the typical 77 token limit of
	/// the CLIP tokenizer, and allows for emphasizing or de-emphasizing certain parts of the prompt. See
	/// [`StableDiffusionImg2ImgOptions::with_prompt`] for more detailed documentation on LPW.
	///
	/// Note that using LPW to emphasize certain parts in the *negative* prompt will have the opposite effect. For
	/// example, `[word]` means the model will only be guided slightly away from `word`.
	pub fn with_negative_prompt<P>(mut self, negative_prompt: P) -> Self
	where
		P: Into<Prompt>
	{
		self.text_config.negative_prompt = Some(negative_prompt.into());
		self
	}

	/// Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
	/// will be used as a starting point, adding more noise to it the larger the `strength`. The number of
	/// denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
	/// be maximum and the denoising process will run for the full number of iterations specified in
	/// `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
	pub fn with_noise_strength(mut self, noise_strength: f32) -> Self {
		self.noise_strength = noise_strength.max(0.0).min(1.0);
		self
	}

	/// Set a seed to use when adding noise. The same seed with the same image, prompt, and parameters will produce
	/// the same image. If `None`, a random seed will be generated.
	///
	/// Seeds are not interchangable between schedulers, and **a seed from Hugging Face diffusers or AUTOMATIC1111's
	/// web UI will *not* generate the same image** in pyke Diffusers.
	pub fn with_seed(mut self, seed: u64) -> Self {
		self.text_config.seed = Some(seed);
		self
	}

	/// Use a random seed, so that each run generates a (slightly) different image.
	pub fn with_random_seed(mut self) -> Self {
		self.text_config.seed = None;
		self
	}

	/// The 'guidance scale' for classifier-free guidance. A lower guidance scale gives the model more freedom, but the
	/// output may not match the prompt. A higher guidance scale mean the model will match the prompt(s) more strictly,
	/// but may introduce artifacts; `7.5` is a good balance.
	pub fn with_guidance_scale(mut self, guidance_scale: f32) -> Self {
		self.text_config.guidance_scale = guidance_scale;
		self
	}

	/// ETA noise seed delta (ENSD). The scheduler will be given an RNG seeded with `seed + ensd`.
	pub fn with_eta_noise_seed_delta(mut self, ensd: u64) -> Self {
		self.text_config.ensd = ensd;
		self
	}

	/// Set a reference image to for generating
	pub fn with_image(mut self, image: &DynamicImage, batch: usize) -> Self {
		// whc -> nchw
		let image = self.img_norm(image);
		let shape = [batch, 3, self.text_config.height as usize, self.text_config.width as usize];
		self.reference_image = Array4::from_shape_fn(shape, |(_, c, h, w)| {
			let pixel = image.get_pixel(w as u32, h as u32);
			match c {
				0 => pixel.0[0],
				1 => pixel.0[1],
				2 => pixel.0[2],
				_ => unreachable!()
			}
		});
		self
	}

	/// Set reference images to for generating, batch size must be equal to `images.len()`
	pub fn with_images(mut self, images: &[DynamicImage]) -> Self {
		// nwhc -> nchw
		let images = images.iter().map(|image| self.img_norm(image)).collect::<Vec<_>>();
		let shape = [images.len(), 3, self.text_config.height as usize, self.text_config.width as usize];
		self.reference_image = Array4::from_shape_fn(shape, |(n, c, h, w)| {
			let pixel = images[n].get_pixel(w as u32, h as u32);
			match c {
				0 => pixel.0[0],
				1 => pixel.0[1],
				2 => pixel.0[2],
				_ => unreachable!()
			}
		});
		self
	}

	fn img_norm(&self, image: &DynamicImage) -> Rgb32FImage {
		let img = match self.preprocessing {
			ImagePreprocessing::Resize => image.resize_exact(self.text_config.width, self.text_config.height, FilterType::Lanczos3),
			ImagePreprocessing::CropFill => image.resize_to_fill(self.text_config.width, self.text_config.height, FilterType::Lanczos3)
		};
		// normalize to [0, 1]
		img.to_rgb32f()
	}

	#[doc = include_str!("_doc/callback-progress.md")]
	pub fn callback_progress<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32) -> bool + 'static
	{
		self.text_config.callback = Some(StableDiffusionCallback::Progress { frequency, cb: Box::new(callback) });
		self
	}

	#[doc = include_str!("_doc/callback-latents.md")]
	pub fn callback_latents<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Array4<f32>) -> bool + 'static
	{
		self.text_config.callback = Some(StableDiffusionCallback::Latents { frequency, cb: Box::new(callback) });
		self
	}

	#[doc = include_str!("_doc/callback-decode-image.md")]
	pub fn callback_decoded<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Vec<DynamicImage>) -> bool + 'static
	{
		self.text_config.callback = Some(StableDiffusionCallback::Decoded { frequency, cb: Box::new(callback) });
		self
	}

	#[doc = include_str!("_doc/callback-approximate-image.md")]
	pub fn callback_approximate<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Vec<DynamicImage>) -> bool + 'static
	{
		self.text_config.callback = Some(StableDiffusionCallback::ApproximateDecoded { frequency, cb: Box::new(callback) });
		self
	}
}
