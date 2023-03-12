use crate::{ImagePreprocessing, StableDiffusionImg2ImgOptions, StableDiffusionTxt2ImgOptions};

impl Default for StableDiffusionImg2ImgOptions {
	fn default() -> Self {
		Self {
			reference_image: Default::default(),
			preprocessing: ImagePreprocessing::CropFill,
			target_height: 512,
			target_width: 512,
			text_config: StableDiffusionTxt2ImgOptions::default()
		}
	}
}

// builder for options
impl StableDiffusionTxt2ImgOptions {
	///  Set the size of the image. **Must be divisible by 8.**
	pub fn with_size(self, height: u32, width: u32) -> OrtResult<Self> {
		self.with_width(width)?.with_height(height)
	}
	///  Set the width of the image. **Must be divisible by 8.**
	pub fn with_width(mut self, width: u32) -> OrtResult<Self> {
		if width % 8 != 0 || width.is_zero() {
			Err(OrtError::DataTypeMismatch {
				actual: TensorElementDataType::Float32,
				requested: TensorElementDataType::Float32
			})?
		}
		self.width = width;
		Ok(self)
	}
	///  Set the height of the image. **Must be divisible by 8.**
	pub fn with_height(mut self, height: u32) -> OrtResult<Self> {
		if height % 8 != 0 || height.is_zero() {
			Err(OrtError::DataTypeMismatch {
				actual: TensorElementDataType::Float32,
				requested: TensorElementDataType::Float32
			})?
		}
		self.height = height;
		Ok(self)
	}
	/// Set the number of steps to take to generate the image. More steps typically yields higher quality images.
	pub fn with_steps(mut self, steps: usize) -> Self {
		self.steps = steps;
		self
	}
	/// Set the prompt(s) to use when generating the image. Typically used to produce classifier-free guidance.
	pub fn with_prompts<P, N>(mut self, positive_prompt: P, negative_prompt: Option<N>) -> Self
	where
		P: Into<Prompt>,
		N: Into<Prompt>
	{
		self.positive_prompt = positive_prompt.into();
		self.negative_prompt = negative_prompt.map(|p| p.into());
		self
	}
	/// Set the seed to use when first generating noise. The same seed with the same scheduler, prompt, & guidance
	pub fn with_seed(mut self, seed: u64) -> Self {
		self.seed = Some(seed);
		self
	}
	/// Set the scale of the guidance. Higher values will result in more guidance, lower values will result in less.
	pub fn with_guidance_scale(mut self, guidance_scale: f32) -> Self {
		self.guidance_scale = guidance_scale;
		self
	}
}

// builder for callbacks
impl StableDiffusionTxt2ImgOptions {
	#[doc = include_str!("callback-progress.md")]
	pub fn callback_progress<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32) -> bool + 'static
	{
		self.callback = Some(StableDiffusionCallback::Progress { frequency, cb: Box::new(callback) });
		self
	}
	#[doc = include_str!("callback-latents.md")]
	pub fn callback_latents<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Array4<f32>) -> bool + 'static
	{
		self.callback = Some(StableDiffusionCallback::Latents { frequency, cb: Box::new(callback) });
		self
	}
	#[doc = include_str!("callback-decode-image.md")]
	pub fn callback_decoded<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Vec<DynamicImage>) -> bool + 'static
	{
		self.callback = Some(StableDiffusionCallback::Decoded { frequency, cb: Box::new(callback) });
		self
	}
	#[doc = include_str!("callback-approximate-image.md")]
	pub fn callback_approximate<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Vec<DynamicImage>) -> bool + 'static
	{
		self.callback = Some(StableDiffusionCallback::ApproximateDecoded { frequency, cb: Box::new(callback) });
		self
	}
}
