use image::DynamicImage;
use ndarray::{concatenate, Array1, Array4, ArrayD, Axis, IxDyn};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use num_traits::{ToPrimitive, Zero};
use ort::tensor::{FromArray, InputTensor, OrtOwnedTensor, TensorElementDataType};
use ort::{OrtError, OrtResult};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
	/// Prompt(s) describing what the model should generate in classifier-free guidance. Typically used to produce
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
			steps: 50,
			seed: None,
			positive_prompt: Default::default(),
			negative_prompt: None,
			callback: None
		}
	}
}

impl StableDiffusionTxt2ImgOptions {
	pub fn with_size(mut self, height: u32, width: u32) -> OrtResult<Self> {
		self.with_width(width)?.with_height(height)
	}
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
	/// Creates a new `StableDiffusionTxt2ImgOptions` with the default options.
	///
	/// # Arguments
	///
	/// * `frequency`: The frequency at which to call the callback, in steps.
	/// * `callback`: The callback to call every `frequency` steps.
	pub fn with_progress_callback<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32) -> bool
	{
		self.callback = Some(StableDiffusionCallback::Progress { frequency, cb: Box::new(callback) });
		self
	}
}
