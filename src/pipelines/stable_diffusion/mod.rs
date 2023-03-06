use std::fmt::Debug;

use image::DynamicImage;
use ndarray::Array4;

use crate::{DiffusionDeviceControl, Prompt};

pub(crate) mod lpw;

mod impl_main;
pub use self::impl_main::*;
mod impl_memory_optimized;
pub use self::impl_memory_optimized::*;

/// Options for the Stable Diffusion pipeline. This includes options like device control and long prompt weighting.
#[derive(Debug, Clone)]
pub struct StableDiffusionOptions {
	/// A [`DiffusionDeviceControl`] object, mapping what device to place each model on.
	pub devices: DiffusionDeviceControl,
	/// Enable or disable long prompt weighting.
	///
	/// Long prompt weighting enables prompts beyond the typical 77 token limit of the CLIP tokenizer, and allows for
	/// emphasizing or de-emphasizing certain parts of the prompt. The weighting syntax and values are the same as
	/// AUTOMATIC1111's implementation:
	///
	/// - `a (word)` - increase attention to `word` by a factor of 1.1
	/// - `a ((word))` - increase attention to `word` by a factor of 1.21 (= 1.1 * 1.1)
	/// - `a [word]` - decrease attention to `word` by a factor of 1.1
	/// - `a (word:1.5)` - increase attention to `word` by a factor of 1.5
	/// - `a (word:0.25)` - decrease attention to `word` by a factor of 4 (= 1 / 0.25)
	/// - `a \(word\)` - use literal `()` characters in prompt
	///
	/// With `()`, a weight can be specified like this: `(text:1.4)`. If the weight is not specified, it is assumed to
	/// be 1.1. Specifying weight only works with `()`, but not with `[]`.
	///
	/// If you want to use any of the literal `()[]` characters in the prompt, use the backslash to escape them:
	/// `anime_\(character\)`.
	///
	/// This method of weighting is slightly different to NovelAI's; NAI uses 1.05 as the multiplier and `{}` instead of
	/// `()`:
	/// - NAI `{word}` = PD `(word:1.05)`
	/// - NAI `{{word}}` = PD `(word:1.1025)`
	/// - NAI `[word]` = PD `(word:0.952)`
	/// - NAI `[[word]]` = PD `(word:0.907)`
	pub lpw: bool
}

impl Default for StableDiffusionOptions {
	fn default() -> Self {
		Self {
			devices: DiffusionDeviceControl::default(),
			lpw: true
		}
	}
}

/// Describes a function to be called on each step of the pipeline.
pub enum StableDiffusionCallback {
	/// A simple callback to be used for e.g. reporting progress updates.
	Progress {
		/// Describes how frequently to call this callback (3 = every 3 steps).
		frequency: usize,
		/// Function Parameters:
		/// - **`step`** (usize): The current step number.
		/// - **`timestep`** (f32): This step's timestep.
		cb: Box<dyn Fn(usize, f32) -> bool>
	},
	/// A callback to receive this step's latents.
	Latents {
		/// Describes how frequently to call this callback (3 = every 3 steps).
		frequency: usize,
		/// Function Parameters:
		/// - **`step`** (usize): The current step number.
		/// - **`timestep`** (f32): This step's timestep.
		/// - **`latents`** (`Array4<f32>`): Scheduler latent outputs for this step.
		cb: Box<dyn Fn(usize, f32, Array4<f32>) -> bool>
	},
	/// A callback to receive this step's fully decoded latents, to be used for e.g. showing image progress visually.
	/// This is very expensive, as it will execute the VAE decoder on each call. See
	/// [`StableDiffusionCallback::ApproximateDecoded`] for an approximated version.
	Decoded {
		/// Describes how frequently to call this callback (3 = every 3 steps).
		frequency: usize,
		/// Function Parameters:
		/// - **`step`** (usize): The current step number.
		/// - **`timestep`** (f32): This step's timestep.
		/// - **`image`** (`Vec<DynamicImage>`): Vector of decoded images for this step.
		cb: Box<dyn Fn(usize, f32, Vec<DynamicImage>) -> bool>
	},
	/// A callback to receive this step's approximately decoded latents, to be used for e.g. showing image progress
	/// visually. This is lower quality than [`StableDiffusionCallback::Decoded`] but much faster.
	///
	/// Approximated images may be noisy and colors will not be accurate (especially if using a fine-tuned VAE).
	ApproximateDecoded {
		/// Describes how frequently to call this callback (3 = every 3 steps).
		frequency: usize,
		/// Function Parameters:
		/// - **`step`** (usize): The current step number.
		/// - **`timestep`** (f32): This step's timestep.
		/// - **`image`** (`Vec<DynamicImage>`): Vector of approximated decoded images for this step.
		cb: Box<dyn Fn(usize, f32, Vec<DynamicImage>) -> bool>
	}
}

impl Debug for StableDiffusionCallback {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("<StableDiffusionCallback>")
	}
}

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
			negative_prompt: None,
			callback: None
		}
	}
}
