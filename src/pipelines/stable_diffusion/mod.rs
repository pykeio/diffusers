use cfg_if::cfg_if;
use image::DynamicImage;
use ndarray::Array4;

use crate::{DiffusionDeviceControl, Prompt};

/// Options for the Stable Diffusion pipeline.
#[derive(Default, Clone)]
pub struct StableDiffusionOptions {
	/// A [`DiffusionDeviceControl`] object, mapping what devices to place each model on.
	pub devices: DiffusionDeviceControl
}

/// Describes a function to be called on each step of the pipeline.
pub enum StableDiffusionCallback {
	/// A simple callback to be used for e.g. reporting progress updates.
	///
	/// The first value describes how frequently to call this callback (3 = every 3 steps).
	///
	/// Function Parameters:
	/// - **`step`** (usize): The current step number.
	/// - **`timestep`** (f32): This step's timestep.
	Progress(usize, Box<dyn Fn(usize, f32) -> bool>),
	/// A callback to receive this step's latents.
	///
	/// The first value describes how frequently to call this callback (3 = every 3 steps).
	///
	/// Parameters:
	/// - **`step`** (usize): The current step number.
	/// - **`timestep`** (f32): This step's timestep.
	/// - **`latents`** (`Array4<f32>`): Scheduler latent outputs for this step.
	Latents(usize, Box<dyn Fn(usize, f32, Array4<f32>) -> bool>),
	/// A callback to receive this step's decoded latents, to be used for e.g. showing image progress visually.
	///
	/// The first value describes how frequently to call this callback (3 = every 3 steps).
	///
	/// Parameters:
	/// - **`step`** (usize): The current step number.
	/// - **`timestep`** (f32): This step's timestep.
	/// - **`image`** (`Vec<DynamicImage>`): Vector of decoded images for this step.
	Decoded(usize, Box<dyn Fn(usize, f32, Vec<DynamicImage>) -> bool>)
}

/// Options for the Stable Diffusion text-to-image pipeline.
pub struct StableDiffusionTxt2ImgOptions {
	/// The height of the image. Must be divisible by 8.
	pub height: u32,
	/// The width of the image. Must be divisible by 8.
	pub width: u32,
	/// The 'guidance scale' of classifier-free guidance. Higher numbers mean the model will match the prompt(s) more
	/// strictly.
	pub guidance_scale: f32,
	/// The number of steps to take to generate the image. More steps typically yields higher quality images.
	pub steps: u16,
	/// An optional seed to use when first generating noise. The same seed will produce the same image.
	pub seed: Option<u64>,
	/// Optional prompt(s) describing what the model should **not** generate in classifier-free guidance. Typically used
	/// to produce safer outputs, e.g. `negative_prompt: Some("gore, violence, blood".into())`. Must have the same
	/// number of prompts as the 'positive' input.
	pub negative_prompt: Option<Prompt>,
	/// eta parameter of the DDIM scheduler.
	pub eta: Option<f32>,
	/// An optional callback to call every `n` steps in the generation process. Can be used to log or display progress.
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
			eta: None,
			callback: None
		}
	}
}

cfg_if! {
	if #[cfg(feature = "onnx")] {
		mod impl_onnx;
		pub use self::impl_onnx::*;
	}
}
