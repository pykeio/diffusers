// Copyright 2022-2023 pyke.io
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::fmt::Debug;

use image::DynamicImage;
use ndarray::Array4;

mod impl_img2img;
mod impl_main;
// mod impl_memory_optimized;
mod impl_txt2img;

pub(crate) mod lpw;
pub(crate) mod text_embeddings;

pub use self::impl_main::*;
// pub use self::impl_memory_optimized::*;
use crate::{DiffusionDeviceControl, Prompt};

/// Options for the Stable Diffusion pipeline. This includes options like device control and long prompt weighting.
#[derive(Default, Debug, Clone)]
pub struct StableDiffusionOptions {
	/// A [`DiffusionDeviceControl`] object, mapping what device to place each model on.
	pub devices: DiffusionDeviceControl
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

/// Options for the Stable Diffusion image-to-image pipeline.
#[derive(Debug)]
pub struct StableDiffusionImg2ImgOptions {
	reference_image: Array4<f32>,
	preprocessing: ImagePreprocessing,
	text_config: StableDiffusionTxt2ImgOptions
}

/// The image preprocessing method to on images that mismatch size.
#[derive(Debug)]
pub enum ImagePreprocessing {
	/// The image is resized to the target size.
	Resize,
	/// Crop extra pixels, fill the rest with black.
	CropFill
}
