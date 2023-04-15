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

pub use self::impl_img2img::{ImagePreprocessing, StableDiffusionImg2ImgOptions};
pub use self::impl_main::StableDiffusionPipeline;
pub use self::impl_txt2img::StableDiffusionTxt2ImgOptions;
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
