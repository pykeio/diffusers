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

//! <img src="https://parcel.pyke.io/v2/cdn/assetdelivery/diffusers/doc/diffusers.webp" width="100%" alt="pyke Diffusers">
//!
//! `pyke-diffusers` is a modular library for pretrained diffusion model inference using [ONNX Runtime], inspired by
//! [Hugging Face diffusers].
//!
//! ONNX Runtime provides optimized inference for both CPUs and GPUs, including both NVIDIA & AMD GPUs via DirectML.
//!
//! `pyke-diffusers` is focused on ease of use, with an API closely modeled after Hugging Face diffusers:
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use std::sync::Arc;
//!
//! use pyke_diffusers::{
//! 	EulerDiscreteScheduler, OrtEnvironment, SchedulerOptimizedDefaults, StableDiffusionOptions,
//! 	StableDiffusionPipeline, StableDiffusionTxt2ImgOptions
//! };
//!
//! let environment = OrtEnvironment::default().into_arc();
//! let mut scheduler = EulerDiscreteScheduler::stable_diffusion_v1_optimized_default()?;
//! let pipeline =
//! 	StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;
//!
//! let imgs = StableDiffusionTxt2ImgOptions::default()
//! 	.with_prompt("photo of a red fox")
//! 	.run(&pipeline, &mut scheduler)?;
//! # Ok(())
//! # }
//! ```
//!
//! See [`StableDiffusionPipeline`] for more info on the Stable Diffusion pipeline.
//!
//! [ONNX Runtime]: https://onnxruntime.ai/
//! [Hugging Face diffusers]: https://github.com/huggingface/diffusers

#![doc(html_logo_url = "https://parcel.pyke.io/v2/cdn/assetdelivery/diffusers/doc/diffusers-square.png")]
#![warn(missing_docs)]
#![warn(rustdoc::all)]
#![warn(clippy::correctness, clippy::suspicious, clippy::complexity, clippy::perf, clippy::style)]
#![allow(clippy::tabs_in_doc_comments)]

#[doc(hidden)]
pub mod clip;
pub(crate) mod config;
pub mod pipelines;
pub mod schedulers;
pub(crate) mod util;

use ort::CPUExecutionProviderOptions;
use ort::CoreMLExecutionProviderOptions;
use ort::DirectMLExecutionProviderOptions;
pub use ort::Environment as OrtEnvironment;
use ort::ExecutionProvider;
use ort::OneDNNExecutionProviderOptions;
use ort::ROCmExecutionProviderOptions;
pub use ort::{ArenaExtendStrategy, CUDAExecutionProviderCuDNNConvAlgoSearch, CUDAExecutionProviderOptions};

pub use self::pipelines::*;
pub use self::schedulers::*;
pub use self::util::prompting;

/// A device on which to place a diffusion model on.
///
/// If a device is not specified, or a configured execution provider is not available, the model will be placed on the
/// CPU.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DiffusionDevice {
	/// Use the CPU as a device. **This is the default device unless another device is specified.**
	CPU,
	/// Use NVIDIA CUDA as a device. Requires an NVIDIA Kepler GPU or later.
	///
	/// First value is the device ID (which can be set to 0 in most cases). Second value is additional execution
	/// provider parameters. These options can be fine tuned for inference on low-VRAM GPUs
	/// (~3 GB free seems to be a good number for the Stable Diffusion v1 float16 UNet at 512x512 resolution); see
	/// [`CUDADeviceOptions`] for an example.
	CUDA(u32, Option<CUDAExecutionProviderOptions>),
	/// Use NVIDIA TensorRT as a device. Requires an NVIDIA Kepler GPU or later.
	TensorRT,
	/// Use Windows DirectML as a device. Requires a DirectX 12 compatible GPU.
	/// Recommended for AMD GPUs.
	///
	/// First value is the device ID (which can be set to 0 in most cases).
	DirectML(u32),
	/// Use ROCm as a device for AMD GPUs.
	ROCm(i32),
	/// Use Intel oneDNN as a device.
	OneDNN,
	/// Use CoreML as a device.
	CoreML,
	/// Custom execution provider w/ options. Other execution providers have not been tested and may not work with some
	/// models.
	Custom(ExecutionProvider)
}

impl From<DiffusionDevice> for ExecutionProvider {
	fn from(value: DiffusionDevice) -> Self {
		match value {
			DiffusionDevice::CPU => ExecutionProvider::CPU(CPUExecutionProviderOptions::default()),
			DiffusionDevice::CUDA(device, options) => {
				let options = options.unwrap_or_default();
				let op = CUDAExecutionProviderOptions {
					device_id: Some(device),
					..options.into()
				};
				ExecutionProvider::CUDA(op)
			}
			DiffusionDevice::TensorRT => ExecutionProvider::TensorRT(Default::default()),
			DiffusionDevice::DirectML(device) => ExecutionProvider::DirectML(DirectMLExecutionProviderOptions { device_id: device }),
			DiffusionDevice::ROCm(device) => ExecutionProvider::ROCm(ROCmExecutionProviderOptions {
				device_id: device,
				..Default::default()
			}),
			DiffusionDevice::OneDNN => ExecutionProvider::OneDNN(OneDNNExecutionProviderOptions::default()),
			DiffusionDevice::CoreML => ExecutionProvider::CoreML(CoreMLExecutionProviderOptions::default()),
			DiffusionDevice::Custom(ep) => ep
		}
	}
}

/// Select which device each model should be placed on.
///
/// For Stable Diffusion on GPUs with <6 GB VRAM, it may be favorable to place the text encoder, VAE decoder, and
/// safety checker on the CPU so the much more intensive UNet can be placed on the GPU:
/// ```
/// # use pyke_diffusers::{DiffusionDevice, DiffusionDeviceControl};
/// let devices = DiffusionDeviceControl {
/// 	unet: DiffusionDevice::CUDA(0, None),
/// 	..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct DiffusionDeviceControl {
	/// The device on which to place the Stable Diffusion variational autoencoder.
	pub vae_encoder: DiffusionDevice,
	/// The device on which to place the Stable Diffusion variational autoencoder decoder.
	pub vae_decoder: DiffusionDevice,
	/// The device on which to place the Stable Diffusion text encoder (CLIP).
	pub text_encoder: DiffusionDevice,
	/// The device on which to place the Stable Diffusion UNet.
	pub unet: DiffusionDevice,
	/// The device on which to place the Stable Diffusion safety checker.
	pub safety_checker: DiffusionDevice
}

impl DiffusionDeviceControl {
	/// Constructs [`DiffusionDeviceControl`] with all models on the same device.
	///
	/// ```no_run
	/// # fn main() -> anyhow::Result<()> {
	/// # use pyke_diffusers::{DiffusionDevice, DiffusionDeviceControl, OrtEnvironment, StableDiffusionPipeline, StableDiffusionOptions};
	/// # let environment = OrtEnvironment::default().into_arc();
	/// let pipeline = StableDiffusionPipeline::new(
	/// 	&environment,
	/// 	"./stable-diffusion-v1-5/",
	/// 	StableDiffusionOptions {
	/// 		devices: DiffusionDeviceControl::all(DiffusionDevice::CUDA(0, None)),
	/// 		..Default::default()
	/// 	}
	/// )?;
	/// # Ok(())
	/// # }
	/// ```
	///
	/// Note that if you are setting `memory_limit` in [`CUDADeviceOptions`], the memory limit is **per session** (aka
	/// per model), NOT for the entire pipeline.
	pub fn all(device: DiffusionDevice) -> Self {
		Self {
			vae_encoder: device.clone(),
			vae_decoder: device.clone(),
			text_encoder: device.clone(),
			unet: device.clone(),
			safety_checker: device
		}
	}
}

impl Default for DiffusionDeviceControl {
	fn default() -> DiffusionDeviceControl {
		DiffusionDeviceControl::all(DiffusionDevice::CPU)
	}
}
