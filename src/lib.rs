//! <img src="https://parcel.pyke.io/v2/cdn/assetdelivery/diffusers/doc/diffusers.png" width="100%" alt="pyke Diffusers">
//!
//! `pyke-diffusers` is a modular library for pretrained diffusion model inference using [ONNX Runtime], inspired by
//! [HuggingFace diffusers].
//!
//! ONNX Runtime provides optimized inference for both CPUs and GPUs, including both NVIDIA & AMD GPUs via DirectML.
//!
//! `pyke-diffusers` is focused on ease of use, with an API closely modeled after HuggingFace diffusers:
//! ```ignore
//! use std::sync::Arc;
//!
//! use pyke_diffusers::{
//! 	EulerDiscreteScheduler, OrtEnvironment, StableDiffusionOptions, StableDiffusionPipeline,
//! 	StableDiffusionTxt2ImgOptions
//! };
//!
//! let environment = Arc::new(OrtEnvironment::builder().build()?);
//! let mut scheduler = EulerDiscreteScheduler::default();
//! let pipeline =
//! 	StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5/", &StableDiffusionOptions::default())?;
//!
//! let imgs = pipeline.txt2img("photo of a red fox", &mut scheduler, &StableDiffusionTxt2ImgOptions::default())?;
//! ```
//!
//! See [`StableDiffusionPipeline`] for more info on the Stable Diffusion pipeline.
//!
//! [ONNX Runtime]: https://onnxruntime.ai/
//! [HuggingFace diffusers]: https://github.com/huggingface/diffusers

#![doc(html_logo_url = "https://parcel.pyke.io/v2/cdn/assetdelivery/diffusers/doc/diffusers-square.png")]
#![warn(missing_docs)]
#![warn(rustdoc::all)]
#![warn(clippy::correctness, clippy::suspicious, clippy::complexity, clippy::perf, clippy::style)]
#![allow(clippy::tabs_in_doc_comments)]

#[cfg(feature = "tokenizers")]
#[doc(hidden)]
pub mod clip;
pub(crate) mod config;
pub mod pipelines;
pub mod schedulers;
pub(crate) mod util;

pub use ort::Environment as OrtEnvironment;
use ort::ExecutionProvider;

pub use self::pipelines::*;
pub use self::schedulers::*;

/// The strategy to use for extending the device memory arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArenaExtendStrategy {
	/// Subsequent memory allocations extend by larger amounts (multiplied by powers of two)
	PowerOfTwo,
	/// Memory allocations extend only by the requested amount.
	SameAsRequested
}

impl Default for ArenaExtendStrategy {
	fn default() -> Self {
		Self::PowerOfTwo
	}
}

impl From<ArenaExtendStrategy> for String {
	fn from(val: ArenaExtendStrategy) -> Self {
		match val {
			ArenaExtendStrategy::PowerOfTwo => "kNextPowerOfTwo".to_string(),
			ArenaExtendStrategy::SameAsRequested => "kSameAsRequested".to_string()
		}
	}
}

/// The type of search done for cuDNN convolution algorithms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CuDNNConvolutionAlgorithmSearch {
	/// Exhaustive kernel search. Will spend more time and memory to find the most optimal kernel for this GPU.
	/// This is the **default** value set by ONNX Runtime.
	Exhaustive,
	/// Heuristic kernel search. Will spend a small amount of time and memory to find an optimal kernel for this
	/// GPU.
	Heuristic,
	/// Uses the default cuDNN kernels that may not be optimized for this GPU. **This is NOT the actual default
	/// value set by ONNX Runtime, the default is set to `Exhaustive`.**
	Default
}

impl Default for CuDNNConvolutionAlgorithmSearch {
	fn default() -> Self {
		Self::Exhaustive
	}
}

impl From<CuDNNConvolutionAlgorithmSearch> for String {
	fn from(val: CuDNNConvolutionAlgorithmSearch) -> Self {
		match val {
			CuDNNConvolutionAlgorithmSearch::Exhaustive => "EXHAUSTIVE".to_string(),
			CuDNNConvolutionAlgorithmSearch::Heuristic => "HEURISTIC".to_string(),
			CuDNNConvolutionAlgorithmSearch::Default => "DEFAULT".to_string()
		}
	}
}

/// Device options for the CUDA execution provider.
///
/// For low-VRAM devices running Stable Diffusion v1, it's best to use a float16 model with the following parameters:
/// ```ignore
/// CUDADeviceOptions {
/// 	memory_limit: Some(3000000000),
/// 	arena_extend_strategy: Some(ArenaExtendStrategy::SameAsRequested),
/// 	..Default::default()
/// }
/// ```
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct CUDADeviceOptions {
	/// The strategy to use for extending the device memory arena. See [`ArenaExtendStrategy`] for more info.
	pub arena_extend_strategy: Option<ArenaExtendStrategy>,
	/// Per-session (aka per-model) memory limit. Models may use all available VRAM if a memory limit is not set.
	/// VRAM usage may be higher than the memory limit (though typically not by much).
	pub memory_limit: Option<usize>,
	/// The type of search done for cuDNN convolution algorithms. See [`CuDNNConvolutionAlgorithmSearch`] for
	/// more info.
	///
	/// **NOTE**: Setting this to any value other than `Exhaustive` seems to break float16 models!
	pub cudnn_conv_algorithm_search: Option<CuDNNConvolutionAlgorithmSearch>
}

impl From<CUDADeviceOptions> for ExecutionProvider {
	fn from(val: CUDADeviceOptions) -> Self {
		let mut ep = ExecutionProvider::cuda();
		if let Some(arena_extend_strategy) = val.arena_extend_strategy {
			ep = ep.with("arena_extend_strategy", arena_extend_strategy);
		}
		if let Some(memory_limit) = val.memory_limit {
			ep = ep.with("gpu_mem_limit", memory_limit.to_string());
		}
		if let Some(cudnn_conv_algorithm_search) = val.cudnn_conv_algorithm_search {
			ep = ep.with("cudnn_conv_algo_search", cudnn_conv_algorithm_search);
		}
		ep
	}
}

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
	CUDA(usize, Option<CUDADeviceOptions>),
	/// Use NVIDIA TensorRT as a device. Requires an NVIDIA Kepler GPU or later.
	TensorRT,
	/// Use Windows DirectML as a device. Requires a DirectX 12 compatible GPU.
	/// Recommended for AMD GPUs.
	///
	/// First value is the device ID (which can be set to 0 in most cases).
	DirectML(usize),
	/// Custom execution provider w/ options. Other execution providers have not been tested and may not work with some
	/// models.
	Custom(ExecutionProvider)
}

impl From<DiffusionDevice> for ExecutionProvider {
	fn from(value: DiffusionDevice) -> Self {
		match value {
			DiffusionDevice::CPU => ExecutionProvider::cpu(),
			DiffusionDevice::CUDA(device, options) => {
				let options = options.unwrap_or_default();
				let mut ep: ExecutionProvider = options.into();
				ep = ep.with("device_id", device.to_string());
				ep
			}
			DiffusionDevice::TensorRT => ExecutionProvider::tensorrt(),
			DiffusionDevice::DirectML(_) => todo!("sorry, not implemented yet, please open an issue"),
			DiffusionDevice::Custom(ep) => ep
		}
	}
}

/// Select which device each model should be placed on.
///
/// For Stable Diffusion on GPUs with <6 GB VRAM, it may be favorable to place the text encoder, VAE decoder, and
/// safety checker on the CPU so the much more intensive UNet can be placed on the GPU:
/// ```ignore
/// DiffusionDeviceControl {
/// 	unet: DiffusionDevice::CUDA(0, None),
/// 	..Default::default()
/// }
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
	/// ```ignore
	/// let pipeline = StableDiffusionPipeline::new(
	/// 	&environment,
	/// 	"./stable-diffusion-v1-5/",
	/// 	&StableDiffusionOptions {
	/// 		devices: DiffusionDeviceControl::all(DiffusionDevice::CUDA(0, None)),
	/// 		..Default::default()
	/// 	}
	/// )?;
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
	fn default() -> Self {
		DiffusionDeviceControl::all(DiffusionDevice::CPU)
	}
}
