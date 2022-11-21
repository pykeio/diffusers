#![doc(html_logo_url = "https://parcel.pyke.io/v2/cdn/assetdelivery/diffusers/doc/diffusers-square.png")]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![warn(rustdoc::all)]
#![warn(clippy::correctness, clippy::suspicious, clippy::complexity, clippy::perf, clippy::style)]

#[doc(hidden)]
pub mod clip;
pub mod pipelines;
pub mod schedulers;
pub(crate) mod util;

pub use ml2::onnx::Environment;
#[cfg(feature = "onnx")]
use ml2::onnx::ExecutionProvider;

pub use self::pipelines::*;
pub use self::schedulers::*;

cfg_if::cfg_if! {
	if #[cfg(feature = "onnx")] {
		/// The strategy to use for extending the device memory arena.
		#[derive(Clone, PartialEq, Eq)]
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
		#[derive(Clone, PartialEq, Eq)]
		pub enum CuDNNConvolutionAlgorithmSearch {
			/// Exhaustive kernel search. Will spend more time and memory to find the most optimal kernel for this GPU.
			Exhaustive,
			/// Heuristic kernel search. Will spend a small amount of time and memory to find an optimal kernel for this
			/// GPU.
			Heuristic,
			/// Uses the default cuDNN kernels that may not be optimized for this GPU.
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
		#[derive(Default, Clone, PartialEq, Eq)]
		pub struct CUDADeviceOptions {
			/// The strategy to use for extending the device memory arena. See [`ArenaExtendStrategy`] for more info.
			pub arena_extend_strategy: Option<ArenaExtendStrategy>,
			/// Per-session memory limit. Models may use all available VRAM if a memory limit is not set. VRAM usage may
			/// be higher than the memory limit (though typically not by much).
			pub memory_limit: Option<usize>,
			/// The type of search done for cuDNN convolution algorithms. See [`CuDNNConvolutionAlgorithmSearch`] for
			/// more info.
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
	}
}

/// A device on which to place a diffusion model on.
#[derive(Clone)]
#[non_exhaustive]
pub enum DiffusionDevice {
	/// Use the CPU as a device. This is the default device unless specified.
	/// It is highly recommended to use ONNX for CPU inference.
	CPU,
	/// Use NVIDIA CUDA as a device. Requires an NVIDIA Kepler GPU or later.
	CUDA(usize, #[cfg(feature = "onnx")] Option<CUDADeviceOptions>),
	/// Use NVIDIA TensorRT as a device. Requires an NVIDIA Kepler GPU or later.
	TensorRT,
	/// Use Windows DirectML as a device. Requires a DirectX 12 compatible GPU.
	/// Recommended for AMD GPUs.
	DirectML(usize),
	/// Custom execution provider w/ options for ONNX. Other execution providers have not been tested and may not work
	/// with some models.
	#[cfg(feature = "onnx")]
	Custom(ExecutionProvider)
}

impl DiffusionDevice {
	/// Returns [`DiffusionDevice::CUDA`] on the first CUDA device if the CUDA execution provider is available, falling
	/// back to [`DiffusionDevice::CPU`] otherwise.
	pub fn cuda_if_available() -> Self {
		#[cfg(feature = "onnx")]
		if ExecutionProvider::cuda().is_available() {
			return Self::CUDA(0, None);
		}
		Self::CPU
	}
}

#[cfg(feature = "onnx")]
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
/// safety checker on the CPU so the much more intensive UNet can be placed on the GPU.
#[derive(Clone)]
pub struct DiffusionDeviceControl {
	/// The device on which to place the Stable Diffusion variational autoencoder.
	#[cfg(feature = "onnx")]
	pub vae_encoder: DiffusionDevice,
	/// The device on which to place the Stable Diffusion variational autoencoder decoder.
	#[cfg(feature = "onnx")]
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
	pub fn all(device: DiffusionDevice) -> Self {
		Self {
			#[cfg(feature = "onnx")]
			vae_encoder: device.clone(),
			#[cfg(feature = "onnx")]
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
