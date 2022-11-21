//! The schedule functions, denoted Schedulers in the library take in the output of a trained model, a sample which the
//! diffusion process is iterating on, and a timestep to return a denoised sample.
//!
//! * Schedulers define the methodology for iteratively adding noise to an image or for updating a sample based on model
//! outputs.
//!   - adding noise in different manners represent the algorithmic processes to train a diffusion model by adding noise
//!     to images.
//!   - for inference, the scheduler defines how to update a sample based on an output from a pretrained model.
//! * Schedulers are often defined by a noise schedule and an update rule to solve the differential equation
//! solution.

use ndarray::{Array1, Array4, ArrayBase, ArrayView1, ArrayView4};
use rand::Rng;

cfg_if::cfg_if! {
	if #[cfg(feature = "scheduler-lms")] {
		mod lms_discrete;
		pub use self::lms_discrete::*;
	}
}
cfg_if::cfg_if! {
	if #[cfg(feature = "scheduler-euler")] {
		mod euler_discrete;
		pub use self::euler_discrete::*;
	}
}
cfg_if::cfg_if! {
	if #[cfg(feature = "scheduler-euler-ancestral")] {
		mod euler_ancestral_discrete;
		pub use self::euler_ancestral_discrete::*;
	}
}

/// A mapping from a beta range to a sequence of betas for stepping the model.
pub enum BetaSchedule {
	/// Linear beta schedule.
	Linear,
	/// Scaled linear beta schedule.
	ScaledLinear,
	/// Pre-trained betas.
	TrainedBetas(Array1<f32>)
}

/// The output returned by a scheduler's `step` function.
pub struct SchedulerStepOutput {
	pub(crate) prev_sample: Array4<f32>,
	pub(crate) pred_original_sample: Option<Array4<f32>>,
	pub(crate) prev_sample_mean: Option<Array4<f32>>,
	pub(crate) derivative: Option<Array4<f32>>
}

#[doc(hidden)]
impl Default for SchedulerStepOutput {
	fn default() -> Self {
		Self {
			prev_sample: Array4::zeros((1, 1, 1, 1)),
			pred_original_sample: None,
			prev_sample_mean: None,
			derivative: None
		}
	}
}

impl SchedulerStepOutput {
	/// Computed sample (`x_{t-1}`) of the previous timestep. `prev_sample` should be used as the next model input in
	/// the denoising loop.
	pub fn prev_sample(&self) -> ArrayView4<'_, f32> {
		self.prev_sample.view()
	}

	/// The predicted denoised sample (`x_{0}`) based on the model output from the current timestep.
	/// `pred_original_sample` can be used to preview progress or for guidance.
	pub fn pred_original_sample(&self) -> Option<ArrayView4<'_, f32>> {
		self.pred_original_sample.as_ref().map(ArrayBase::view)
	}

	/// Mean averaged `prev_sample`. Same as `prev_sample`, only mean-averaged over previous timesteps.
	pub fn prev_sample_mean(&self) -> Option<ArrayView4<'_, f32>> {
		self.prev_sample_mean.as_ref().map(ArrayBase::view)
	}

	/// Derivative of predicted original image sample (`x_0`).
	pub fn derivative(&self) -> Option<ArrayView4<'_, f32>> {
		self.derivative.as_ref().map(ArrayBase::view)
	}
}

/// A scheduler to be used in diffusion pipelines.
#[allow(clippy::len_without_is_empty)]
pub trait DiffusionScheduler: Default + Clone {
	/// Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm.
	fn scale_model_input(&mut self, sample: ArrayView4<'_, f32>, timestep: f32) -> Array4<f32>;

	/// Sets the number of inference steps. This should be called before `step` to properly compute the sigmas and
	/// timesteps.
	fn set_timesteps(&mut self, num_inference_steps: u16);

	/// Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
	/// process from the learned model outputs (most often the predicted noise).
	fn step<R: Rng + ?Sized>(
		&mut self,
		model_output: ArrayView4<'_, f32>,
		timestep: f32,
		step_index: Option<usize>,
		sample: ArrayView4<'_, f32>,
		rng: &mut R
	) -> SchedulerStepOutput;

	/// Adds noise to the given samples.
	// NOTE: in huggingface diffusers, `timestep` is an array of shape `[batch_size]`, but all elements are identical
	// in both the Stable Diffusion img2img and inpaint pipelines, so this was simplified to a single float
	fn add_noise(&mut self, original_samples: ArrayView4<'_, f32>, noise: ArrayView4<'_, f32>, timestep: f32) -> Array4<f32>;

	/// Returns the computed scheduler timesteps.
	fn timesteps(&self) -> ArrayView1<'_, f32>;

	/// Returns the initial sigma noise value.
	fn init_noise_sigma(&self) -> f32;

	/// Returns the number of train timesteps.
	fn len(&self) -> usize;
}
