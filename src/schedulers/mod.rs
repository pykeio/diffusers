//! The schedule functions, denoted *Schedulers* in the library, take in the output of a trained model, a sample which
//! the diffusion process is iterating on, and a timestep, returning a denoised sample.
//!
//! * Schedulers define the methodology for iteratively adding noise to an image or for updating a sample based on model
//! outputs.
//!   - adding noise in different manners represent the algorithmic processes to train a diffusion model by adding noise
//!     to images.
//!   - for inference, the scheduler defines how to update a sample based on an output from a pretrained model.
//! * Schedulers are often defined by a *noise schedule* and an *update rule* to solve the differential equation
//! solution.
//!
//! Schedulers change how the output image forms. Some schedulers may produce higher quality results than others.
//! In the case of Stable Diffusion v1, [`EulerDiscreteScheduler`] and [`EulerAncestralDiscreteScheduler`] are
//! exceptionally creative and can produce high quality results in as few as 20 steps.

use ndarray::{Array1, Array4, ArrayBase, ArrayView1, ArrayView4};
use num_traits::ToPrimitive;
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
cfg_if::cfg_if! {
	if #[cfg(feature = "scheduler-ddim")] {
		mod ddim;
		pub use self::ddim::*;
	}
}
cfg_if::cfg_if! {
	if #[cfg(feature = "scheduler-ddpm")] {
		mod ddpm;
		pub use self::ddpm::*;
	}
}
cfg_if::cfg_if! {
	if #[cfg(feature = "scheduler-dpm-solver")] {
		mod dpm_solver;
		pub use self::dpm_solver::*;
	}
}

/// Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
/// `(1-beta)` over time from `t = [0,1]`.
///
/// Contains a function `alpha_bar` that takes an argument `t` and transforms it to the cumulative product of `(1-beta)`
/// up to that part of the diffusion process.
pub(crate) fn betas_for_alpha_bar(num_diffusion_timesteps: usize, max_beta: f32) -> Array1<f32> {
	let alpha_bar = |time_step: usize| f32::cos((time_step as f32 + 0.008) / 1.008 * std::f32::consts::FRAC_PI_2).powi(2);
	let mut betas = Vec::with_capacity(num_diffusion_timesteps);
	for i in 0..num_diffusion_timesteps {
		let t1 = i / num_diffusion_timesteps;
		let t2 = (i + 1) / num_diffusion_timesteps;
		betas.push((1.0 - alpha_bar(t2) / alpha_bar(t1)).min(max_beta));
	}
	Array1::from_vec(betas)
}

/// A mapping from a beta range to a sequence of betas for stepping the model.
#[derive(Debug, Clone, PartialEq)]
pub enum BetaSchedule {
	/// Linear beta schedule.
	Linear,
	/// Scaled linear beta schedule for latent diffusion models.
	ScaledLinear,
	/// Glide cosine schedule
	SquaredcosCapV2,
	/// GeoDiff sigmoid schedule
	Sigmoid,
	/// Pre-trained betas.
	TrainedBetas(Array1<f32>)
}

/// Scheduler prediction type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPredictionType {
	/// predict epsilon (noise)
	Epsilon,
	/// predict sample (data / `x0`)
	Sample,
	/// predict [v-objective](https://arxiv.org/abs/2202.00512)
	VPrediction
}

/// The output returned by a scheduler's `step` function.
#[derive(Clone)]
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
	/// Scheduler timestep type.
	type TimestepType: Copy + Clone + ToPrimitive;

	/// Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
	/// current timestep.
	fn scale_model_input(&mut self, sample: ArrayView4<'_, f32>, timestep: Self::TimestepType) -> Array4<f32>;

	/// Sets the number of inference steps. This should be called before `step` to properly compute the sigmas and
	/// timesteps.
	fn set_timesteps(&mut self, num_inference_steps: usize);

	/// Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
	/// process from the learned model outputs (most often the predicted noise).
	fn step<R: Rng + ?Sized>(
		&mut self,
		model_output: ArrayView4<'_, f32>,
		timestep: Self::TimestepType,
		sample: ArrayView4<'_, f32>,
		rng: &mut R
	) -> SchedulerStepOutput;

	/// Adds noise to the given samples.
	// NOTE: in huggingface diffusers, `timestep` is an array of shape `[batch_size]`, but all elements are identical
	// in both the Stable Diffusion img2img and inpaint pipelines, so this was simplified to a single float
	fn add_noise(&mut self, original_samples: ArrayView4<'_, f32>, noise: ArrayView4<'_, f32>, timestep: Self::TimestepType) -> Array4<f32>;

	/// Returns the computed scheduler timesteps.
	fn timesteps(&self) -> ArrayView1<'_, Self::TimestepType>;

	/// Returns the initial sigma noise value.
	fn init_noise_sigma(&self) -> f32;

	/// Returns the number of train timesteps.
	fn len(&self) -> usize;
}
