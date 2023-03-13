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

use anyhow::{anyhow, Context};
use ndarray::{concatenate, s, Array1, Array4, ArrayView4, Axis, Zip};
use rand::Rng;
use rgsl::IntegrationWorkspace;

use super::{BetaSchedule, DiffusionScheduler, SchedulerStepOutput};
use crate::{util::interpolation::LinearInterpolatorAccelerated, SchedulerOptimizedDefaults};

/// [Linear multistep][lm] scheduler for discrete beta schedules. Based on the [original `k-diffusion`
/// implementation][kdif] by Katherine Crowson.
///
/// [lm]: https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Bashforth_methods
/// [kdif]: https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181
pub struct LMSDiscreteScheduler {
	workspace: IntegrationWorkspace,
	betas: Array1<f32>,
	alphas: Array1<f32>,
	alphas_cumprod: Array1<f32>,
	sigmas: Array1<f32>,
	init_noise_sigma: f32,
	timesteps: Array1<f32>,
	num_train_timesteps: usize,
	num_inference_steps: Option<usize>,
	has_scale_input_been_called: bool,
	derivatives: Vec<Array4<f32>>
}

impl Default for LMSDiscreteScheduler {
	fn default() -> Self {
		Self::new(1000, 0.0001, 0.02, &BetaSchedule::Linear).unwrap()
	}
}

impl Clone for LMSDiscreteScheduler {
	fn clone(&self) -> Self {
		Self {
			workspace: IntegrationWorkspace::new(self.num_train_timesteps).unwrap(),
			betas: self.betas.clone(),
			alphas: self.alphas.clone(),
			alphas_cumprod: self.alphas_cumprod.clone(),
			sigmas: self.sigmas.clone(),
			init_noise_sigma: self.init_noise_sigma,
			timesteps: self.timesteps.clone(),
			num_train_timesteps: self.num_train_timesteps,
			num_inference_steps: self.num_inference_steps,
			has_scale_input_been_called: self.has_scale_input_been_called,
			derivatives: self.derivatives.clone()
		}
	}
}

impl LMSDiscreteScheduler {
	/// Creates a new instance of the scheduler.
	///
	/// # Parameters
	/// - **`num_train_timesteps`**: number of diffusion steps used to train the model.
	/// - **`beta_start`**: the starting `beta` value of inference.
	/// - **`beta_end`**: the final `beta` value.
	/// - **`beta_schedule`**: the beta schedule, a mapping from a beta range to a sequence of betas for stepping the
	///   model; see [`BetaSchedule`]
	///
	/// # Errors
	/// Can error if:
	/// - `num_train_timesteps` is 0
	/// - `beta_start` or `beta_end` are not normal numbers (not zero, infinite, `NaN`, or subnormal)
	/// - `beta_end` is less than or equal to `beta_start`
	/// - the given [`BetaSchedule`] is not supported by this scheduler
	///
	/// # Panics
	/// - if the GSL `IntegrationWorkspace` could not be created, which can happen if `num_train_timesteps` is
	///   exceptionally large
	pub fn new(num_train_timesteps: usize, beta_start: f32, beta_end: f32, beta_schedule: &BetaSchedule) -> anyhow::Result<Self> {
		if num_train_timesteps == 0 {
			anyhow::bail!("num_train_timesteps ({num_train_timesteps}) must be >0");
		}
		if !beta_start.is_normal() || !beta_end.is_normal() {
			anyhow::bail!("beta_start ({beta_start}) and beta_end ({beta_end}) must be normal (not zero, infinite, NaN, or subnormal)");
		}
		if beta_start >= beta_end {
			anyhow::bail!("beta_start must be < beta_end");
		}

		let betas = match beta_schedule {
			BetaSchedule::TrainedBetas(betas) => betas.clone(),
			BetaSchedule::Linear => Array1::linspace(beta_start, beta_end, num_train_timesteps),
			BetaSchedule::ScaledLinear => {
				let mut betas = Array1::linspace(beta_start.sqrt(), beta_end.sqrt(), num_train_timesteps);
				betas.par_map_inplace(|f| *f = f.powi(2));
				betas
			}
			_ => anyhow::bail!("{beta_schedule:?} not implemented for LMSDiscreteScheduler")
		};

		let mut alphas = betas.clone();
		alphas.par_map_inplace(|f| {
			*f = 1.0 - *f;
		});

		let alphas_cumprod = alphas
			.view()
			.into_iter()
			.scan(1.0, |prod, alpha| {
				*prod *= *alpha;
				Some(*prod)
			})
			.collect::<Array1<_>>();

		let mut sigmas = alphas_cumprod.clone();
		sigmas.par_map_inplace(|f| {
			*f = ((1.0 - *f) / *f).sqrt();
		});
		sigmas = concatenate![Axis(0), sigmas.slice(s![..;-1]), Array1::zeros(1,)];

		// standard deviation of the initial noise distribution
		let init_noise_sigma = *sigmas
			.iter()
			.reduce(|a, b| if a > b { a } else { b })
			.ok_or_else(|| anyhow!("init_noise_sigma could not be reduced from sigmas - this should never happen"))?;

		let timesteps = Array1::linspace(num_train_timesteps as f32 - 1.0, 0.0, num_train_timesteps);

		Ok(Self {
			workspace: IntegrationWorkspace::new(num_train_timesteps).unwrap(),
			betas,
			alphas,
			alphas_cumprod,
			sigmas,
			init_noise_sigma,
			timesteps,
			num_inference_steps: None,
			num_train_timesteps,
			has_scale_input_been_called: false,
			derivatives: vec![]
		})
	}

	/// Compute a linear multistep coefficient.
	///
	/// # Assertions
	/// `t` must be greater than or equal to `current_order`.
	///
	/// # Panics
	/// Panics if GSL's `qags` function does not succeed, which should never happen under normal circumstances.
	pub fn get_lms_coefficient(&mut self, order: usize, t: usize, current_order: usize) -> f32 {
		assert!(t >= current_order);
		let workspace = self.workspace.qags(
			|tau| {
				let mut prod = 1.0_f64;
				for k in 0..order {
					if k == current_order {
						continue;
					}

					let t_k = t as isize - k as isize;
					let t_k = if t_k < 0 { self.sigmas.len() - t_k.unsigned_abs() } else { t_k as usize };
					prod *= (tau - f64::from(self.sigmas[t_k])) / f64::from(self.sigmas[t - current_order] - self.sigmas[t_k]);
				}
				prod
			},
			f64::from(self.sigmas[t]),
			f64::from(self.sigmas[t + 1]),
			1.49e-8,
			1e-4,
			50
		);
		if workspace.0.is_success() {
			workspace.1 as f32
		} else {
			panic!("qags did not succeed; rgsl returned error: {:?}", workspace.0)
		}
	}
}

impl DiffusionScheduler for LMSDiscreteScheduler {
	type TimestepType = f32;

	fn order() -> usize {
		1
	}

	/// Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm.
	///
	/// # Panics
	/// Panics if the given `timestep` is out of this scheduler's bounds (see `num_train_timesteps`).
	fn scale_model_input(&mut self, sample: ArrayView4<'_, f32>, timestep: f32) -> Array4<f32> {
		let step_index = self
			.timesteps
			.iter()
			.position(|&p| p == timestep)
			.with_context(|| format!("timestep out of this schedulers bounds: {timestep}"))
			.unwrap();

		let sigma = self
			.sigmas
			.get(step_index)
			.expect("step_index out of sigma bounds - this shouldn't happen");

		self.has_scale_input_been_called = true;

		sample.to_owned() / (sigma.powi(2) + 1.0).sqrt()
	}

	fn set_timesteps(&mut self, num_inference_steps: usize) {
		let timesteps = Array1::linspace(self.num_train_timesteps as f32 - 1.0, 0.0, num_inference_steps);

		let mut sigmas = self.alphas_cumprod.clone();
		sigmas.par_map_inplace(|f| {
			*f = ((1.0 - *f) / *f).sqrt();
		});

		let sigmas_xa = Array1::range(0.0, sigmas.len() as f32, 1.0);
		let mut interpolator = LinearInterpolatorAccelerated::new(sigmas_xa.view(), sigmas.view());
		let n_timesteps = timesteps.len();
		let mut sigmas_int = Array1::zeros((n_timesteps + 1,));
		for (i, x) in timesteps.iter().enumerate() {
			sigmas_int[i] = interpolator.eval(*x);
		}
		sigmas_int[n_timesteps] = 0.0;

		self.sigmas = sigmas_int;
		self.timesteps = timesteps;
		self.derivatives = Vec::new();
	}

	fn step<R: Rng + ?Sized>(&mut self, model_output: ArrayView4<'_, f32>, timestep: f32, sample: ArrayView4<'_, f32>, _rng: &mut R) -> SchedulerStepOutput {
		assert!(self.has_scale_input_been_called);

		let order = 4;

		let step_index = self
			.timesteps
			.iter()
			.position(|&p| p == timestep)
			.with_context(|| format!("timestep out of this schedulers bounds: {timestep}"))
			.unwrap();

		let sigma = self
			.sigmas
			.get(step_index)
			.expect("step_index out of sigma bounds - this shouldn't happen");

		// 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
		let mut pred_original_sample = sample.to_owned();
		Zip::indexed(pred_original_sample.view_mut()).par_for_each(|i, f| {
			*f = (*sigma).mul_add(-model_output[i], *f);
		});

		// 2. convert to an ODE derivative
		let mut derivative = pred_original_sample.clone();
		Zip::indexed(derivative.view_mut()).par_for_each(|i, f| {
			*f = (sample[i] - *f) / *sigma;
		});

		self.derivatives.push(derivative.clone());
		if self.derivatives.len() > order {
			self.derivatives.remove(0);
		}

		// 3. compute linear multistep coefficients
		let order = order.min(step_index + 1);
		let lms_coeffs: Vec<_> = (0..order).map(|o| self.get_lms_coefficient(order, step_index, o)).collect();

		// 4. compute previous sample based on the derivatives path
		let mut prev_samples = Vec::new();
		for (coeff, derivative) in lms_coeffs.iter().zip(self.derivatives.iter().rev()) {
			prev_samples.push(derivative * *coeff);
		}

		let mut prev_samples = prev_samples.iter().cloned();
		let first_prev_sample = prev_samples.next().unwrap();
		let mut prev_sample = prev_samples.fold(first_prev_sample, |acc, x| acc + x);
		Zip::indexed(prev_sample.view_mut()).par_for_each(|i, f| {
			*f += sample[i];
		});

		SchedulerStepOutput {
			prev_sample,
			pred_original_sample: Some(pred_original_sample),
			..Default::default()
		}
	}

	fn add_noise(&mut self, original_samples: ArrayView4<'_, f32>, noise: ArrayView4<'_, f32>, timestep: f32) -> Array4<f32> {
		let step_index = self
			.timesteps
			.iter()
			.position(|&p| p == timestep)
			.with_context(|| format!("timestep out of this schedulers bounds: {timestep}"))
			.unwrap();

		let sigma = self
			.sigmas
			.get(step_index)
			.expect("step_index out of sigma bounds - this shouldn't happen");

		let mut noisy_samples = original_samples.to_owned();
		Zip::indexed(noisy_samples.view_mut()).par_for_each(|i, f| {
			*f += noise[i] * *sigma;
		});
		noisy_samples
	}

	fn timesteps(&self) -> ndarray::ArrayView1<'_, f32> {
		self.timesteps.view()
	}

	fn init_noise_sigma(&self) -> f32 {
		self.init_noise_sigma
	}

	fn len(&self) -> usize {
		self.num_train_timesteps
	}
}

impl SchedulerOptimizedDefaults for LMSDiscreteScheduler {
	fn stable_diffusion_v1_optimized_default() -> anyhow::Result<Self>
	where
		Self: Sized
	{
		Self::new(1000, 0.00085, 0.012, &BetaSchedule::ScaledLinear)
	}
}
