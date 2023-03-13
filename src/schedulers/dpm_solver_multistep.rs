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

use anyhow::Context;
use ndarray::{s, Array1, Array4, ArrayView4};
use rand::Rng;

use super::{betas_for_alpha_bar, BetaSchedule, DiffusionScheduler, SchedulerStepOutput};
use crate::{SchedulerOptimizedDefaults, SchedulerPredictionType};

/// The algorithm type for the solver.
///
/// We recommend to use `DPMSolverPlusPlus` with `solver_order: 2` for guided sampling (e.g. Stable Diffusion).
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum DPMSolverAlgorithmType {
	/// Implements the algorithms defined in <https://arxiv.org/abs/2211.01095>.
	#[default]
	DPMSolverPlusPlus,
	/// Implements the algorithms defined in <https://arxiv.org/abs/2206.00927>.
	DPMSolver
}

/// The solver type for the second-order solver. The solver type slightly affects the sample quality, especially with a
/// small number of steps. We empirically find that `Midpoint` solvers produce slightly better output, so we recommend
/// to use the `Midpoint` type.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum DPMSolverType {
	#[default]
	Midpoint,
	Heun
}

/// Additional configuration for the [`DPMSolverMultistepScheduler`].
#[derive(Debug, Clone)]
pub struct DPMSolverMultistepSchedulerConfig {
	/// The order of DPM-Solver; can be `1` or `2` or `3`. We recommend to use `solver_order=2` for guided
	/// sampling, and `solver_order=3` for unconditional sampling.
	pub solver_order: usize,
	/// Whether to use the "dynamic thresholding" [method introduced by Imagen](https://arxiv.org/abs/2205.11487).
	/// For pixel-space diffusion models, you can set both `algorithm_type: DPMSolverAlgorithmType::DPMSolverPlusPlus`
	/// and `thresholding: true` to use the dynamic thresholding. Note that the thresholding method is unsuitable for
	/// latent-space diffusion models (such as stable-diffusion).
	///
	/// **NOTE**: this is currently unimplemented due to complexity, please open an issue and I will get to it ASAP.
	pub thresholding: bool,
	/// The ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen's.
	pub dynamic_thresholding_ratio: f32,
	/// The threshold value for dynamic thresholding. Valid only when `thresholding: true` and
	/// `algorithm_type: DPMSolverAlgorithmType::DPMSolverPlusPlus`.
	pub sample_max_value: f32,
	/// The algorithm type for the solver, see [`DPMSolverAlgorithmType`]. We recommend to use `DPMSolverPlusPlus` with
	/// `solver_order=2` for guided sampling (e.g. Stable Diffusion).
	pub algorithm_type: DPMSolverAlgorithmType,
	/// The solver type for the second-order solver. The solver type slightly affects the sample quality, especially
	/// with a small number of steps. See [`DPMSolverType`] for more info.
	pub solver_type: DPMSolverType,
	/// Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. We empirically
	/// find this can stabilize the sampling of DPM-Solver for `steps < 15`, especially for steps <= 10.
	pub lower_order_final: bool
}

impl Default for DPMSolverMultistepSchedulerConfig {
	fn default() -> Self {
		Self {
			solver_order: 2,
			thresholding: false,
			dynamic_thresholding_ratio: 0.995,
			sample_max_value: 1.0,
			algorithm_type: DPMSolverAlgorithmType::DPMSolverPlusPlus,
			solver_type: DPMSolverType::Midpoint,
			lower_order_final: true
		}
	}
}

/// [DPM-Solver][dpm] (and the improved version [DPM-Solver++][dpm++]) is a fast dedicated high-order solver for
/// diffusion ODEs with the convergence order guarantee. Empirically, sampling by DPM-Solver with only 20 steps can
/// generate very high-quality samples, and it can generate quite good samples even in only 10 steps.
///
/// For more details, see the original paper: https://arxiv.org/abs/2206.00927 and https://arxiv.org/abs/2211.01095
///
/// Currently, we support the multistep DPM-Solver for both noise prediction models and data prediction models. We
/// recommend to use `solver_order: 2` for guided sampling, and `solver_order: 3` for unconditional sampling.
///
/// [dpm]: https://arxiv.org/abs/2206.00927
/// [dpm++]: https://arxiv.org/abs/2211.01095
#[derive(Clone)]
pub struct DPMSolverMultistepScheduler {
	alphas_cumprod: Array1<f32>,
	alpha_t: Array1<f32>,
	sigma_t: Array1<f32>,
	lambda_t: Array1<f32>,
	init_noise_sigma: f32,
	timesteps: Array1<usize>,
	num_train_timesteps: usize,
	num_inference_steps: Option<usize>,
	config: DPMSolverMultistepSchedulerConfig,
	prediction_type: SchedulerPredictionType,
	model_outputs: Vec<Option<Array4<f32>>>,
	lower_order_nums: usize
}

impl Default for DPMSolverMultistepScheduler {
	fn default() -> Self {
		Self::new(1000, 0.0001, 0.02, &BetaSchedule::Linear, &SchedulerPredictionType::Epsilon, None).unwrap()
	}
}

impl DPMSolverMultistepScheduler {
	/// Creates a new instance of the scheduler.
	///
	/// # Parameters
	/// - **`num_train_timesteps`**: number of diffusion steps used to train the model.
	/// - **`beta_start`**: the starting `beta` value of inference.
	/// - **`beta_end`**: the final `beta` value.
	/// - **`beta_schedule`**: the beta schedule, a mapping from a beta range to a sequence of betas for stepping the
	///   model; see [`BetaSchedule`]
	/// - **`prediction_type`**: the output prediction type; see [`SchedulerPredictionType`]
	///
	/// # Errors
	/// Can error if:
	/// - `num_train_timesteps` is 0
	/// - `beta_start` or `beta_end` are not normal numbers (not zero, infinite, `NaN`, or subnormal)
	/// - `beta_end` is less than or equal to `beta_start`
	pub fn new(
		num_train_timesteps: usize,
		beta_start: f32,
		beta_end: f32,
		beta_schedule: &BetaSchedule,
		prediction_type: &SchedulerPredictionType,
		config: Option<DPMSolverMultistepSchedulerConfig>
	) -> anyhow::Result<Self> {
		if num_train_timesteps == 0 {
			anyhow::bail!("num_train_timesteps ({num_train_timesteps}) must be >0");
		}
		if !beta_start.is_normal() || !beta_end.is_normal() {
			anyhow::bail!("beta_start ({beta_start}) and beta_end ({beta_end}) must be normal (not zero, infinite, NaN, or subnormal)");
		}
		if beta_start >= beta_end {
			anyhow::bail!("beta_start must be < beta_end");
		}

		let config = config.unwrap_or_default();

		let betas = match beta_schedule {
			BetaSchedule::TrainedBetas(betas) => betas.clone(),
			BetaSchedule::Linear => Array1::linspace(beta_start, beta_end, num_train_timesteps),
			BetaSchedule::ScaledLinear => {
				let mut betas = Array1::linspace(beta_start.sqrt(), beta_end.sqrt(), num_train_timesteps);
				betas.par_map_inplace(|f| *f = f.powi(2));
				betas
			}
			BetaSchedule::SquaredcosCapV2 => betas_for_alpha_bar(num_train_timesteps, 0.999),
			_ => anyhow::bail!("{beta_schedule:?} not implemented for DDIMScheduler")
		};

		let alphas = 1.0 - betas;

		let alphas_cumprod = alphas
			.view()
			.into_iter()
			.scan(1.0, |prod, alpha| {
				*prod *= *alpha;
				Some(*prod)
			})
			.collect::<Array1<_>>();

		let alpha_t = alphas_cumprod.map(|f| f.sqrt());
		let sigma_t = alphas_cumprod.map(|f| (1.0 - f).sqrt());
		let lambda_t = alpha_t.map(|f| f.log(std::f32::consts::E)) - sigma_t.map(|f| f.log(std::f32::consts::E));

		let timesteps = Array1::linspace(num_train_timesteps as f32 - 1.0, 0.0, num_train_timesteps).map(|f| *f as usize);

		// standard deviation of the initial noise distribution
		let init_noise_sigma = 1.0;

		Ok(Self {
			alphas_cumprod,
			alpha_t,
			sigma_t,
			lambda_t,
			init_noise_sigma,
			timesteps,
			num_inference_steps: None,
			num_train_timesteps,
			prediction_type: *prediction_type,
			config: config.clone(),
			lower_order_nums: 0,
			model_outputs: vec![None; config.solver_order]
		})
	}

	fn convert_model_output(&self, model_output: ArrayView4<'_, f32>, timestep: usize, sample: ArrayView4<f32>) -> Array4<f32> {
		match self.config.algorithm_type {
			DPMSolverAlgorithmType::DPMSolverPlusPlus => {
				let x0_pred = match self.prediction_type {
					SchedulerPredictionType::Epsilon => {
						let alpha_t = self.alpha_t[timestep];
						let sigma_t = self.sigma_t[timestep];
						(&sample - sigma_t * &model_output) / alpha_t
					}
					SchedulerPredictionType::Sample => model_output.to_owned(),
					SchedulerPredictionType::VPrediction => {
						let alpha_t = self.alpha_t[timestep];
						let sigma_t = self.sigma_t[timestep];
						alpha_t * &sample - sigma_t * &model_output
					}
				};
				if self.config.thresholding {
					todo!("thresholding not yet implemented for DPMSolverMultistepScheduler, please open an issue");
				}
				x0_pred
			}
			DPMSolverAlgorithmType::DPMSolver => match self.prediction_type {
				SchedulerPredictionType::Epsilon => model_output.to_owned(),
				SchedulerPredictionType::Sample => {
					let alpha_t = self.alpha_t[timestep];
					let sigma_t = self.sigma_t[timestep];
					(&sample - alpha_t * &model_output) / sigma_t
				}
				SchedulerPredictionType::VPrediction => {
					let alpha_t = self.alpha_t[timestep];
					let sigma_t = self.sigma_t[timestep];
					alpha_t * &model_output + sigma_t * &sample
				}
			}
		}
	}

	fn dpm_solver_first_order_update(&self, model_output: Array4<f32>, timestep: usize, prev_timestep: usize, sample: ArrayView4<f32>) -> Array4<f32> {
		let (lambda_t, lambda_s) = (self.lambda_t[prev_timestep], self.lambda_t[timestep]);
		let (alpha_t, alpha_s) = (self.alpha_t[prev_timestep], self.alpha_t[timestep]);
		let (sigma_t, sigma_s) = (self.sigma_t[prev_timestep], self.sigma_t[timestep]);
		let h = lambda_t - lambda_s;
		match self.config.algorithm_type {
			DPMSolverAlgorithmType::DPMSolverPlusPlus => (sigma_t / sigma_s) * &sample - (alpha_t * (std::f32::consts::E.powf(-h) - 1.0)) * model_output,
			DPMSolverAlgorithmType::DPMSolver => (alpha_t / alpha_s) * &sample - (sigma_t * (std::f32::consts::E.powf(h) - 1.0)) * model_output
		}
	}

	fn multistep_dpm_solver_second_order_update(
		&self,
		model_output_list: &Vec<Option<Array4<f32>>>,
		timestep_list: [usize; 2],
		prev_timestep: usize,
		sample: ArrayView4<f32>
	) -> Array4<f32> {
		assert_eq!(timestep_list.len(), model_output_list.len());

		let (t, s0, s1) = (prev_timestep, timestep_list[timestep_list.len() - 1], timestep_list[timestep_list.len() - 2]);
		let (m0, m1) = (model_output_list[model_output_list.len() - 1].as_ref().unwrap(), model_output_list[model_output_list.len() - 2].as_ref().unwrap());
		let (lambda_t, lambda_s0, lambda_s1) = (self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]);
		let (alpha_t, alpha_s0) = (self.alpha_t[t], self.alpha_t[s0]);
		let (sigma_t, sigma_s0) = (self.sigma_t[t], self.sigma_t[s0]);
		let (h, h_0) = (lambda_t - lambda_s0, lambda_s0 - lambda_s1);
		let r0 = h_0 / h;
		let (d0, d1) = (m0, (1.0 / r0) * (m0 - m1));
		match self.config.algorithm_type {
			DPMSolverAlgorithmType::DPMSolverPlusPlus => match self.config.solver_type {
				DPMSolverType::Midpoint => {
					((sigma_t / sigma_s0) * &sample)
						- (alpha_t * (std::f32::consts::E.powf(-h) - 1.0)) * d0
						- 0.5 * (alpha_t * (std::f32::consts::E.powf(-h) - 1.0)) * d1
				}
				DPMSolverType::Heun => {
					((sigma_t / sigma_s0) * &sample) - (alpha_t * (std::f32::consts::E.powf(-h) - 1.0)) * d0
						+ (alpha_t * ((std::f32::consts::E.powf(-h) - 1.0) / h + 1.0)) * d1
				}
			},
			DPMSolverAlgorithmType::DPMSolver => match self.config.solver_type {
				DPMSolverType::Midpoint => {
					(alpha_t / alpha_s0) * &sample
						- (sigma_t * (std::f32::consts::E.powf(h) - 1.0)) * d0
						- 0.5 * (sigma_t * (std::f32::consts::E.powf(h) - 1.0)) * d1
				}
				DPMSolverType::Heun => {
					(alpha_t / alpha_s0) * &sample
						- (sigma_t * (std::f32::consts::E.powf(h) - 1.0)) * d0
						- (sigma_t * ((std::f32::consts::E.powf(h) - 1.0) / h - 1.0)) * d1
				}
			}
		}
	}

	fn multistep_dpm_solver_third_order_update(
		&self,
		model_output_list: &Vec<Option<Array4<f32>>>,
		timestep_list: [usize; 3],
		prev_timestep: usize,
		sample: ArrayView4<f32>
	) -> Array4<f32> {
		assert_eq!(timestep_list.len(), model_output_list.len());

		let (t, s0, s1, s2) =
			(prev_timestep, timestep_list[timestep_list.len() - 1], timestep_list[timestep_list.len() - 2], timestep_list[timestep_list.len() - 3]);
		let (m0, m1, m2) = (
			model_output_list[model_output_list.len() - 1].as_ref().unwrap(),
			model_output_list[model_output_list.len() - 2].as_ref().unwrap(),
			model_output_list[model_output_list.len() - 3].as_ref().unwrap()
		);
		let (lambda_t, lambda_s0, lambda_s1, lambda_s2) = (self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1], self.lambda_t[s2]);
		let (alpha_t, alpha_s0) = (self.alpha_t[t], self.alpha_t[s0]);
		let (sigma_t, sigma_s0) = (self.sigma_t[t], self.sigma_t[s0]);
		let (h, h_0, h_1) = (lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2);
		let (r0, r1) = (h_0 / h, h_1 / h);
		let d0 = m0;
		let (d1_0, d1_1) = ((1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2));
		let d1 = &d1_0 + (r0 / (r0 + r1)) * (&d1_0 - &d1_1);
		let d2 = (1.0 / (r0 + r1)) * (d1_0 - d1_1);

		match self.config.algorithm_type {
			DPMSolverAlgorithmType::DPMSolverPlusPlus => {
				(sigma_t / sigma_s0) * &sample - (alpha_t * (std::f32::consts::E.powf(-h) - 1.0)) * d0
					+ (alpha_t * ((std::f32::consts::E.powf(-h) - 1.0) / h + 1.0)) * d1
					- (alpha_t * ((std::f32::consts::E.powf(-h) - 1.0 + h) / h.powi(2) - 0.5)) * d2
			}
			DPMSolverAlgorithmType::DPMSolver => {
				(alpha_t / alpha_s0) * &sample
					- (sigma_t * (std::f32::consts::E.powf(h) - 1.0)) * d0
					- (sigma_t * ((std::f32::consts::E.powf(h) - 1.0) / h - 1.0)) * d1
					- (sigma_t * ((std::f32::consts::E.powf(h) - 1.0 - h) / h.powi(2) - 0.5)) * d2
			}
		}
	}
}

impl DiffusionScheduler for DPMSolverMultistepScheduler {
	type TimestepType = usize;

	fn order() -> usize {
		1
	}

	fn scale_model_input(&mut self, sample: ArrayView4<'_, f32>, _: usize) -> Array4<f32> {
		sample.to_owned()
	}

	fn set_timesteps(&mut self, num_inference_steps: usize) {
		self.num_inference_steps = Some(num_inference_steps);

		let timesteps = Array1::linspace(self.num_train_timesteps as f32 - 1.0, 0.0, num_inference_steps).map(|f| *f as usize);

		self.timesteps = timesteps;
		self.model_outputs = vec![None; self.config.solver_order as _];
		self.lower_order_nums = 0;
	}

	fn step<R: Rng + ?Sized>(&mut self, model_output: ArrayView4<'_, f32>, timestep: usize, sample: ArrayView4<'_, f32>, _: &mut R) -> SchedulerStepOutput {
		let step_index = self
			.timesteps
			.iter()
			.position(|&p| p == timestep)
			.with_context(|| format!("timestep out of this schedulers bounds: {timestep}"))
			.unwrap();

		let prev_timestep = if step_index == self.timesteps.len() - 1 { 0 } else { self.timesteps[step_index + 1] };
		let lower_order_final = (step_index == self.timesteps.len() - 1) && self.config.lower_order_final && self.timesteps.len() < 15;
		let lower_order_second = (step_index == self.timesteps.len() - 2) && self.config.lower_order_final && self.timesteps.len() < 15;

		let model_output = self.convert_model_output(model_output, timestep, sample);
		for i in 0..self.config.solver_order - 1 {
			self.model_outputs[i] = self.model_outputs[i + 1].clone();
		}
		let m_len = self.model_outputs.len();
		self.model_outputs[m_len - 1] = Some(model_output.clone());

		let prev_sample = if self.config.solver_order == 1 || self.lower_order_nums < 1 || lower_order_final {
			self.dpm_solver_first_order_update(model_output, timestep, prev_timestep, sample)
		} else if self.config.solver_order == 2 || self.lower_order_nums < 2 || lower_order_second {
			let timestep_list = [self.timesteps[step_index - 1], timestep];
			self.multistep_dpm_solver_second_order_update(&self.model_outputs, timestep_list, prev_timestep, sample)
		} else {
			let timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep];
			self.multistep_dpm_solver_third_order_update(&self.model_outputs, timestep_list, prev_timestep, sample)
		};

		SchedulerStepOutput { prev_sample, ..Default::default() }
	}

	fn add_noise(&mut self, original_samples: ArrayView4<'_, f32>, noise: ArrayView4<'_, f32>, timestep: usize) -> Array4<f32> {
		self.alphas_cumprod[timestep].sqrt() * &original_samples + (1.0 - self.alphas_cumprod[timestep]).sqrt() * &noise
	}

	fn timesteps(&self) -> ndarray::ArrayView1<'_, usize> {
		self.timesteps.view()
	}

	fn init_noise_sigma(&self) -> f32 {
		self.init_noise_sigma
	}

	fn len(&self) -> usize {
		self.num_train_timesteps
	}
}

impl SchedulerOptimizedDefaults for DPMSolverMultistepScheduler {
	fn stable_diffusion_v1_optimized_default() -> anyhow::Result<Self>
	where
		Self: Sized
	{
		Self::new(
			1000,
			0.00085,
			0.012,
			&BetaSchedule::ScaledLinear,
			&SchedulerPredictionType::Epsilon,
			Some(DPMSolverMultistepSchedulerConfig {
				solver_order: 2,
				thresholding: false,
				dynamic_thresholding_ratio: 0.995,
				sample_max_value: 1.0,
				algorithm_type: DPMSolverAlgorithmType::DPMSolverPlusPlus,
				solver_type: DPMSolverType::Midpoint,
				lower_order_final: true
			})
		)
	}
}
