use anyhow::{anyhow, Context};
use ndarray::{concatenate, s, Array1, Array4, ArrayView4, Axis, Zip};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use rand::Rng;

use super::{BetaSchedule, DiffusionScheduler, SchedulerStepOutput};
use crate::{util::interpolation::LinearInterpolatorAccelerated, SchedulerOptimizedDefaults};

/// Euler scheduler (Algorithm 2) from [Karras et al. (2022)](https://arxiv.org/abs/2206.00364).
///
/// Based on the original [`k-diffusion` implementation by Katherine Crowson][kd].
///
/// [kd]: https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L51
#[derive(Clone)]
pub struct EulerDiscreteScheduler {
	alphas_cumprod: Array1<f32>,
	sigmas: Array1<f32>,
	init_noise_sigma: f32,
	timesteps: Array1<f32>,
	num_train_timesteps: usize,
	num_inference_steps: Option<usize>,
	has_scale_input_been_called: bool
}

impl Default for EulerDiscreteScheduler {
	fn default() -> Self {
		Self::new(1000, 0.0001, 0.02, &BetaSchedule::Linear).unwrap()
	}
}

impl EulerDiscreteScheduler {
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
			_ => anyhow::bail!("{beta_schedule:?} not implemented for EulerDiscreteScheduler")
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

		let mut sigmas = alphas_cumprod.clone();
		sigmas.par_map_inplace(|f| {
			*f = ((1.0 - *f) / *f).sqrt();
		});
		sigmas = concatenate![Axis(0), sigmas.slice(s![..;-1]), Array1::zeros(1,)];

		let timesteps = Array1::linspace(0.0, num_train_timesteps as f32 - 1.0, num_train_timesteps)
			.slice(s![..;-1])
			.to_owned();

		// standard deviation of the initial noise distribution
		let init_noise_sigma = *sigmas
			.iter()
			.reduce(|a, b| if a > b { a } else { b })
			.ok_or_else(|| anyhow!("init_noise_sigma could not be reduced from sigmas - this should never happen"))?;

		Ok(Self {
			alphas_cumprod,
			sigmas,
			init_noise_sigma,
			timesteps,
			num_inference_steps: None,
			num_train_timesteps,
			has_scale_input_been_called: false
		})
	}
}

impl DiffusionScheduler for EulerDiscreteScheduler {
	type TimestepType = f32;

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
		self.num_inference_steps = Some(num_inference_steps);

		let timesteps = Array1::linspace(0.0_f32, (self.num_train_timesteps - 1) as f32, num_inference_steps)
			.slice(s![..;-1])
			.to_owned();

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
	}

	fn step<R: Rng + ?Sized>(&mut self, model_output: ArrayView4<'_, f32>, timestep: f32, sample: ArrayView4<'_, f32>, rng: &mut R) -> SchedulerStepOutput {
		let s_churn = 0.0_f32;
		let s_tmin = 0.0_f32;
		let s_tmax = f32::INFINITY;
		let s_noise = 1.0_f32;

		assert!(self.has_scale_input_been_called);

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

		let gamma = if s_tmin <= *sigma && *sigma <= s_tmax {
			(s_churn / (self.sigmas.len() as f32 - 1.0)).min(2.0_f32.sqrt() - 1.0)
		} else {
			0.0
		};
		let eps = Array4::<f32>::random_using(model_output.raw_dim(), StandardNormal, rng) * s_noise;
		let sigma_hat = *sigma * (gamma + 1.0);
		let sample = if gamma > 0.0 {
			sample.to_owned() + eps * sigma_hat.mul_add(sigma_hat, -sigma.powi(2)).sqrt()
		} else {
			sample.to_owned()
		};

		let pred_original_sample = sample.clone() - sigma_hat * model_output.to_owned();

		let derivative = (sample.clone() - &pred_original_sample) / sigma_hat;

		let dt = self.sigmas[step_index + 1] - sigma_hat;

		let prev_sample = sample + derivative * dt;

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

impl SchedulerOptimizedDefaults for EulerDiscreteScheduler {
	fn stable_diffusion_v1_optimized_default() -> anyhow::Result<Self>
	where
		Self: Sized
	{
		Self::new(1000, 0.00085, 0.012, &BetaSchedule::ScaledLinear)
	}
}
