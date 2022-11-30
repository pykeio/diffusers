use anyhow::Context;
use ndarray::{s, Array1, Array4, ArrayView4};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use rand::Rng;

use super::{betas_for_alpha_bar, BetaSchedule, DiffusionScheduler, SchedulerStepOutput};
use crate::{SchedulerOptimizedDefaults, SchedulerPredictionType};

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum DDPMVarianceType {
	FixedSmall,
	FixedSmallLog,
	FixedLarge,
	FixedLargeLog,
	Learned,
	LearnedRange
}

impl Default for DDPMVarianceType {
	fn default() -> Self {
		Self::FixedSmall
	}
}

/// Additional configuration for the [`DDPMScheduler`].
#[derive(Default, Debug, Clone)]
pub struct DDPMSchedulerConfig {
	/// Option to predicted sample between -1 and 1 for numerical stability.
	pub clip_sample: bool,
	/// Option to clip the variance used when adding noise to the denoised sample.
	pub variance_type: DDPMVarianceType
}

/// [Denoising diffusion probabilistic models][ddpm] (DDPMs) explores the connections between denoising score matching
/// and Langevin dynamics sampling.
///
/// [ddpm]: https://arxiv.org/abs/2006.11239
#[derive(Clone)]
pub struct DDPMScheduler {
	alphas: Array1<f32>,
	betas: Array1<f32>,
	alphas_cumprod: Array1<f32>,
	init_noise_sigma: f32,
	timesteps: Array1<f32>,
	num_train_timesteps: usize,
	num_inference_steps: Option<usize>,
	config: DDPMSchedulerConfig,
	prediction_type: SchedulerPredictionType
}

impl Default for DDPMScheduler {
	fn default() -> Self {
		Self::new(1000, 0.0001, 0.02, &BetaSchedule::Linear, &SchedulerPredictionType::Epsilon, None).unwrap()
	}
}

impl DDPMScheduler {
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
		config: Option<DDPMSchedulerConfig>
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
			BetaSchedule::Sigmoid => unimplemented!()
		};

		let alphas = 1.0 - &betas;

		let alphas_cumprod = alphas
			.view()
			.into_iter()
			.scan(1.0, |prod, alpha| {
				*prod *= *alpha;
				Some(*prod)
			})
			.collect::<Array1<_>>();

		let timesteps = Array1::linspace(0.0, num_train_timesteps as f32 - 1.0, num_train_timesteps)
			.slice(s![..;-1])
			.to_owned();

		// standard deviation of the initial noise distribution
		let init_noise_sigma = 1.0;

		Ok(Self {
			alphas,
			betas,
			alphas_cumprod,
			init_noise_sigma,
			timesteps,
			num_inference_steps: None,
			num_train_timesteps,
			prediction_type: *prediction_type,
			config
		})
	}

	fn get_variance(&self, timestep: usize) -> f32 {
		let alpha_prod_t = self.alphas_cumprod[timestep];
		let alpha_prod_t_prev = if timestep > 0 { self.alphas_cumprod[timestep - 1] } else { 1.0 };

		let variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * self.betas[timestep];
		match self.config.variance_type {
			DDPMVarianceType::FixedSmall => variance.max(1e-20),
			DDPMVarianceType::FixedSmallLog => variance.max(1e-20).log(std::f32::consts::E),
			DDPMVarianceType::FixedLarge => self.betas[timestep],
			DDPMVarianceType::FixedLargeLog => self.betas[timestep].log(std::f32::consts::E),
			DDPMVarianceType::Learned => variance,
			DDPMVarianceType::LearnedRange => unimplemented!()
		}
	}
}

impl DiffusionScheduler for DDPMScheduler {
	type TimestepType = f32;

	fn scale_model_input(&mut self, sample: ArrayView4<'_, f32>, _: f32) -> Array4<f32> {
		sample.to_owned()
	}

	fn set_timesteps(&mut self, num_inference_steps: usize) {
		let num_inference_steps = num_inference_steps.min(self.num_train_timesteps);
		self.num_inference_steps = Some(num_inference_steps);

		let timesteps = Array1::range(0.0, self.num_train_timesteps as f32, (self.num_train_timesteps / num_inference_steps) as f32)
			.slice(s![..;-1])
			.to_owned();

		self.timesteps = timesteps;
	}

	fn step<R: Rng + ?Sized>(&mut self, model_output: ArrayView4<'_, f32>, timestep: f32, sample: ArrayView4<'_, f32>, rng: &mut R) -> SchedulerStepOutput {
		let timestep = self
			.timesteps
			.iter()
			.position(|&p| p == timestep)
			.with_context(|| format!("timestep out of this schedulers bounds: {timestep}"))
			.unwrap();

		// 1. compute alphas, betas
		let alpha_prod_t = self.alphas_cumprod[timestep];
		let alpha_prod_t_prev = if timestep > 0 { self.alphas_cumprod[timestep - 1] } else { 1.0 };
		let beta_prod_t = 1.0 - alpha_prod_t;
		let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

		// 2. compute predicted original sample from predicted noise also called "predicted x_0" of formula (15)
		let mut pred_original_sample = match self.prediction_type {
			SchedulerPredictionType::Epsilon => (sample.to_owned() - beta_prod_t.sqrt() * model_output.to_owned()) / alpha_prod_t.sqrt(),
			SchedulerPredictionType::Sample => model_output.to_owned(),
			_ => unimplemented!()
		};

		// 3. clip predicted x_0
		if self.config.clip_sample {
			pred_original_sample = pred_original_sample.map(|f| f.clamp(-1.0, 1.0));
		}

		// 4. compute coefficients for pred_original_sample x_0 and current sample x_t (formula 7)
		let pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * self.betas[timestep]) / beta_prod_t;
		let current_sample_coeff = self.alphas[timestep].sqrt() * beta_prod_t_prev / beta_prod_t;

		// 5. compute predicted previous sample µ_t (formula 7)
		let pred_prev_sample = pred_original_sample_coeff * &pred_original_sample + current_sample_coeff * sample.to_owned();

		// 6. add noise
		let mut variance = Array4::zeros(pred_prev_sample.raw_dim());
		if timestep > 0 {
			let variance_noise = Array4::<f32>::random_using(model_output.raw_dim(), StandardNormal, rng);
			if self.config.variance_type == DDPMVarianceType::FixedSmallLog {
				variance = self.get_variance(timestep) * variance_noise;
			} else {
				variance = self.get_variance(timestep).sqrt() * variance_noise;
			}
		}

		let prev_sample = pred_prev_sample + variance;

		SchedulerStepOutput {
			prev_sample,
			pred_original_sample: Some(pred_original_sample),
			..Default::default()
		}
	}

	fn add_noise(&mut self, original_samples: ArrayView4<'_, f32>, noise: ArrayView4<'_, f32>, timestep: f32) -> Array4<f32> {
		let timestep = self
			.timesteps
			.iter()
			.position(|&p| p == timestep)
			.with_context(|| format!("timestep out of this schedulers bounds: {timestep}"))
			.unwrap();
		self.alphas_cumprod[timestep].sqrt() * original_samples.to_owned() + (1.0 - self.alphas_cumprod[timestep]).sqrt() * noise.to_owned()
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

impl SchedulerOptimizedDefaults for DDPMScheduler {
	fn stable_diffusion_v1_optimized_default() -> anyhow::Result<Self>
	where
		Self: Sized
	{
		Self::new(1000, 0.00085, 0.012, &BetaSchedule::ScaledLinear, &SchedulerPredictionType::Epsilon, None)
	}
}
