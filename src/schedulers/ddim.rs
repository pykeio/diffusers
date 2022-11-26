use ndarray::{s, Array1, Array4, ArrayView4};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use rand::Rng;

use super::{betas_for_alpha_bar, BetaSchedule, DiffusionScheduler, SchedulerStepOutput};
use crate::SchedulerPredictionType;

/// Additional configuration for the [`DDIMScheduler`].
#[derive(Debug, Clone)]
pub struct DDIMSchedulerConfig {
	/// Option to predicted sample between -1 and 1 for numerical stability.
	pub clip_sample: bool,
	/// Each diffusion step uses the value of alphas product at that step and at the previous one. For the final step,
	/// there is no previous alpha. When this option is true, the previous alpha product is fixed to `1`, otherwise it
	/// uses the value of alpha at step 0.
	pub set_alpha_to_one: bool,
	/// An offset added to inference steps. You can use a combination of `steps_offset: 1` and `set_alpha_to_one: true`
	/// to make the last step use step 0 for the previous alpha product, as done in Stable Diffusion.
	pub steps_offset: isize
}

/// [Denoising diffusion implicit models][ddim] is a scheduler that extends the denoising procedure introduced in
/// denoising diffusion probabilistic models (DDPMs) with non-Markovian guidance.
///
/// [ddim]: https://arxiv.org/abs/2010.02502
#[derive(Clone)]
pub struct DDIMScheduler {
	alphas_cumprod: Array1<f32>,
	final_alpha_cumprod: f32,
	init_noise_sigma: f32,
	timesteps: Array1<usize>,
	num_train_timesteps: usize,
	num_inference_steps: Option<usize>,
	config: DDIMSchedulerConfig,
	prediction_type: SchedulerPredictionType
}

impl Default for DDIMScheduler {
	fn default() -> Self {
		Self::new(1000, 0.0001, 0.02, &BetaSchedule::Linear, &SchedulerPredictionType::Epsilon, None).unwrap()
	}
}

impl Default for DDIMSchedulerConfig {
	fn default() -> Self {
		Self {
			clip_sample: false,
			set_alpha_to_one: false,
			steps_offset: 1
		}
	}
}

impl DDIMScheduler {
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
		config: Option<DDIMSchedulerConfig>
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
			BetaSchedule::SquaredcosCapV2 => betas_for_alpha_bar(num_train_timesteps, 0.999)
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

		// At every step in DDIM, we are looking into the previous alphas_cumprod
		// For the final step, there is no previous alphas_cumprod because we are already at 0
		// `set_alpha_to_one` decides whether we set this parameter simply to one or whether we use the final alpha of the
		// "non-previous" one
		let final_alpha_cumprod = if config.set_alpha_to_one { 1.0 } else { alphas_cumprod[0] };

		let timesteps = Array1::linspace(0.0, num_train_timesteps as f32 - 1.0, num_train_timesteps)
			.slice(s![..;-1])
			.map(|f| *f as usize)
			.to_owned();

		// standard deviation of the initial noise distribution
		let init_noise_sigma = 1.0;

		Ok(Self {
			alphas_cumprod,
			final_alpha_cumprod,
			init_noise_sigma,
			timesteps,
			num_inference_steps: None,
			num_train_timesteps,
			prediction_type: *prediction_type,
			config
		})
	}

	fn get_variance(&self, timestep: usize, prev_timestep: isize) -> f32 {
		let alpha_prod_t = self.alphas_cumprod[timestep];
		let alpha_prod_t_prev = if prev_timestep >= 0 {
			self.alphas_cumprod[prev_timestep as usize]
		} else {
			self.final_alpha_cumprod
		};
		let beta_prod_t = 1.0 - alpha_prod_t;
		let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

		(beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)
	}
}

impl DiffusionScheduler for DDIMScheduler {
	type TimestepType = usize;

	fn scale_model_input(&mut self, sample: ArrayView4<'_, f32>, _: usize) -> Array4<f32> {
		sample.to_owned()
	}

	fn set_timesteps(&mut self, num_inference_steps: usize) {
		self.num_inference_steps = Some(num_inference_steps);

		let step_ratio = self.num_train_timesteps / num_inference_steps;

		let timesteps = Array1::range(0.0, (num_inference_steps - 1) as f32, 1.0)
			.slice(s![..;-1])
			.map(|f| (f * step_ratio as f32).round() as isize)
			.to_owned();

		self.timesteps = (timesteps + self.config.steps_offset).map(|f| *f as usize);
	}

	fn step<R: Rng + ?Sized>(&mut self, model_output: ArrayView4<'_, f32>, timestep: usize, sample: ArrayView4<'_, f32>, rng: &mut R) -> SchedulerStepOutput {
		const ETA: f32 = 0.0;
		const USE_CLIPPED_MODEL_OUTPUT: bool = false;

		// 1. get previous step value (=t-1)
		let prev_timestep = timestep as isize - (self.num_train_timesteps / self.num_inference_steps.unwrap()) as isize;

		// 2. compute alphas, betas
		let alpha_prod_t = self.alphas_cumprod[timestep];
		let alpha_prod_t_prev = if prev_timestep >= 0 {
			self.alphas_cumprod[prev_timestep as usize]
		} else {
			self.final_alpha_cumprod
		};
		let beta_prod_t = 1.0 - alpha_prod_t;

		// 3. compute predicted original sample from predicted noise - also called "predicted x_0" of formula (12)
		let mut model_output = model_output.to_owned();
		let mut pred_original_sample = match self.prediction_type {
			SchedulerPredictionType::Epsilon => (sample.to_owned() - beta_prod_t.sqrt() * model_output.clone()) / alpha_prod_t.sqrt(),
			SchedulerPredictionType::Sample => model_output.clone(),
			SchedulerPredictionType::VPrediction => {
				model_output = alpha_prod_t.sqrt() * model_output.clone() + beta_prod_t.sqrt() * sample.to_owned();
				alpha_prod_t.sqrt() * sample.to_owned() - beta_prod_t.sqrt() * model_output.clone()
			}
		};

		// 4. clip predicted x_0
		if self.config.clip_sample {
			pred_original_sample = pred_original_sample.map(|f| f.clamp(-1.0, 1.0));
		}

		// 5. compute variance: "sigma_t(η)" -> see formula (16)
		// σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
		let variance = self.get_variance(timestep, prev_timestep);
		let std_dev_t = ETA * variance.sqrt();

		if USE_CLIPPED_MODEL_OUTPUT {
			// model_output is always re-derived from the clipped x_0 in Glide
			model_output = (sample.to_owned() - alpha_prod_t.sqrt() * pred_original_sample.clone()) / beta_prod_t.sqrt();
		}

		// 6. compute direction pointing to x_t of formula (12)
		let pred_sample_direction = (1.0 - alpha_prod_t_prev - std_dev_t.powi(2)).sqrt() * model_output.clone();

		// 7. compute x_t without random noise of formula (12)
		let mut prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample.clone() + pred_sample_direction;

		if ETA > 0.0 {
			let variance_noise = Array4::<f32>::random_using(model_output.raw_dim(), StandardNormal, rng);
			let variance = self.get_variance(timestep, prev_timestep).sqrt() * ETA * variance_noise;
			prev_sample = prev_sample + variance;
		}

		SchedulerStepOutput {
			prev_sample,
			pred_original_sample: Some(pred_original_sample),
			..Default::default()
		}
	}

	fn add_noise(&mut self, original_samples: ArrayView4<'_, f32>, noise: ArrayView4<'_, f32>, timestep: usize) -> Array4<f32> {
		self.alphas_cumprod[timestep].sqrt() * original_samples.to_owned() + (1.0 - self.alphas_cumprod[timestep]).sqrt() * noise.to_owned()
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
