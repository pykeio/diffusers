use std::sync::Arc;

use pyke_diffusers::{
	ArenaExtendStrategy, CUDADeviceOptions, DiffusionDevice, DiffusionDeviceControl, EulerDiscreteScheduler, OrtEnvironment, SchedulerOptimizedDefaults,
	StableDiffusionOptions, StableDiffusionPipeline, StableDiffusionTxt2ImgOptions
};

fn main() -> anyhow::Result<()> {
	let environment = OrtEnvironment::default().into_arc();
	let mut scheduler = EulerDiscreteScheduler::stable_diffusion_v1_optimized_default()?;
	let pipeline = StableDiffusionPipeline::new(
		&environment,
		"./stable-diffusion-v1-5/",
		StableDiffusionOptions {
			devices: DiffusionDeviceControl {
				unet: DiffusionDevice::CUDA(
					0,
					Some(CUDADeviceOptions {
						memory_limit: Some(3000000000),
						arena_extend_strategy: Some(ArenaExtendStrategy::SameAsRequested),
						..Default::default()
					})
				),
				..Default::default()
			},
			..Default::default()
		}
	)?;

	let imgs = pipeline.txt2img("photo of a red fox", &mut scheduler, StableDiffusionTxt2ImgOptions { steps: 20, ..Default::default() })?;
	imgs[0].clone().into_rgb8().save("result.png")?;

	Ok(())
}
