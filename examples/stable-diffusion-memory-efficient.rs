use std::sync::Arc;

use pyke_diffusers::{
	EulerDiscreteScheduler, OrtEnvironment, SchedulerOptimizedDefaults, StableDiffusionMemoryOptimizedPipeline, StableDiffusionOptions,
	StableDiffusionTxt2ImgOptions
};

fn main() -> anyhow::Result<()> {
	let environment = OrtEnvironment::default().into_arc();
	let mut scheduler = EulerDiscreteScheduler::stable_diffusion_v1_optimized_default()?;
	let pipeline = StableDiffusionMemoryOptimizedPipeline::new(&environment, "./stable-diffusion-v1-5/", StableDiffusionOptions::default())?;

	let imgs = pipeline.txt2img("photo of a red fox", &mut scheduler, StableDiffusionTxt2ImgOptions { steps: 20, ..Default::default() })?;
	imgs[0].clone().into_rgb8().save("result.png")?;

	Ok(())
}
