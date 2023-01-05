use std::sync::Arc;

use pyke_diffusers::{
	DPMSolverMultistepScheduler, OrtEnvironment, SchedulerOptimizedDefaults, StableDiffusionMemoryOptimizedPipeline, StableDiffusionOptions,
	StableDiffusionTxt2ImgOptions
};

fn main() -> anyhow::Result<()> {
	let environment = Arc::new(OrtEnvironment::builder().with_name("Stable Diffusion").build()?);
	let mut scheduler = DPMSolverMultistepScheduler::stable_diffusion_v1_optimized_default()?;
	let pipeline = StableDiffusionMemoryOptimizedPipeline::new(&environment, "./stable-diffusion-v1-5/", &StableDiffusionOptions::default())?;

	let imgs = pipeline.txt2img(
		"photo of a red fox",
		&mut scheduler,
		&StableDiffusionTxt2ImgOptions {
			width: 512,
			height: 768,
			guidance_scale: 9.0,
			negative_prompt: Some("blurry, jpeg artifacts...".into()),
			steps: 9,
			..Default::default()
		}
	)?;
	imgs[0].clone().into_rgb8().save("result.png")?;

	Ok(())
}
