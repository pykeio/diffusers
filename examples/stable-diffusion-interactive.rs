use std::{cell::RefCell, env};

use kdam::{tqdm, BarExt};
use pyke_diffusers::{
	ArenaExtendStrategy, CUDADeviceOptions, DiffusionDevice, DiffusionDeviceControl, EulerDiscreteScheduler, OrtEnvironment, SchedulerOptimizedDefaults,
	StableDiffusionCallback, StableDiffusionOptions, StableDiffusionPipeline, StableDiffusionTxt2ImgOptions
};
use requestty::Question;
use show_image::{ImageInfo, ImageView, WindowOptions};

#[show_image::main]
fn main() -> anyhow::Result<()> {
	let environment = OrtEnvironment::default().into_arc();
	let mut scheduler = EulerDiscreteScheduler::stable_diffusion_v1_optimized_default()?;

	let mut path = env::current_dir()?;
	path.push(env::args().nth(1).expect("expected a path to a model as the first argument"));

	let pipeline = StableDiffusionPipeline::new(
		&environment,
		path,
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
			lpw: true
		}
	)?;

	loop {
		let prompt = requestty::prompt_one(Question::input("prompt").message("🔮 enter your prompt").build());
		if let Ok(prompt) = prompt {
			let prompt = prompt.as_string().unwrap();

			let imgs = {
				let pb = RefCell::new(tqdm!(total = 20, desc = "generating"));
				pipeline.txt2img(
					prompt,
					&mut scheduler,
					StableDiffusionTxt2ImgOptions {
						steps: 20,
						callback: Some(StableDiffusionCallback::Progress {
							frequency: 1,
							cb: Box::new(move |step, _| {
								pb.borrow_mut().update_to(step);
								true
							})
						}),
						..Default::default()
					}
				)?
			};

			let window = show_image::create_window(prompt, WindowOptions::default())?;
			let image = imgs[0].clone().into_rgb8();
			let image = ImageView::new(ImageInfo::rgb8(512, 512), &image);
			window.set_image("result", image)?;
			window.run_function(|mut handle| {
				handle.set_inner_size([512, 512]);
			});
		} else {
			break;
		}
	}

	Ok(())
}
