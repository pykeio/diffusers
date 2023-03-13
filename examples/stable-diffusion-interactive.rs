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

use std::{cell::RefCell, env};

use kdam::{tqdm, BarExt};
use pyke_diffusers::{
	ArenaExtendStrategy, CUDADeviceOptions, DiffusionDevice, DiffusionDeviceControl, EulerDiscreteScheduler, OrtEnvironment, SchedulerOptimizedDefaults,
	StableDiffusionOptions, StableDiffusionPipeline, StableDiffusionTxt2ImgOptions
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
				StableDiffusionTxt2ImgOptions::default()
					.with_steps(20)
					.with_prompts(prompt, None)
					.callback_progress(1, move |step, _| {
						pb.borrow_mut().update_to(step);
						true
					})
					.run(&pipeline, &mut scheduler)?
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
