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
						memory_limit: Some(3500000000),
						arena_extend_strategy: Some(ArenaExtendStrategy::SameAsRequested),
						..Default::default()
					})
				),
				..Default::default()
			},
			..Default::default()
		}
	)?;

	let mut imgs = StableDiffusionTxt2ImgOptions::default()
		.with_steps(20)
		.with_prompts("photo of a red fox", None)
		.run(&pipeline, &mut scheduler)?;

	imgs.remove(0).into_rgb8().save("result.png")?;

	Ok(())
}
