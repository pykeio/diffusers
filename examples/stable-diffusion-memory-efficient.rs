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
