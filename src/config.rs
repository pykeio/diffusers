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

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
//#[serde(tag = "type")]
#[serde(rename_all = "kebab-case")]
pub enum DiffusionFramework {
	Onnx // Ort { opset: u8 }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum TokenizerConfig {
	#[serde(rename_all = "kebab-case")]
	CLIPTokenizer {
		path: String,
		model_max_length: usize,
		bos_token: u32,
		eos_token: u32
	}
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct CLIPFeatureExtractorConfig {
	pub resample: u32,
	pub size: u32,
	pub crop: [u32; 2],
	pub crop_center: bool,
	pub rgb: bool,
	pub normalize: bool,
	pub resize: bool,
	pub image_mean: Vec<f32>,
	pub image_std: Vec<f32>
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct CLIPTextModelConfig {
	pub path: String
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct UNetConfig {
	pub path: String
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct VAEConfig {
	pub encoder: Option<String>,
	pub decoder: String
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct SafetyCheckerConfig {
	pub path: String
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct StableDiffusionModelHashes {
	pub text_encoder: String,
	pub unet: String,
	pub vae_encoder: Option<String>,
	pub vae_decoder: String,
	pub safety_checker: Option<String>
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct StableDiffusionConfig {
	pub tokenizer: TokenizerConfig,
	pub feature_extractor: Option<CLIPFeatureExtractorConfig>,
	pub text_encoder: CLIPTextModelConfig,
	pub vae: VAEConfig,
	pub unet: UNetConfig,
	pub safety_checker: Option<SafetyCheckerConfig>,
	pub hashes: StableDiffusionModelHashes
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "pipeline", rename_all = "kebab-case")]
#[non_exhaustive]
pub enum DiffusionPipeline {
	StableDiffusion {
		framework: DiffusionFramework,
		#[serde(flatten)]
		inner: StableDiffusionConfig
	}
}
