use crate::{ImagePreprocessing, StableDiffusionImg2ImgOptions, StableDiffusionTxt2ImgOptions};

impl Default for StableDiffusionImg2ImgOptions {
	fn default() -> Self {
		Self {
			reference_image: Default::default(),
			preprocessing: ImagePreprocessing::CropFill,
			target_height: 512,
			target_width: 512,
			text_config: StableDiffusionTxt2ImgOptions::default()
		}
	}
}

impl StableDiffusionImg2ImgOptions {}
