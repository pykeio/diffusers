use image::imageops::FilterType;
use image::{DynamicImage, Rgb32FImage};
use ndarray::{Array4, Ix};

use crate::{ImagePreprocessing, Prompt, StableDiffusionCallback, StableDiffusionImg2ImgOptions, StableDiffusionTxt2ImgOptions};

impl Default for StableDiffusionImg2ImgOptions {
	fn default() -> Self {
		Self {
			reference_image: Array4::default((1, 1, 1, 1)),
			preprocessing: ImagePreprocessing::CropFill,
			text_config: StableDiffusionTxt2ImgOptions::default()
		}
	}
}

// builder for options
impl StableDiffusionImg2ImgOptions {
	/// Get the size of each reference image.
	pub fn get_size(&self) -> (u32, u32) {
		(self.text_config.width, self.text_config.height)
	}
	/// Get the dimensions of reference images.
	pub fn get_dimensions(&self) -> (Ix, Ix, Ix, Ix) {
		self.reference_image.dim()
	}
	/// Set the size of the image. **Size will be rounded to a multiple of 8.**
	pub fn with_size(self, width: u32, height: u32) -> Self {
		self.with_width(width).with_height(height)
	}
	/// Set the width of the image. **Width will be rounded to a multiple of 8.**
	#[inline]
	pub fn with_width(mut self, width: u32) -> Self {
		self.text_config.width = (width / 8).max(1) * 8;
		self.drop_reference_image();
		self
	}
	/// Set the height of the image. **Height will be rounded to a multiple of 8.**
	#[inline]
	pub fn with_height(mut self, height: u32) -> Self {
		self.text_config.height = (height / 8).max(1) * 8;
		self.drop_reference_image();
		self
	}
	#[inline(always)]
	fn drop_reference_image(&mut self) {
		self.reference_image = Array4::default((1, 1, 1, 1));
	}
	/// The number of steps to take to generate the image. More steps typically yields higher quality images.
	pub fn with_steps(mut self, steps: usize) -> Self {
		self.text_config.steps = steps;
		self
	}
	/// Set the prompt(s) to use when generating the image.
	pub fn with_prompt<P, N>(mut self, positive_prompt: P) -> Self
	where
		P: Into<Prompt>
	{
		self.text_config.positive_prompt = positive_prompt.into();
		self
	}
	/// Set the negative prompt(s) to use when generating the image.
	pub fn with_negative_prompt<P>(mut self, negative_prompt: P) -> Self
	where
		P: Into<Prompt>
	{
		self.text_config.negative_prompt = Some(negative_prompt.into());
		self
	}
	/// Set with given seed, so that each run generates the same image.
	pub fn with_seed(mut self, seed: u64) -> Self {
		self.text_config.seed = Some(seed);
		self
	}
	/// Use a random seed, so that each run generates a different image.
	pub fn with_random_seed(mut self) -> Self {
		self.text_config.seed = None;
		self
	}
	/// The 'guidance scale' for classifier-free guidance. A lower guidance scale gives the model more freedom, but the
	/// output may not match the prompt. A higher guidance scale mean the model will match the prompt(s) more strictly,
	/// but may introduce artifacts; `7.5` is a good balance.
	pub fn with_guidance_scale(mut self, guidance_scale: f32) -> Self {
		self.text_config.guidance_scale = guidance_scale;
		self
	}
	/// Set a reference image to for generating
	pub fn with_image(mut self, image: &DynamicImage, batch: usize) -> Self {
		// whc -> nchw
		let image = self.img_norm(image);
		let shape = [batch, 3, self.text_config.height as usize, self.text_config.width as usize];
		self.reference_image = Array4::from_shape_fn(shape, |(_, c, h, w)| {
			let pixel = image.get_pixel(w as u32, h as u32);
			match c {
				0 => pixel.0[0],
				1 => pixel.0[1],
				2 => pixel.0[2],
				_ => unreachable!()
			}
		});
		self
	}
	/// Set reference images to for generating, batch size must be equal to `images.len()`
	pub fn with_images(mut self, images: &[DynamicImage]) -> Self {
		// nwhc -> nchw
		let images = images.iter().map(|image| self.img_norm(image)).collect::<Vec<_>>();
		let shape = [images.len(), 3, self.text_config.height as usize, self.text_config.width as usize];
		self.reference_image = Array4::from_shape_fn(shape, |(n, c, h, w)| {
			let pixel = images[n].get_pixel(w as u32, h as u32);
			match c {
				0 => pixel.0[0],
				1 => pixel.0[1],
				2 => pixel.0[2],
				_ => unreachable!()
			}
		});
		self
	}
	fn img_norm(&self, image: &DynamicImage) -> Rgb32FImage {
		let img = match self.preprocessing {
			ImagePreprocessing::Resize => image.resize_exact(self.text_config.width, self.text_config.height, FilterType::Lanczos3),
			ImagePreprocessing::CropFill => image.resize_to_fill(self.text_config.width, self.text_config.height, FilterType::Lanczos3)
		};
		// normalize to [0, 1]
		img.to_rgb32f()
	}
}

// builder for callbacks
impl StableDiffusionImg2ImgOptions {
	#[doc = include_str!("_doc/callback-progress.md")]
	pub fn callback_progress<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32) -> bool + 'static
	{
		self.text_config.callback = Some(StableDiffusionCallback::Progress { frequency, cb: Box::new(callback) });
		self
	}
	#[doc = include_str!("_doc/callback-latents.md")]
	pub fn callback_latents<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Array4<f32>) -> bool + 'static
	{
		self.text_config.callback = Some(StableDiffusionCallback::Latents { frequency, cb: Box::new(callback) });
		self
	}
	#[doc = include_str!("_doc/callback-decode-image.md")]
	pub fn callback_decoded<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Vec<DynamicImage>) -> bool + 'static
	{
		self.text_config.callback = Some(StableDiffusionCallback::Decoded { frequency, cb: Box::new(callback) });
		self
	}
	#[doc = include_str!("_doc/callback-approximate-image.md")]
	pub fn callback_approximate<F>(mut self, frequency: usize, callback: F) -> Self
	where
		F: Fn(usize, f32, Vec<DynamicImage>) -> bool + 'static
	{
		self.text_config.callback = Some(StableDiffusionCallback::ApproximateDecoded { frequency, cb: Box::new(callback) });
		self
	}
}
