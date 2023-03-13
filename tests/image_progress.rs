use image::io::Reader;
use pyke_diffusers::StableDiffusionImg2ImgOptions;

#[test]
fn keep_image_size() {
	let image = Reader::open("assets/diffusers.png").unwrap().decode().unwrap();
	let i2i = StableDiffusionImg2ImgOptions::default().with_size(512, 256).with_image(&image, 4);
	let view = i2i.get_dimensions();
	assert_eq!(view, (4, 3, 256, 512));
}

#[test]
fn keep_image_size_x4() {
	let image = Reader::open("assets/diffusers.png").unwrap().decode().unwrap();
	let images = vec![image.clone(), image.clone(), image.clone(), image.clone()];
	let i2i = StableDiffusionImg2ImgOptions::default().with_size(512, 256).with_images(&images);
	let view = i2i.get_dimensions();
	assert_eq!(view, (4, 3, 256, 512));
}
