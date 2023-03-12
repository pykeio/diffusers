A callback to receive this step's fully decoded latents, to be used for e.g. showing image progress visually.
This is very expensive, as it will execute the VAE decoder on each call. See
[`StableDiffusionCallback::ApproximateDecoded`] for an approximated version.

## Callback Parameters:

- **`step`** (usize): The current step number.
- **`timestep`** (f32): This step's timestep.
- **`image`** (`Vec<DynamicImage>`): Vector of decoded images for this step.

## Callback Return

- **bool**: whether rendering should be stopped

## Callback Example

```no_run
let mut images = Vec::new();
let callback = move |_: usize, _: f32, image: Vec<DynamicImage>| -> bool {
    images.extend(image);
    true
};
```