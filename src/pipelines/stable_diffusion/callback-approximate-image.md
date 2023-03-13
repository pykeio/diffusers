A callback to receive this step's approximately decoded latents, to be used for e.g. showing image progress
visually. This is lower quality than [`StableDiffusionCallback::Decoded`] but much faster.

Approximated images may be noisy and colors will not be accurate (especially if using a fine-tuned VAE).

## Callback Parameters:

- **`step`** (usize): The current step number.
- **`timestep`** (f32): This step's timestep.
- **`image`** (`Vec<DynamicImage>`): Vector of approximated decoded images for this step.

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