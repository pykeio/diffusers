<div align=center>
    <img src="https://parcel.pyke.io/v2/cdn/assetdelivery/diffusers/doc/diffusers.png" width="75%" alt="pyke Diffusers">
    <hr />
</div>

pyke Diffusers is a modular [Rust](https://rust-lang.org/) library for pretrained diffusion model inference to generate images, videos, or audio, using [ONNX Runtime](https://onnxruntime.ai/) as a backend for extremely optimized generation on both CPU & GPU.

## Prerequisites
You'll need **[Rust](https://rustup.rs) v1.62.1+** to use pyke Diffusers.

- If using CPU: recent (no earlier than Haswell/Zen) x86-64 CPU for best results. ARM64 supported but not recommended. For acceleration, see notes for [OpenVINO](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#requirements), [oneDNN](https://onnxruntime.ai/docs/execution-providers/oneDNN-ExecutionProvider.html), [ACL](https://onnxruntime.ai/docs/execution-providers/ACL-ExecutionProvider.html), [SNPE](https://onnxruntime.ai/docs/execution-providers/SNPE-ExecutionProvider.html)
- If using CUDA: **CUDA v11.[x](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility)**, **cuDNN v8.2.x** <sup>[more info](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)</sup>
- If using TensorRT: **CUDA v11.[x](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility)**, **TensorRT v8.4** <sup>[more info](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)</sup>
- If using ROCm: **ROCm v5.2** <sup>[more info](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)</sup>
- If using DirectML: **DirectX 12 compatible GPU**, **Windows 10 v1903+** <sup>[more info](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)</sup>

Only generic CPU, CUDA, and TensorRT have prebuilt binaries available. Other execution providers will require you to manually build them; see the ONNX Runtime docs for more info. Additionally, you'll need to [make `ml2` link to your custom-built binaries](https://github.com/allie-project/ml2#execution-providers).

### LMS notes
> **Note**:
> **By default, the LMS scheduler is not enabled**, and this section can simply be skipped.

If you plan to enable the `all-schedulers` or `scheduler-lms` feature, you will need to install binaries for the GNU Scientific Library. See the [installation instructions for `rust-GSL`](https://github.com/GuillaumeGomez/rust-GSL#installation) to set up GSL.

## Installation
```toml
[dependencies]
pyke-diffusers = "0.1"
# if you'd like to use CUDA:
pyke-diffusers = { version = "0.1", features = [ "cuda" ] }
```

The default features enable some commonly used schedulers and pipelines.

## Usage
```rust
use pyke_diffusers::{Environment, EulerDiscreteScheduler, StableDiffusionOptions, StableDiffusionPipeline, StableDiffusionTxt2ImgOptions};

let environment = Arc::new(Environment::builder().build()?);
let mut scheduler = EulerDiscreteScheduler::default();
let pipeline = StableDiffusionPipeline::new(&environment, "./stable-diffusion-v1-5", &StableDiffusionOptions::default())?;

let imgs = pipeline.txt2img("photo of a red fox", &mut scheduler, &StableDiffusionTxt2ImgOptions::default())?;
imgs[0].clone().into_rgb8().save("result.png")?;
```

See the docs for more detailed information & examples.
