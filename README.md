<div align=center>
    <img src="https://parcel.pyke.io/v2/cdn/assetdelivery/diffusers/doc/diffusers.png" width="100%" alt="pyke Diffusers">
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

Only generic CPU, CUDA, and TensorRT have prebuilt binaries available. Other execution providers will require you to manually build them; see the ONNX Runtime docs for more info. Additionally, you'll need to [make `ort` link to your custom-built binaries](https://github.com/pykeio/ort#execution-providers).

### LMS notes
> **Note**:
> **By default, the LMS scheduler is not enabled**, and this section can simply be skipped.

If you plan to enable the `all-schedulers` or `scheduler-lms` feature, you will need to install binaries for the GNU Scientific Library. See the [installation instructions for `rust-GSL`](https://github.com/GuillaumeGomez/rust-GSL#installation) to set up GSL.

## Installation
```toml
[dependencies]
pyke-diffusers = "0.1"
# if you'd like to use CUDA:
pyke-diffusers = { version = "0.1", features = [ "ort-cuda" ] }
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

See [the docs](https://docs.rs/pyke-diffusers) for more detailed information & examples.

### Converting models
To convert a model from a HuggingFace `diffusers` model:
1. Create and activate a virtual environment.
2. Install script requirements: `python3 -m pip install -r requirements.txt`
3. If you are converting a model directly from HuggingFace, log in to HuggingFace Hub with `huggingface-cli login` - this can be skipped if you have the model on disk
5. Convert your model with `scripts/hf2pyke.py`:
    - To convert a float32 model from HF (recommended for CPU): `python3 scripts/hf2pyke.py runwayml/stable-diffusion-v1-5 ~/pyke-diffusers-sd15/`
    - To convert a float32 model from disk: `python3 scripts/hf2pyke.py ~/stable-diffusion-v1-5/ ~/pyke-diffusers-sd15/`
    - To convert a float16 model from HF (recommended for GPU): `python3 scritps/hf2pyke.py runwayml/stable-diffusion-v1-5@fp16 ~/pyke-diffusers-sd15-fp16/`
    - To convert a float16 model from disk: `python3 scripts/hf2pyke.py ~/stable-diffusion-v1-5-fp16/ ~/pyke-diffusers/sd15-fp16/ -f16`

Float16 models are faster on GPUs, but are **not hardware-independent** (due to an ONNX Runtime issue). Float16 models must be converted on the hardware they will be run on. Float32 models are hardware-independent, but are recommended only for x86 CPU inference or older NVIDIA GPUs.

### ONNX Runtime binaries
When running the examples in this repo on Windows, you'll need to *copy the `onnxruntime*` dylibs from `target/debug/` to `target/debug/examples/`* on first run. You'll also need to copy the dylibs to `target/debug/deps/` if your project uses pyke Diffusers in a Cargo test.
