# Importing models
pyke Diffusers can import models from both original Stable Diffusion checkpoints (aka `.ckpt` or `.safetensors` files) or [Hugging Face Diffusers models](https://huggingface.co/models?pipeline_tag=text-to-image).

To start with these scripts, you'll need to clone the pyke Diffusers repository somewhere:

```sh
$ git clone https://github.com/pykeio/diffusers
$ cd diffusers
```

Then, activate a virtual environment (not needed if running in Colab) and install the script requirements:
```sh
# skip this if you aren't using CUDA
$ python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu116

$ python3 -m pip install -r requirements.txt
```

## Converting from a Hugging Face Diffusers model
> Some Hugging Face models require you to log in to download them. To do this, you'll need create or copy a token [from the Hugging Face settings page](https://hf.co/settings/tokens). The `READ` permission is suffice. Then, run `huggingface-cli login` and enter your token to log in. Adding a Git credential is not required.

The `hf2pyke.py` script will convert a Hugging Face Diffusers model to a pyke Diffusers model.

This will download the `runwayml/stable-diffusion-v1-5` model from the Hub and convert it, placing the result in `~/diffusers-models/stable-diffusion-v1-5`:
```sh
$ python3 scripts/hf2pyke.py runwayml/stable-diffusion-v1-5 ~/diffusers-models/stable-diffusion-v1-5
```

If you have a Hugging Face model saved to disk, you can also provide the path to it:
```sh
$ python3 scripts/hf2pyke.py /mnt/storage/stable-diffusion-v1-5 ~/diffusers-models/stable-diffusion-v1-5
```

You can also convert the model directly to float16 format. Float16 models use less disk space, RAM, and run faster on modern GPUs, with little quality loss. Float16 is recommended for GPU inference, especially on systems with low VRAM (< 10 GB).
```sh
$ python3 scripts/hf2pyke --fp16 runwayml/stable-diffusion-v1-5 ~/diffusers-models/stable-diffusion-v1-5-float16
```

### More options
`hf2pyke.py` has a few more options for performance or compatibility. Below are some commonly used ones. See `python3 scripts/hf2pyke.py --help` for the full list of options.

- `--skip-safety-checker`: Skips converting the safety checker. Enable this when converting models without a safety checker, like Stable Diffusion v2-based models.
- `--simplify-small-models`: Runs ONNX simplifier on all models but the UNet for a speed boost.
- `--overrude-unet-sample-size`: Override the sample size passed to the UNet. Set to `64` when converting Stable Diffusion v2 models on low-VRAM devices to avoid an OOM crash.
- `--no-accelerate`: Disables using `accelerate`. Enable this on Apple silicon devices, or if you get errors involving `device_map`.

## Converting from a .ckpt or .safetensors
soon™️
