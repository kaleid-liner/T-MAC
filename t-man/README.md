<h1 align="center">
T-MAN
</h1>

<h3 align="center">
Efficient Low-Bit LLM Inference on NPU
</h3>

T-MAN extends the idea of utilizing table lookup for low-bit inference to Qualcomm NPU. It is **the first NPU inference framework** to work with [BitNet](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) and [GPTQ](https://github.com/ModelCloud/GPTQModel) quantization formats on NPU.

T-MAN currently supports the latest LLMs such as BitNet, Qwen3, and Llama3. By utilizing the SOTA QAT method BitNet and [BitDistiller](https://github.com/DD-DuDa/BitDistiller), T-MAN achieves accuracy with efficiency.

By achieving up to 50 t/s token generation for [BitNet-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) on Snapdragon 8G3, T-MAN is 2x faster than T-MAC. T-MAN is also 1.4x faster for Llama-3.1-8B compared to Qualcomm [QNN](https://aihub.qualcomm.com/mobile/models/llama_v3_1_8b_instruct).

As only NPU is required, T-MAN does not impact the performance of your commonly used apps that rely on CPU and GPU.

### Use the Android App

- Get the apk from the [release page]().
- Choose a model (e.g., Qwen3-8B) in settings, and the model files will be downloaded automatically. (internet access needed)
- Load the model.
- Enjoy your conversation!

### :warning: Notice

- The demo APK is currently only tested on Android 8GEN3.
- The models are exported using [KV Cache Mode](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/oss_scripts/llama/README.md). The prefill is slow. We will release the hybrid models soon.

### Build

If you want to build yourself, please refer to [build.md](docs/build.md).
