<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

### ✅ 2026年9月更新

- **Qwen3.5 MoE 架构 GGUF 支持（多模态、MTP 可用）** - 补齐 qwen35moe 架构 GGUF 的模型构建、专家权重加载与多模态输入链路；mmproj 图片输入与 MTP 投机解码均已在单张 32 GiB gfx906 GPU 上实测可用。

### ✅ 2026年8月更新

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

**已测试并确认可用：**

- ✅ Qwen/Qwen3.5-0.8B(原始F16权重)
- ✅ Qwen/Qwen3.5-2B(原始F16权重)
- ✅ Qwen/Qwen3.5-4B(原始F16权重)
- ✅ Qwen/Qwen3.5-9B(原始F16权重)
- ✅ unsloth/Qwen3.5-27B-GGUF
- ✅ unsloth/Qwen3.6-27B-GGUF
- ✅ unsloth/Qwen3.8-27B-GGUF
- ✅ unsloth/Qwen3.6-35B-A3B-GGUF

**已知无法运行：**
- ❌ **Qwen/Qwen3.5-35B-A3B-GPTQ-Int4** - MoE + GPTQ Int4 量化组合存在兼容性问题，导致服务无法正常启动或推理失败
- ❌ **cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit** - 在 ROCm 6.4 + gfx906 上实测仍无法正常运行，现象包括返回整串 `!` 或在引擎初始化/推理阶段崩溃

**未经测试的模型（同步自上游，可能不工作）：**
- 🔄 其他 MoE 模型（exaone_moe, glm4_moe_lite 等）
- 🔄 视觉模型（colqwen3, molmo2 等）
- 🔄 音频模型（whisper, funasr 等）
- 🔄 嵌入模型（colbert, voyage 等）
- 🔄 其他新模型

如需使用这些模型，建议先测试基本功能。遇到问题请提交 issue。

## 项目简介

这是 [vLLM](https://github.com/vllm-project/vllm) 的修改版本，专门用于 AMD gfx906 系列 GPU。它包含了针对 gfx906 架构的 ROCm 兼容性优化和变通方案。

**原作者**: [nalanzeyu](https://github.com/nalanzeyu/vllm-gfx906)

## 系统要求

- **硬件**: AMD gfx906 GPU（Radeon VII、Radeon Pro VII、Instinct MI50、Instinct MI60）
- **ROCm**: 支持以下社区兼容组合（均需要内核模式驱动）
  - ROCm 6.3 + PyTorch 2.9 + triton-gfx906 v3.5.0+gfx906（原有兼容组合）
  - ROCm 7.2 + PyTorch 2.11 + triton-rocm 3.6.0（当前仓库依赖组合）
- **Python**: 3.10+
- **操作系统**: Linux（在 Ubuntu 上测试）

ROCm 6.3 组合的 Triton 安装方式请参阅
[triton-gfx906 安装指南](https://github.com/nlzy/triton-gfx906/tree/v3.5.0+gfx906)。
当前仓库的依赖文件默认使用 ROCm 7.2 组合。该组合需要可用于 gfx906 的
rocBLAS Tensile 文件；必要时通过 `ROCBLAS_TENSILE_LIBPATH` 指定其路径。

## 安装方式

### 从源码构建

```bash
uv pip install vllm
vllm download-kernels  # precompiled kernels for faster startup
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
