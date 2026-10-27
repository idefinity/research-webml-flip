---
license: other
library_name: flip
tags:
  - 3d
  - gaussian-splatting
  - triposplat
  - distillation
  - int4
  - webgpu
base_model:
  - VAST-AI/TripoSplat
---

# flip student · 2 steps · int4

TripoSplat 的扩散模型（DiT）的一个 2 步学生，量化到 int4。

上游用 20 步 Euler + CFG 3 生成 3D 高斯；这个学生用 **2 步、无 CFG** 做同一件事，
T4 上出图从 91–95 秒降到 4.6–5.0 秒（19×），权重从 1.48 GB 压到 209 MB。

**这不是一条完整管线。** 图像编码（DINOv3 + Flux2 VAE）、抠图（BiRefNet）、
解码（octree + Gaussian decoder）都是上游的，不在这里。这个仓库只替换
`ckpts/diffusion_models/triposplat_fp16.safetensors` 这一个文件。

## 文件

| 文件 | 内容 |
|---|---|
| `model.safetensors` | 763 个张量：204 个 int4（U8 成对）+ 204 个 fp16 scale + 355 个 fp16（norm/bias/embedding） |
| `dit.int4.onnx` | 同一个模型，整图 ONNX：203 个 `com.microsoft::MatMulNBits` + 剩下的 fp32 |
| `quantize.json` | 量化语义：4 bit、每 32 权重一个 fp16 scale、对称、zero point 8、沿 K 打包 |
| `config.json` | 结构参数与采样参数 |

## `dit.int4.onnx`

静态形状，输入是编码器的原始输出：

```
latent   [1, 8192, 16]     噪声 / 当前状态
camera   [1, 1, 5]
feature1 [1, 4101, 1280]   DINOv3
feature2 [1, 4096, 128]    Flux2 VAE（图内部补 5 个零 token，对齐 4101）
t        [1]
-> out_latent [1, 8192, 16], out_camera [1, 1, 5]
```

6265 个节点，294 MB。259 个 MatMul 里 203 个变成了 int4；**剩下的 56 个就是 28 层注意力的
QKᵀ 和 PV**——它们没有权重，量化不到，也不该量化。

**这张图不是最快的路径，是最互通的路径。** 在 ORT-web 1.31.1 / apple metal-3 上实测
（`webgpu/onnx-probe.html`）：

| 图 | ORT WebGPU | 我们的 WGSL |
|---|---|---|
| int4 MatMulNBits 1024→3072 @12294 | 789 GFLOP/s | 1591–1796 |
| 注意力，1 头 L=12294（物化） | 406 | 1789 |
| `com.microsoft::Attention` 16 头 | 1023 | 1789 |
| 卷积 3×3 128→128 @128² | 711 | 443 |

也就是说：矩阵乘交给 ORT 是 2.3 倍代价，注意力是 4.4 倍。**想跑进 60 秒，注意力得用
我们自己的核**，靠 `preferredOutputLocation: 'gpu-buffer'` 和 ORT 交接；其余走 ONNX。
卷积反而是 ORT 更快，所以 VAE 就该用 ONNX。

这张图在 CPU 上没有端到端跑过（一次前向 5 分钟，内核撑不住）。int4 的正确性证据是
矩阵乘级别的：fp32 对比 int4，cosine 0.997、rel_l2 0.073。

## 怎么接进上游

```python
from safetensors.torch import load_file
from triposplat import TripoSplatPipeline         # VAST-AI/TripoSplat

# 1. 反量化：U8 成对 + fp16 scale -> 权重
#    even k 在低四位，odd k 在高四位
w = (packed_to_int8(u8) - 8).float() * scale.repeat_interleave(32, dim=1)

# 2. 装进上游模块，存成 fp16 safetensors
# 3. 采样时：steps=2, shift=1.0, guidance=1.0（无 CFG）
```

`steps=50, shift=3, guidance=7` 是教师的参数，套在学生上会得到一个糊掉的模型。

## 数值（都是实测的）

量化质量，在真实采样状态上比 fp32 与「读回 int4 再反量化」的前向：

| 项 | 值 |
|---|---|
| latent cosine | 0.992 |
| latent rel-l2 | 0.123 |
| camera cosine | 0.999 |
| 密度 | 4.50 bits/weight |
| 体积 | 209,328,746 字节 |

蒸馏效果，同一 seed 与教师 20 步比（3 张 fuliji 图）：

| 项 | 学生 2 步 | 教师 20 步 |
|---|---|---|
| 出图耗时（T4） | 4.6–5.0 s | 91–95 s |
| opacity 均值 | 0.48–0.50 | 0.88–0.94 |
| latent cosine vs 教师 | 0.49 / 0.67 / 0.68 | — |
| camera cosine vs 教师 | 0.98 | — |

**诚实的短处**：opacity 掉到教师的一半，scale 偏大（0.0057 对 0.0030）。
肉眼上花纹和形体保住了，边缘略虚。latent 余弦只有 0.49–0.68，但它不是质量指标——
教师自己在 4 步时的 latent 距离更远（0.46–0.61），出图反而更糊。

## 怎么训的

- 教师：TripoSplat 官方权重，20 步 Euler，guidance 3
- 数据：`DownFlow/fuliji` 的 32 张图 + 上游示例图
- 目标：段平均速度 `v* = (x_{k+n} − x_k) / Σdt`。学生 2 步的网格是 `[0, 15, 20]`
  （按 σ 等分，每段 Δσ = 0.5），不是按步数均分的 `[0, 10, 20]`
- 训练：16 epoch，Adam，最终 loss 0.0787（camera 0.0213 / latent 0.0776）

2 步学生的上限由教师 20 步的终点决定，与教师 2 步无关——后者带 CFG，不是一个可比的对象。

## 许可

权重由 TripoSplat 蒸馏而来，上游条款见 [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat)。
此权重仅供研究，不商用。代码在 GPL-3.0 下发布。

## 引用

```bibtex
@software{flip_student_2step,
  title  = {flip: a distilled, int4 TripoSplat for WebGPU},
  year   = {2026},
  note   = {2-step student of the TripoSplat diffusion model}
}
```
