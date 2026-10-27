# 计划：TripoSplat 的 WebGPU 发行版

目标：浏览器里丢一张图，出 65536 个高斯，60 秒以内，首屏下载不超过 300 MB。

上游是 PyTorch + CUDA 的参考实现。我们要造的是它的另一个发行版：**五段全部蒸馏、全部量化，
以 ONNX 为主干，跑在 onnxruntime-web 的 WebGPU EP 上**。仓库里放的是 ONNX 图、一个 JS 运行时、
以及把上游蒸馏成这些图的脚本。

---

## 一、每一段要动什么

我们量过每一段在这块 GPU 上的时间（`webgpu/budget.html`，Apple M 系列）：

| 段 | 现在 | 参数量 | 蒸馏 | 量化 | 浏览器里的难点 |
|---|---|---|---|---|---|
| 抠图 BiRefNet | 未测，估 3–8 s | 220 M | **必须换小模型** | int4 linear + fp16 conv | Swin 的窗口注意力 + 可变形卷积，两样都不好搬 |
| 图像编码 DINOv3 ViT-H | 4.8 s | 630 M | **必须**：特征蒸馏到小 ViT | int4 | 4101 个 token 的全注意力 |
| 图像编码 Flux2 VAE | 15 s | 30 M | **必须**：卷积最贵 | conv 留 fp16 | 12 个 resnet block，1024² 起 |
| 扩散 DiT | 26.2 s | 370 M | 已做（2 步学生） | **已做**（int4，209 MB） | 12294 token 的注意力 |
| octree 解码 | 0.5 s | 49 M | 不必 | int4 | 8 层采样循环要在 JS 里驱动 |
| Gaussian 解码 | 1.4 s | 235 M | 不必 | int4 | 16 块 × 2048 点，形状规整，好搬 |

**关键一点：前端蒸馏不动 DiT。** 特征蒸馏只要求学生编码器输出和教师**同一个维度的特征**
（DINO 1280 维、VAE 4096×256），DiT 的输入契约不变。所以已经训好的 2 步学生继续有效，
前端换掉不用重训扩散模型。

**代价**：前端特征对不齐，最后的高斯就会歪。所以每一段蒸馏都要配一个「出 PLY 看接触表」的验收，
不能只看特征余弦。

---

## 二、ONNX 怎么用，哪里用不了

ONNX Runtime 1.29 起，原生 WebGPU EP 带上了 **MatMulBnb4**（4 bit 权重）、PagedAttention、
以及一批整数算子。这就是我们能用 ONNX 的前提。

**用 ONNX 的部分**：所有带权重的矩阵乘。量化器用 `onnxruntime.quantization.matmul_nbits_quantizer`，
`bits=4, block_size=32`，和我们现有的 packer 同一个语义（对称、zero point 8、沿 K 打包）。

**q4f16 而不是 q4**：权重 4 bit、激活和 I/O 用 fp16，norm/softmax/rotary 留 fp32。
Metal 有 `shader-f16`，所以走 q4f16；没有这个特性的适配器（部分 Linux Vulkan）退回 q4。

**卷积不量化。** `ConvInteger` / `QLinearConv` 在 WebGPU EP 上没有实现，量化了会掉回 CPU，
比不量化慢一个数量级。VAE 和抠图的卷积留 fp16。

**这三个问号已经量过了**（`webgpu/onnx-probe.html`，ORT-web 1.31.1，apple metal-3，
shader-f16 有）：

| 图 | ORT WebGPU | 我们手写的 WGSL | 差 |
|---|---|---|---|
| int4 1024→3072 @12294 | 98.0 ms · **789 GFLOP/s** | 1591–1796 | 2.3× |
| int4 1024→4096 @12294 | 130.2 ms · 792 | 同上 | 2.3× |
| attention 1 头 L=12294 | 95.3 ms · **406 GFLOP/s** | 1789 | 4.4× |
| conv 3×3 128→128 @128² | 6.8 ms · **711 GFLOP/s** | 443 | 0.6×（ORT 更快） |
| 同上，走 wasm EP | 83.2 ms · 58 | — | 12×，证明没有回退 |

结论三条：

1. **`MatMulNBits` 真的跑在 WebGPU EP 上。** 不是 wasm 回退：同一个卷积图在 wasm 上是
   58 GFLOP/s，在 WebGPU 上是 711。
2. **注意力必须自己来。** ORT 把它拆成 2 个 MatMul + Softmax（17 个节点），score 矩阵真的物化了，
   406 GFLOP/s。按这个速率，DiT 的注意力是 16 头 × 95.3 ms × 24 块 × 2 步 = **73 秒**，
   而我们融合核是 16.6 秒。**纯 ONNX 的 DiT 一次前向约 94 秒，我们自己的核是 26 秒。**
   所以走 c：ORT 管除注意力以外的部分，注意力用我们的 WGSL，靠 `gpu-buffer` 交接。
3. **卷积交给 ORT。** 711 比我们手写的直接卷积快 60%，VAE 的 6.75 TFLOP 落在 9.5 秒
   （原来按手写核估的是 6.6–15.2 秒）。这一头不用自己写。

**所以架构是混合的，而且理由不是偏好、是量出来的**：

| 段 | 跑在哪 |
|---|---|
| VAE / 抠图（卷积为主） | ONNX + WebGPU EP（711 GFLOP/s，比我们快） |
| DINO 学生 / DiT 的矩阵乘 | ONNX int4（789，可用但不是最快） |
| DiT 的注意力 | 我们的 WGSL（4.4×） |
| octree 8 层循环 + 解码 | ONNX 单层图 + JS 循环 |
| Gaussian 光栅化 | `site/src/viewer.js` |

代价：仓库里跑着两个运行时。交接靠 `preferredOutputLocation: 'gpu-buffer'`，
ORT-web 既支持把输出留在显存，也支持直接吃显存输入。

**还没量的**：fp16 激活（q4f16）那一档，和 `com.microsoft::Attention` 在 WebGPU 上有没有
融合核（P0 v8 正在跑）。如果融合核存在，第 2 条可以推翻。

---

## 三、仓库结构

```
flip-web/
  onnx/
    dit.q4f16.onnx (+ .data)        2 步学生，int4
    decoder.q4f16.onnx               octree + Gaussian 头部
    dino_student.q4f16.onnx          蒸馏后的小 ViT
    vae_student.onnx                 蒸馏后的小卷积网，fp16
    matte.q4f16.onnx                 替换 BiRefNet
  web/
    app.js       五段编排、ORT session 管理、gpu-buffer 交接
    attn.wgsl    如果第 2 节走了 c，这里是注意力
    splat.wgsl   3DGS 光栅化
  tools/
    distill_dino.py    特征蒸馏（对着教师 1280 维输出做 cosine）
    distill_vae.py     特征蒸馏（4096×256）
    distill_matte.py   对着 BiRefNet 的 alpha 做回归
    export_onnx.py     PyTorch → ONNX，固定 shape
    quantize.py        matmul_nbits_quantizer，q4f16
    check_parity.py    每一步的 int4 vs fp16 数值对齐
  eval/
    run_sheet.py       一图多法，拼接触表
  configs/*.json       每段的 shape、采样参数、量化参数
```

权重全部内嵌在 ONNX 里（MatMulNBits 把权重放在 initializer，外部数据用 `.data` 旁挂），
所以没有单独的权重文件，也没有 `manifest.json`。

---

## 四、顺序

**P0 · 三个问题，只用现有的 2 步学生，不训任何新模型。**
- ORT-web 能不能加载 q4f16 的 DiT 并跑完一次前向？`MatMulNBits` 要真的落在 WebGPU EP 上，
  不是悄悄回退到 wasm。做法：解析 session 的算子分布，数一数有多少节点落在 CPU。
- 注意力物化要多少内存、多少时间？（决定第 2 节 a/b/c）
- fp16 卷积在 WebGPU EP 上的实际 GFLOP/s？（我们自己的直接卷积是 443，用这个当基线）

产物：一页 `web/probe.html`，三个数字。这三个数决定后面所有事。

导出本身要在有 torch 的机器上跑（Kaggle），本地只有浏览器和 numpy。cond 用随机张量就够——
这一步量的是速度，不是质量。

**P1 · DiT 端到端。** 导出 → 量化 → 浏览器跑 2 步 → 拿到 latent → 和 Kaggle 上那份逐元素比。
验收：latent cosine ≥ 0.99，时间落进预算。

**P2 · 解码器。** octree 8 层循环用 JS 驱动，Gaussian 头一张图。验收：从 P1 的 latent 出 PLY，
和 k2n 那六张比顶点数、opacity 均值。这一步顺便量一下「Gaussian 解码器 16 块减到 8 块」
损失多少——它占 160 MB 下载，也占 1.4 s。

**P3 · 前端蒸馏，这是唯一需要新训练的部分。**
- DINO → 学生：输入 1024²，输出 4101×1280，对着教师做 cosine 回归。目标 20 M 参数级。
- VAE → 学生：输出 4096 个 token。注意两件事：上游是**随机**编码（吃 generator），蒸馏要对齐均值那一支；
  通道数上游自己就对不上——代码里 unshuffle 出来是 256，`FLOW_MODEL_ARGS` 写 `cond2_channels=128`，
  动手前先把这条核实。
- 验收两层：特征 cosine ≥ 0.98，以及最终 PLY 的接触表由人看。

**P4 · 抠图。** 目标是把 Swin-L + 可变形卷积换成一个能在 WGSL 里写清楚的网。
最简单的办法是先问一句：输入图带 alpha 时能不能整段跳过？能就跳过。

**P5 · 拼装。** 五段串起来，一页应用，量真实端到端。这时才第一次知道 60 秒够不够。

**P6 · 发布。** HF 模型仓放 ONNX + JS + 脚本 + 卡片；数字全部用实测的。

---

## 五、验收线

| 项 | 线 |
|---|---|
| 端到端（M 系列 Mac，Chrome） | ≤ 60 s |
| 首屏下载 | ≤ 300 MB（见下面的账） |
| 数值 | 每一段的 int4/fp16 vs 教师 cosine ≥ 0.99；前端特征 ≥ 0.98 |
| 观感 | 和教师 20 步的接触表并排，人看 |

**下载预算的账**（int4 按 0.5625 B/参数）：

| 部件 | 参数 | int4 体积 |
|---|---|---|
| DiT（已做） | 370 M | 209 MB |
| octree 解码 49 M + Gaussian 解码 235 M | 284 M | 160 MB |
| DINO 学生（目标 20 M） | 20 M | 11 MB |
| VAE 学生（fp16） | 5 M | 10 MB |
| 抠图 | 5 M | 3 MB |
| **合计** | | **393 MB** |

超了 93 MB。**Gaussian 解码器比 DiT 只小一点**，16 块 × 14.7 M = 235 M，是下一个该蒸的地方：
降到 8 块省 65 MB，顺带把 1.4 s 减半。不做这一步，300 MB 这条线过不去。

---

## 六、风险

- **注意力是架构的分水岭。** 上面第 2 节 a/b/c 没量出来之前，不要写第二行代码。
- **下载预算的大头在两个解码器，不在前端。** DiT 209 MB + 解码器 160 MB 已经 369 MB，
  前端三个学生挤在剩下的一点点里。想守住 300 MB，先动 Gaussian 解码器的块数（见第五节的账）。
- **VAE 的卷积是纯算力**，1024² 的第一层一步就是 310 GFLOP。蒸馏到更浅的网络
  比量化有用得多——量化省的是带宽，卷不动这块。
- **可变形卷积搬不动**。BiRefNet 的解码器用了它，ORT-web 和 WGSL 都没有现成实现，
  所以 P4 是换网，不是移植。
- **octree 每一层都要重读 8192 个 latent**（我们测过：kv 投影 3.1 ms + cross-attn 6.6 ms，
  与点数无关）。搬进浏览器后这笔固定开销依然在，8 层 ≈ 80 ms，把 kv 缓存到层间就能省掉。
- **许可**：蒸馏并重分发上游模型的输出，取决于 TripoSplat、DINOv3、FLUX.2、BiRefNet 各自的条款。
  我们只发自己蒸馏出来的图，上游权重不进仓库；但「蒸馏权重算不算衍生作品」这条要提前问清楚。
