# 计划：TripoSplat 的 WebGPU 发行版

目标：浏览器里丢一张图，出 65536 个高斯，60 秒以内，首屏下载不超过 600 MB。

上游是 PyTorch + CUDA 的参考实现。我们要造的是它的另一个发行版：**五段全部蒸馏、全部量化，
以 ONNX 为主干，跑在 onnxruntime-web 的 WebGPU EP 上**。仓库里放的是 ONNX 图、一个 JS 运行时、
以及把上游蒸馏成这些图的脚本。

---

## 零、2026-10-29 的实际位置（数字都是量出来的）

| 段 | 状态 | 证据 |
|---|---|---|
| DiT 蒸馏 + int4 | **完成** | 2 步 4.03 s（T4）；发布的 safetensors 反量化回模型 0 missing / 0 unexpected |
| DiT 端到端出 PLY | **完成** | 65536 个高斯；latent vs 教师 20 步 cos 0.684（fp32 学生 0.682） |
| 解码器 ONNX | 导出完成 | octree 一层（静态 2048，padding 免费）+ 高斯头 int4 150 MB |
| DiT 在浏览器里跑 | **完成** | 融合成 28 个 `MultiHeadAttention` 的 294 MB 图，会话 978 ms；2 步采样（apple metal-3） |
| 编码器（DINO / VAE / 抠图） | **完成**：三个学生都训完，e2e 跑过 | DINO 40 ep cos **0.811**、VAE 20 ep cos **0.967**、matte 20 ep（bce 0.363）；ONNX fp32 77.6 / 31.3 / 11.1 MB，parity **1.000000**；已发 `idefinity/flip-student-2step` |
| 整条链（浏览器，真编码器） | **完成** | 56.3 s 出 65,536 个高斯：装 1.5 / 编码 25.5 / 采样 23.9 / 解码 5.5 |
| 权重分发 | 可用 | 仓已公开；浏览器先试 huggingface.co，失败换 hf-mirror.com |

**解码那一半已经在浏览器里跑通了**（`webgpu/e2e.html`，metal-3，用 Kaggle 那一跑的
latent 当输入）：八叉树 8 层 3.87 s + 高斯头 2.43 s = 6.3 s，出 65536 个高斯，
opacity 0.4213 / scale 0.00581 / bbox x[-0.25, 0.30] —— 和 Kaggle 上同一个 latent
解出来的 PLY（opacity 0.392–0.420、bbox x[-0.254, 0.302]）对得上，差异落在解码器
自己那 4 次解码的散布里。

**后半段已经整条接通**（`site/index.html?local=../w/`）：页面加载 4 个图 + 4 个夹具
（526 MB，Cache API 留底），丢照片后真跑 2 步 DiT + 八叉树 + 高斯头，65,536 个高斯
直接上屏。浏览器里算出来的 opacity 0.4090 / scale 0.00598 / bbox x[-0.251, 0.300]，
落在上面那组参考值里。

**关于"25.2 s/步"**：那是冷启动第一次提交的数字，含着着色器编译。稳态下两步一共
23.9 s（见下面 2026-10-29 的整条链计时），所以别拿 25.2 去乘步数算总时长。

**编码器这一半也接通了**（2026-10-28 起头，10-29 在 Kaggle T4 收尾）：
`scripts/k3_encoder_distill.py` 把 BiRefNet / DINOv3 ViT-H / Flux2-VAE 蒸成三个学生
（ConvNeXt 19.4 M、卷积 VAE 7.8 M、U-Net 2.8 M），导出 fp32 ONNX 与教师对拍 cos 1.000。浏览器端 `site/src/pipeline.js` 新增
`loadCondition(bitmap)`：短边 1024 / 吸到 32、抠图、腐蚀 1 px、1.2× 方形裁切、贴黑底，
再跑 DINO + VAE 得到 `feature1 [1,4101,1280]` / `feature2 [1,4096,128]`。编码器**按需加载**
（约 120 MB），首屏仍是原来的 526 MB；取不到就退回夹具，页脚写明。

**整条链在浏览器里跑完了**（2026-10-29，M 系列 Mac / Chrome 156 / apple metal-3，
真编码器不是夹具）：56.3 s 出 65,536 个高斯 —— 装图 1.5 s、编码 25.5 s、采样 23.9 s、
解码 5.5 s；opacity 0.3501 / scale 0.00466 / bbox x[-0.344, 0.324]。
60 s 的线守住了，余量 3.7 s。

**e2e 的数字要说清楚它测的是什么。** 用教师的 DiT 和教师解码器，只把条件换成学生的
（4 张图，`encoder/e2e.json`）：特征 cos 0.797–0.859（DINO）、0.953–0.978（VAE）；
两步之后的 latent 与教师条件的 latent 比，cos **0.800–0.884**。换 DINO 条件、
保留教师 VAE 条件时是 0.796–0.883 —— 两个数几乎一样，说明差距全在 DINO 那一支，
VAE 学生基本是透明的。这就是 0.811 的特征 cos 在端到端上的样子：不是坏的，
但离"和教师分不出"还差一截。

**这一版发布的东西**：`matte_student.onnx` 换成 20 ep 的（11,064,381 B，
sha256 `e2842c50…`），dino / vae 两张图重导后与线上字节完全一致（`c12170f8…` /
`ddfc0e40…`），`encoder_meta.json` 一个字节没变，所以站点里的 `ENCODER_FILES`
不用动。

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

## 二·零 已经做出来的东西

`dit.int4.onnx` 已经在仓库里（294 MB，6265 节点）。导出脚本 `scripts/k2p_onnx_export.py`
从 `k2e/student.pt`（1,482,301,473 字节）装进上游的 `LatentSeqMMFlowModel`，追踪一遍，
再用 `MatMulNBitsQuantizer(bits=4, block_size=32, is_symmetric=True)` 量化。

| | fp32 | int4 |
|---|---|---|
| 体积 | 1517 MB | **294 MB** |
| 节点 | 6265 | 6265 |
| 算子 | 259 MatMul + 3 Gemm | **203 `com.microsoft::MatMulNBits`** + 56 MatMul + 3 Gemm |

**留下的 56 个 MatMul 全是注意力**：28 层（24 主 block + 2 noise refiner + 2 context refiner）
各自的 QKᵀ 和 PV。它们没有权重，量化不到，也正是跑得最慢的那部分（406 对 1789）。

静态形状，输入是编码器原始输出：
`latent [1,8192,16]`、`camera [1,1,5]`、`feature1 [1,4101,1280]`、`feature2 [1,4096,128]`、`t [1]`。

导出踩到的四个坑，都在脚本注释里：

1. `aten::polar` 没有 ONNX symbolic，opset 17 也没有复数 → 卡在 `RePo3DRotaryEmbedding`，
   还没碰到矩阵乘。改成实数版（乘 e^{iθ} 就是 2×2 旋转），与复数路径对拍 2.4e-7。
2. 编码器给 VAE 特征补的 5 个零 token 在模型外面 → 烘进图里，输入契约变成「原始 VAE 输出」。
3. 必须静态导出：`F.layer_norm(h, h.shape[-1:])` 在 dynamic_axes 下变成 Gather → ListConstruct，
   导出器说 "not constant"。
4. 不能按猜的长度预分配权重文件：torch 的 ckpt 是 zip，目录在末尾，错了就报 "corrupted"。

**没做的验证**：这张图在 CPU 上没有端到端跑过（一次前向 5 分钟，内核撑不住）。
int4 的证据是矩阵乘级别的（cosine 0.997、rel_l2 0.073）。

**2026-10-28 补上的验证与修正。** 这张图在 CPU 上端到端跑过了（78 s/次），
顺手查出原来的量化有两个洞：

1. `MatMulNBitsQuantizer` 把 K<32 的层也量化了 —— `input_layer`（K=16）和
   `cam_refiner.mlp.0`（K=5）被塞进一个 32 宽的块，第一个线性层就丢了 6.7% 的信号；
2. 2-D 输入的 Linear 导出成 `Gemm`，量化器不碰它，所以 `t_embedder` / `adaLN_modulation`
   在 ONNX 里是 fp32、在发布的 int4 pack 里是 4 bit —— 两个"同一个模型"其实不是同一个。

`scripts/k2w_onnx_pack.py` 用 `assets/k2n/model.safetensors` 重建这张图：K<32 的层
退回 fp32 MatMul，其余 `MatMulNBits` 装 pack 的 nibble，所有 `f.*` initializer 用 pack
的值覆盖。之后逐层对 torch（同一份 pack）：cam_refiner rel 1e-7、最深的 block 3.7e-3；
CPU 上 1 步速度 cos 0.999992、2 步末 latent cos 0.999955（对 k2u 夹具）。
浏览器里 2 步 cos 0.9933 —— 差的那一点是 WebGPU 的累加顺序，不是图本身。

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

**还没量的**：fp16 激活（q4f16）那一档。

**已经量到的（2026-10-28，浏览器里）**：`dit.int4.onnx` 在 ORT-web 上 2.5 s 建好会话，
输入名对得上，然后 `OrtRun` 死在 **SafeInt 整数溢出** —— 注意力被物化成
`[1, 16, 12294, 12294]` fp32 = **9.66 GB**。第 2 条不是"慢"这一层的问题，是**跑不起来**。

**后来解决了**：`scripts/k2v_fuse_attention.py` 把 28 个注意力块换成
`com.microsoft::MultiHeadAttention`（scale 从两个 Mul 挪进 op 属性，算术不变），
score 矩阵不再物化。同一张图在 ORT-web 1.31.1 / apple metal-3 上：会话 978 ms，
2 步采样 25.2 s/步，2 步末 latent vs 夹具 cos 0.9933。WGSL 注意力不再是必需品。

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
| 首屏下载 | ≤ 600 MB（见下面的账） |
| 数值 | 每一段的 int4/fp16 vs 教师 cosine ≥ 0.99；前端特征 ≥ 0.98 |
| 观感 | 和教师 20 步的接触表并排，人看 |

**下载预算的账**（int4 按 0.5625 B/参数；实测列是本机文件的真实字节）：

| 部件 | 参数 | int4 预算 | 实测 |
|---|---|---|---|
| DiT（已做） | 370 M | 209 MB | **294 MB**（`dit.k2n.onnx`） |
| octree 解码（已做） | 49 M | 28 MB | **58 MB** |
| Gaussian 解码（已做） | 235 M | 132 MB | **150 MB** |
| 条件夹具（退回用，懒加载） | — | — | **24 MB** |
| DINO 学生（19.4 M） | 20 M | 11 MB | **77.6 MB** fp32 / 38.8 fp16 |
| VAE 学生（7.8 M） | 5 M | 10 MB | **31.3 MB** fp32 / 15.7 fp16 |
| 抠图（2.8 M） | 5 M | 3 MB | **11.1 MB** fp32 / 5.6 fp16 |
| **合计** | | **393 MB** | **646 MB** fp32 / **586 MB** fp16（含夹具） |

首屏还是 **526 MB + 约 30 MB 前端**：编码器（120 MB fp32 / 60 MB fp16）改成按需加载，
只有真的丢图时才取；夹具留在首屏（没它就没有退回条件）。于是 600 MB 的首屏线守住了；
整场下载 fp16 编码器约 586 MB。
**Gaussian 解码器 16→8 块仍是可选优化**（省 75 MB、解码时间减半）；真要走回 300 MB，
再单独开一档。


---

## 六、风险

- **注意力是架构的分水岭。** 上面第 2 节 a/b/c 没量出来之前，不要写第二行代码。
- **下载预算的大头是两个解码器，不在前端。** 实测 DiT 294 MB + 解码器 208 MB = 502 MB。
  预算已放宽到 600 MB，所以 16→8 块只是优化项；若以后要压回 300 MB，先动它（见第五节的账）。
- **VAE 的卷积是纯算力**，1024² 的第一层一步就是 310 GFLOP。蒸馏到更浅的网络
  比量化有用得多——量化省的是带宽，卷不动这块。
- **可变形卷积搬不动**。BiRefNet 的解码器用了它，ORT-web 和 WGSL 都没有现成实现，
  所以 P4 是换网，不是移植。
- **octree 每一层都要重读 8192 个 latent**（我们测过：kv 投影 3.1 ms + cross-attn 6.6 ms，
  与点数无关）。搬进浏览器后这笔固定开销依然在，8 层 ≈ 80 ms，把 kv 缓存到层间就能省掉。
- **许可**：蒸馏并重分发上游模型的输出，取决于 TripoSplat、DINOv3、FLUX.2、BiRefNet 各自的条款。
  我们只发自己蒸馏出来的图，上游权重不进仓库；但「蒸馏权重算不算衍生作品」这条要提前问清楚。

---

## 七、断点（2026-10-29，可恢复）

**已发布**（HF `idefinity/flip-student-2step`，根目录）：

| 文件 | 字节 | sha256 | 来自 |
|---|---|---|---|
| `dino_student.onnx` | 77,557,789 | `c12170f8…` | DINO 40 ep，cos 0.811 |
| `vae_student.onnx` | 31,328,848 | `ddfc0e40…` | VAE 20 ep，cos 0.967 |
| `matte_student.onnx` | 11,064,381 | `e2842c50…` | matte **20 ep**（bce 0.363） |
| `encoder_meta.json` | 965 | — | shapes / params / preprocess |

dino / vae 重导之后与线上旧版逐字节相同，`encoder_meta.json` 也一个字节没变 ——
所以这一版实际只覆盖了 `matte_student.onnx`。

**encoder/ 下多的三样**：

- `encoder/e2e.json` —— 4 张图的教师-学生对照（cos1 / cos2 / latent cos / opacity / scale / bbox）。
- `encoder/checkpoints/matte_full_e20.pt`（33,180,883，带 optimizer，epoch 20）——
  再训就从这里接着来，不用从头 7 ep 的两个多小时。
- `encoder/code/k3_encoder_distill.py`、`encoder/code/k3_kaggle_driver.py` ——
  训练脚本与那一跑的驱动器（见下）。

**这一跑是怎么跑的**：Kaggle T4（15 GB），`kkonda/flip-k3-encoder`，一次提交跑完六个阶段，
57 分钟：

```
restore   从 HF 取 inputs.tgz + 三个断点（走 huggingface.co，不是镜像）
setup     clone TripoSplat @ d8db9e0 + 4.46 GB 教师权重
prep      200 张图 9.83 s/张 → 33 min
train     matte 7 → 20，13 ep × 74 s ≈ 16 min，acc 0.5299 → 0.6372
export    三张 ONNX + 与 ORT CPU 对拍，parity 1.000000
e2e       教师 DiT 2 步 + 教师解码器，条件换成学生的（4 张图）
```

复现只要两句：notebook 里下载 `encoder/code/k3_kaggle_driver.py` 再执行它。
换机器时唯一的坑是 **不要在 Kaggle 上设 `HF_ENDPOINT`** —— 镜像的 HEAD 少了
`x-repo-commit`，`snapshot_download` 会直接报 "Distant resource does not seem to be
on huggingface.co"；那是这台 Mac 才需要的设置，Kaggle 直连 HF 是通的（0.2 s）。

**本机浏览器复现**（这台 Mac 的代理白名单里没有 huggingface.co，也没有
cdn.jsdelivr.net，所以得先把字节放到本地）：

```sh
# 一次性：把 4 个 ONNX 放进 w/，把 ONNX Runtime 1.31.1 的 dist 放进 w/ort/
npm pack onnxruntime-web@1.31.1     # 或从 unpkg 取那 7 个文件
python3 -m http.server 8899 --bind 127.0.0.1
# Chrome 带 --no-sandbox（这台机器的沙箱起不来），打开：
#   site/index.html?local=../w/&ort=/w/ort/
```

`?ort=` 是这次加的参数：默认还是 jsDelivr，给一个目录就用那份 dist。
用 `site/index.html?local=../w/` 一把跑完（截图与计时都来自这条路）。

**没做的**：fp16 编码器（约 60 MB，能把 120 MB 的懒加载砍一半，也能把编码那 25 s 压下去）；
DINO 学生加大或换 ViT（0.811 → ~0.85 的猜测，没试）；对 **学生条件** 微调 2 步 DiT
（真正能补上域差的那一步）；观感验收 —— 还没有跟教师 20 步并排看过接触表，
截图里那团高斯偏暗、形状不实，和 0.80–0.88 的 latent cos 是一致的。
token 只在本地用了一次（`key.secret`，已 gitignore）；不要把 token 写进任何提交。
