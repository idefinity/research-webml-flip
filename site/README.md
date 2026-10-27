# site — 翻面的前端

一个不做构建、不引 CDN、不连后端的静态站点。打开就是四段流程。

```sh
# 在仓库根目录起（要 ../assets 里的示例图与 PLY）
python3 -m http.server 8811 --bind 127.0.0.1
open http://127.0.0.1:8811/site/
```

## 四段

| 段 | 文件 | 干什么 |
|---|---|---|
| 权重 | `src/weights.js` + `src/loader.js` + `LOADER_WGSL` | 真的读 `assets/k2n/int4.bin`：209,276,416 字节，559 个张量，边读边验。格子按**字节**翻面，一条亮带扫过去。这是唯一可以随便用 GPU 的时刻 |
| 投图 | `src/main.js` | 拖入 / 点击 / 粘贴。`createImageBitmap` 本地解码，一次网络请求都没有 |
| 思考 | `src/think.js` | Canvas2D 的像素阶梯：分辨率 20→126、色阶 5→8，随进度往上爬。**全程零 WebGPU 提交** |
| 结果 | `src/viewer.js` + `src/ply.js` | 高斯泼溅：EWA 协方差、2σ 四边形、CPU 计数排序、前乘 alpha |

## 权重那一段是真的

`src/weights.js` 流式读 `assets/k2n/int4.bin`，用 `manifest.json` 的偏移做三件事：

1. 进度 = 已读字节 ÷ 209,276,416。不是计时器。本机 850 MB/s 时 0.25 秒读完，所以扫过速度有个 1.5 秒的下限，但**显示的数字永远不超过真实已读字节**。
2. 读的过程中报出当前张量名：`blocks.15.attn.qkv`、`context_refiner.0.mlp.mlp.0`……顺序就是文件里的物理顺序。
3. 每个张量读完就抽查：int4 的 fp16 scale（必须是有限正数）、码值 word（不能是空的 `0x00000000` / `0xffffffff`）、fp16 张量的值。抽 4,577 个点，任何一个不对，`data-weights` 就变成 `check` 并把行标红。

`manifest.json` 的账：**204 个 int4 张量**（369.95 M 参数，176.4 MB 码值 + 22.1 MB scale，11.56 M 个 scale）、**355 个 fp16 张量**（bias / norm gamma / freqs / shift_table，合计 1.1 MB）。约 200 MiB 的东西里，88% 是 4 bit 码值。

`?weights=<url>` / `?manifest=<url>` 可以换源（比如以后指向 HF）。文件不在时（发布时 200 MB 不进 git）整段退化成占位进度，页面照常走完。

三条设计约束写在代码注释里，这里重复一次，因为改的人容易破坏：

1. **图片不出浏览器。** 唯一的网络请求是示例图/PLY 的 `fetch`。用户选的图走 `File` → `createImageBitmap`。
2. **思考态让出 GPU。** `think.js` 只有 Canvas2D；这时设备队列里一个任务都没有。这是"把性能留给推理"的具体含义。
3. **没有构建步骤。** 原生 ES module + 一个 CSS 文件。加依赖会破坏第一条以外的所有约定，尤其是一条命令就能跑。

## 参数

- `?speed=N` — 把演示推理的时间轴压到 1/N，截图和自动检查用。
- `?nogpu=1` — 假装没有 WebGPU，走降级路径（加载动画退成 CSS，结果页给文字）。没有它这条分支只能靠换浏览器才测得到。
- `?weights=<url>` — 换权重源。默认 `../assets/k2n/int4.bin`。
- `?manifest=<url>` — 换 manifest 源。

## 接真推理

现在已经有一半是真的：权重真的读了 209 MB，还验过。缺的是把它们变成数字。

接缝两处：

```js
// src/weights.js —— 已经有了。读出来的 result.bytes 就是 209 MB 的 Uint8Array，
// 下一步该做的是按 manifest 切块、上传成 GPUBuffer。
streamWeights({ manifestUrl, binUrl, onProgress })

// src/pipeline.js —— 还是演示。runInference 只推进时间，不动数据。
runInference({ bitmap, onProgress })
```

接上采样之后把 `MODE` 改成 `'live'`：页脚的"演示管线"会自动消失。UI 一行都不用动。

## 现在缺什么

- **权重只到位了，还没上 GPU。** 209 MB 在 JS 堆里躺着（`window.__flipWeights.bytes`），没有 `GPUBuffer`，也没有一次 matmul。
- **结果是预烘焙的。** 任何输入都会落到 `assets/k2g/building_stone_house_t20.ply`。页脚写明了这一点，别删那行字。
- 只有一个示例，因为仓库里只有一对"同一张输入 ↔ 同一个 PLY"。
- 整个 902 MB 的预算里，只有 DiT 的 209 MB 打包了；DINOv3、VAE、八叉树、高斯解码器都还没有 int4 产物。
- `*.ply` 和 `assets/k2n/*.bin` 都在 `.gitignore` 里。上线时要从 HF 取。
- 移动端只保证不崩，没调过手感。
