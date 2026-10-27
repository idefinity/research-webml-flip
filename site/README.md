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
| 权重 | `src/loader.js` + `src/shaders.js` 的 `LOADER_WGSL` | 全屏 WGSL 动画：格子按进度翻面，一条亮带扫过去。这是唯一可以随便用 GPU 的时刻 |
| 投图 | `src/main.js` | 拖入 / 点击 / 粘贴。`createImageBitmap` 本地解码，一次网络请求都没有 |
| 思考 | `src/think.js` | Canvas2D 的像素阶梯：分辨率 20→126、色阶 5→8，随进度往上爬。**全程零 WebGPU 提交** |
| 结果 | `src/viewer.js` + `src/ply.js` | 高斯泼溅：EWA 协方差、2σ 四边形、CPU 计数排序、前乘 alpha |

三条设计约束写在代码注释里，这里重复一次，因为改的人容易破坏：

1. **图片不出浏览器。** 唯一的网络请求是示例图/PLY 的 `fetch`。用户选的图走 `File` → `createImageBitmap`。
2. **思考态让出 GPU。** `think.js` 只有 Canvas2D；这时设备队列里一个任务都没有。这是"把性能留给推理"的具体含义。
3. **没有构建步骤。** 原生 ES module + 一个 CSS 文件。加依赖会破坏第一条以外的所有约定，尤其是一条命令就能跑。

## 参数

- `?speed=N` — 把演示推理的时间轴压到 1/N，截图和自动检查用。
- `?nogpu=1` — 假装没有 WebGPU，走降级路径（加载动画退成 CSS，结果页给文字）。没有它这条分支只能靠换浏览器才测得到。

## 接真推理

`src/pipeline.js` 是唯一的接缝，两个函数：

```js
loadWeights({ onProgress })            // 分档下载，字节数已经是真的
runInference({ bitmap, onProgress })   // 进度 0..1 + 当前阶段
```

接上真管线后把 `MODE` 改成 `'live'`：页脚的"演示管线"会自动消失。UI 一行都不用动。

## 现在缺什么

- **结果是预烘焙的。** 任何输入都会落到 `assets/k2g/building_stone_house_t20.ply`。页脚写明了这一点，别删那行字。
- 只有一个示例，因为仓库里只有一对"同一张输入 ↔ 同一个 PLY"。
- `*.ply` 在 `.gitignore` 里。仓库里这几个是本地文件，上线时要从 HF 取。
- 移动端只保证不崩，没调过手感。
