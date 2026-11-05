# site — 翻面的前端

一个不做构建、不引 CDN 以外的依赖、不连后端的静态站点。打开就是四段流程。

```sh
# 一条命令：在仓库根起服务，打开带 ?local=../w/&ort=/w/ort/ 的页面（Ctrl-C 停）
./scripts/browser/serve_site.sh

# 或者自己起（要 ../assets 里的示例图和夹具图，所以根目录起，不是 site/）
python3 -m http.server 8811 --bind 127.0.0.1
open http://127.0.0.1:8811/site/
```

## 看一堆 PLY：`ply-gallery.html`

导出目录里几个、几十个 PLY 时，一个一个改 `?ply=` 太慢。这条命令列出全部并附上原图：

```sh
# 先给目录生成清单（读同目录的 report.json 取 opacity / 高斯数）
python3 scripts/make_ply_manifest.py out/teacher-2026-10-29

# 再打开画廊
GALLERY=out/teacher-2026-10-29 ./scripts/browser/serve_site.sh
```

左边是列表（缩略图 + 变体 + opacity + 高斯数），右边一个能转的球：拖动旋转、
滚轮缩放、`←` / `→` 换图、空格开关自转。地址里的 `?i=N` 可以直达第 N 个。
换目录只要重跑这两条命令，页面不改。

`ply-test.html` 那次是单张、固定机位、给截图用的；画廊是给人看的，两者不互相替代。

模型默认从 HF 取（huggingface.co → hf-mirror.com）。本机两个域名不通时用
`?local=../w/` 从仓库里的 `w/` 读，文件名会被压平：

```
dit.k2n.onnx  octree_level.int4.onnx  gs.int4.onnx  decoder_meta.json
cond_feature1.npy  cond_feature2.npy  noise_0.npy  noise_1.npy
```

## 四段

| 段 | 文件 | 干什么 |
|---|---|---|
| 权重 | `src/pipeline.js` + `src/hf.js` + `src/loader.js` | 首屏真读 4 个图 + 4 个夹具，526 MB，字节进进度条。HF 先试、镜像兜底，字节进 Cache API。格子按**字节**翻面 |
| 编码 | `src/pipeline.js` | 第一次丢图时懒加载 3 个 ONNX（约 197 MB fp32）：RMBG 抠图 → 裁切贴黑底 → DINO + VAE → 条件 |
| 投图 | `src/main.js` | 拖入 / 点击 / 粘贴。`createImageBitmap` 本地解码 |
| 思考 | `src/think.js` | Canvas2D 的像素阶梯：分辨率 20→126、色阶 5→8，随进度往上爬。**全程零 WebGPU 提交** |
| 结果 | `src/pipeline.js` + `src/decoder.js` + `src/viewer.js` | 4 步 int4 DiT → 八叉树 8 层 → 高斯头 → 65,536 个高斯上屏 |

## 输入

`<input accept="image/*">`，拖拽和粘贴只看 `type` 是不是以 `image/` 开头，所以格式的
上限就是这个浏览器 `createImageBitmap` 能解的那一批：**JPEG、PNG、WebP、AVIF、
GIF（首帧）、BMP**。WebP 是支持的，带不带 alpha 都行 —— 2026-10-29 用一张 853×1280 的
WebP 跑过整条链，结果和同一张 JPEG 一致。Chrome 解不了 HEIC/HEIF（iPhone 直接导出的
那种），那一路会落到 `fail()`，不会静默出错。

- 尺寸随意：短边缩到 1024、边长吸到 32 的倍数；EXIF 方向由浏览器自己摆正。
- 图自带 alpha（PNG / WebP 透明底）就跳过抠图，直接拿它算 1.2× 方形裁切框；
  alpha 全是 255 的等于没有，仍然走 RMBG。
- 白底、单人、侧身的效果最好；背景越花，RMBG 越容易把杂物留在 alpha 里。
- 模型输出是 **Y 向下**的：上游 `save_ply` 只做 `(x, z, -y)`，不改变上下。viewer 在
  `setSplat()` 里统一翻一次（`toViewFrame`），PLY 和实时解码两条路都走这里。
  `ply-test.html?raw=1` 可以看文件原样。2026-10-29 之前所有渲染都是倒立的，包括线上页面。
- 抠图之后按 k3 的配方裁切（2026-11-02 修正）：裁切框看腐蚀 1 px 的 alpha，
  bbox 用 `alpha > 8`。老师用 BiRefNet，背景直接是 0，所以 k3 用 `> 0`；RMBG
  背景有 1–3/255 的薄雾，`> 0` 会把框撑到整帧。框心压住 bbox，超出画面的部分
  留黑，**不往画面里夹** —— 夹框会把人物推离训练位置：hf_0000 上横竖各偏
  ~100 px，DINO cos 从 0.997 掉到 0.84。先裁 RGBA、放大到 1024²，再贴黑底。
  乘务员那张 600×855 WebP 上对真 BiRefNet 教师画布：旧配方 DINO 0.972 /
  VAE 0.735，修正后 0.994 / 0.966，画布 MAE 3.3/255。
- `?in=512` 只把**照片先缩到短边 512**，抠图/裁切都在这个尺度上算；最后仍画回
  1024² 再进 DINO/VAE —— 那两个 ONNX 的输入形状和输出 token 数（4101 / 4096）是定死的，
  改不了。所以这不是"喂 512 给模型"，是"少采一半细节再放大回模型要的尺寸"。
  2026-10-30 在 af2 / 820 / hf_0000 上量过：opacity 0.600→0.494、0.633→0.486、
  0.551→0.505，**更糊更淡**，不是优化。留着只为以后学生换成 512 输入时做对照。

## 推理那一段是真的

`src/pipeline.js` 在启动时下载并建好 3 个 ORT-web 会话（DiT / octree / gs），
丢图之后：

1. 4 步采样：t=1000 → 900 → 750 → 500 → 0（k12 原生网格，无 CFG），`latent -= v * dt`；
2. 八叉树 8 层循环在 JS 里驱动单层 ONNX 图（采样表达不了，所以循环在 JS）；
3. 高斯头出一张 `[1,2048,480]`，`buildGaussians` 合成 16 个 float 一组的高斯。

4 步末 latent 与教师 20 步的 cos 0.684（见 `PUBLISH.md`）；解码器的验收是同一个
latent 解出来的 opacity / scale / bbox 落在 Kaggle 那几次解码的散布里。

## 编码器也是真的（按需）

`loadCondition(bitmap)` 在第一次用到时下载三个 ONNX（`idefinity/flip-student-2step`）：

1. 短边缩到 1024、边长吸到 32 的倍数；自带 alpha 就跳过抠图；
2. 否则 `rmbg.onnx`（RMBG-1.4）出 alpha；腐蚀 1 px、从 alpha 求 1.2× 方形框；
3. 先裁 RGBA、放大到 1024²，再贴到黑底（k3 的顺序），`rgb/255` 进 DINO、`rgb*2-1` 进 VAE；
4. 得到 `feature1 [1,4101,1280]`、`feature2 [1,4096,128]`，交给 4 步 DiT。

DINO / VAE 两个学生来自 `scripts/k3_encoder_distill.py`（DINOv3 ViT-H / Flux2-VAE 当教师）。
抠图不是学生：2026-11-02 用完整 BiRefNet ONNX 对拍，RMBG 的 soft IoU 0.99，
裁切框差 2–4 px。

特征 cos 0.81（DINO）、0.97（VAE），latent 与教师比 cos 0.80–0.88（`assets/k3/e2e.json`）。
VAE 基本透明，差距都在 DINO。

编码器取不到或跑不动时直接报 `编码器不可用：…`，不拿夹具冒充这一趟的结果。

## 三条设计约束

1. **图片不出浏览器。** 网络请求只有模型（含编码器）与夹具；用户选的图走 `File` → `createImageBitmap`。
2. **思考态让出 GPU。** `think.js` 只有 Canvas2D；这时设备队列里一个任务都没有。
3. **没有构建步骤。** 原生 ES module + 一个 CSS 文件。

## 参数

- `?local=<base>` — 从本地目录读模型、夹具与编码器（文件名压平），到不了 HF 时用。
- `?ort=<base>` — 换一份 ONNX Runtime 1.31.1 的 `dist/`（默认 jsDelivr）。到不了 CDN 时用；
  `.mjs` 和 `.wasm` 必须同版本。
- `?nogpu=1` — 假装没有 WebGPU，走降级路径。
- 模型取不到时直接报 `启动失败：…`，不换夹具、不换预烘焙结果。
