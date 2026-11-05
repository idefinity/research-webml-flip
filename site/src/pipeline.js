// 管线。前半段是编码器（抠图 / DINOv3 / Flux2-VAE，按需加载），后半段是
// 4 步 int4 DiT → 八叉树 + 高斯头 → splat。
//
// 编码器第一次用到某张图时才下载（约 197 MB，含 88 MB 的 RMBG 抠图），之后进
// Cache。取不到或跑不动就直接报错，不拿夹具冒充这一趟的结果；只有显式的 fix=1
// 才走夹具。
//
// 模型默认从 HF 取（huggingface.co → hf-mirror.com），?local=../w/ 改读本地目录；
// 图字节进 Cache API，第二次打开不重下。

import { hfFetch } from './hf.js';
import { createSession, ort } from './onnx.js';
import { parseNPY } from './npy.js';
import { sampleOctree, buildGaussians } from './decoder.js';

export const MODE = 'real';

export const CONDITION_IMAGE = '../assets/k2u/hf_0000.jpg';
export const CONDITION_NOTE = '编码器按需加载：首次约 197 MB，之后走缓存';

const DIT_REPO = 'idefinity/flip-student-2step';
const WEB_REPO = 'idefinity/flip-web';
const ENC_REPO = 'idefinity/flip-student-2step';

// bytes 是本机量的实际长度，只用于把进度条按真实总量摊开。
// ?dit=<file> 换 DiT 图做 A/B（配合 ?local=../w/ 使用），换别的文件只会
// 让进度条略微偏一点，所以表里没有的名字就退回出厂长度。
const DIT_BYTES = {
  'dit.k12.int4.mha.onnx': 233692741,
  'dit.k12.int4.mha16.onnx': 233705212,
  'dit.k12.int4.q4f16.onnx': 227978045,
  'dit.k12.int4.nbits16.onnx': 227965437,
  'dit.k12.int4.b32.onnx': 268375301,
  'dit.k12.int4.b32n16.onnx': 245306645,
};
const DIT_FILE = new URLSearchParams(location.search).get('dit') || 'dit.k12.int4.mha.onnx';

const FILES = [
  { key: 'dit', repo: DIT_REPO, file: DIT_FILE, bytes: DIT_BYTES[DIT_FILE] || 233692741 },
  { key: 'octree', repo: WEB_REPO, file: 'decoder/octree_level.int4.onnx', bytes: 57871367 },
  { key: 'gs', repo: WEB_REPO, file: 'decoder/gs.int4.onnx', bytes: 150445946 },
  { key: 'meta', repo: WEB_REPO, file: 'decoder/decoder_meta.json', bytes: 3946, json: true },
  { key: 'feature1', repo: WEB_REPO, file: 'k2u/cond_feature1.npy', bytes: 20997248 },
  { key: 'feature2', repo: WEB_REPO, file: 'k2u/cond_feature2.npy', bytes: 2099840 },
  { key: 'noise0', repo: WEB_REPO, file: 'k2u/noise_0.npy', bytes: 524416 },
  { key: 'noise1', repo: WEB_REPO, file: 'k2u/noise_1.npy', bytes: 148 },
];

const ENCODER_FILES = [
  { key: 'dino', file: 'dino_student.onnx', bytes: 77557789 },
  { key: 'vae', file: 'vae_student.onnx', bytes: 31328848 },
  { key: 'matte', file: 'rmbg.onnx', bytes: 88217533 },
  { key: 'emeta', file: 'encoder_meta.json', bytes: 1044, json: true },
];

export const ARTIFACT_BYTES = FILES.reduce((a, f) => a + f.bytes, 0);
export const ENCODER_BYTES = ENCODER_FILES.reduce((a, f) => a + f.bytes, 0);

export const STAGES = [
  { key: 'encode', label: '编码 · 抠图 + DINO + VAE', from: 0.0, to: 0.18 },
  { key: 'sample', label: '采样 · 4 步 int4', from: 0.18, to: 0.8 },
  { key: 'decode', label: '解码 · 八叉树 + 高斯', from: 0.8, to: 1.0 },
];

export const SAMPLES = [
  { id: 'fuliji', label: '示例照片', image: CONDITION_IMAGE },
];

export function stageAt(p) {
  for (let i = STAGES.length - 1; i >= 0; i--) {
    if (p >= STAGES[i].from) return STAGES[i];
  }
  return STAGES[0];
}

function localBase() {
  return new URLSearchParams(location.search).get('local') || '';
}

function localName(file, base) {
  return base ? file.replace(/.*\//, '') : file;
}

/** main.js 的 signal 是个只有 aborted 的壳；fetch 只认真 AbortSignal。 */
function netSignal(signal) {
  return signal && typeof signal.addEventListener === 'function' ? signal : null;
}

// ---------------------------------------------------------------- 模型加载

let S = null;

export function modelsReady() {
  return !!S;
}

export function encoderReady() {
  return !!(S && S.encoder);
}

/**
 * 下载 4 个图 + 4 个夹具，建 3 个会话。onProgress 报的是字节：
 * { file, loaded, total, p, host, rate, phase }。
 */
export async function initModels({ onProgress, signal } = {}) {
  if (S) return S;
  const base = localBase();
  const net = netSignal(signal);
  const total = ARTIFACT_BYTES;
  const got = {};
  let done = 0;
  let topRate = 0;

  for (const f of FILES) {
    const r = await hfFetch({
      repo: f.repo,
      file: localName(f.file, base),
      localBase: base,
      signal: net,
      onProgress(e) {
        if (e.rate > topRate) topRate = e.rate;
        onProgress && onProgress({
          file: f.file, loaded: done + (e.read || 0), total,
          p: Math.min(1, (done + (e.read || 0)) / total),
          host: e.host, rate: e.rate || topRate, phase: e.phase,
        });
      },
    });
    got[f.key] = r.bytes;
    done += r.bytes.length;
    onProgress && onProgress({
      file: f.file, loaded: done, total, p: done / total,
      host: r.host, rate: r.rate, phase: 'done',
    });
  }

  onProgress && onProgress({ file: DIT_FILE, loaded: total, total, p: 1, phase: 'session' });
  const dit = await createSession(got.dit, { label: 'dit' });
  const octree = await createSession(got.octree, { label: 'octree' });
  const gs = await createSession(got.gs, { label: 'gs' });

  const dec = new TextDecoder();
  const npy = (k) => parseNPY(got[k].buffer);

  S = {
    dit, octree, gs,
    meta: JSON.parse(dec.decode(got.meta)),
    feature1: npy('feature1'),
    feature2: npy('feature2'),
    noise0: npy('noise0'),
    noise1: npy('noise1'),
    encoder: null,
    encoderErr: null,
    o: await ort(),
    host: base ? 'local' : 'hf',
  };
  return S;
}

// --------------------------------------------------------------- 编码器

let encoderPromise = null;
let encoderFailed = false;

async function buildEncoderSession(o, bytes, label) {
  try {
    return await createSession(bytes, { label, providers: ['webgpu'] });
  } catch (e) {
    console.warn(`[flip] ${label} webgpu 建会话失败，退回 wasm`, e);
    return createSession(bytes, { label, providers: ['wasm'] });
  }
}

/**
 * 懒加载编码器。失败只记一次，之后直接返回 null（调用方报错，不再退回夹具）。
 * onProgress 报 { file, loaded, total, p, host, phase }。
 */
export async function ensureEncoder({ onProgress, signal } = {}) {
  const s = await initModels({ signal });
  if (s.encoder) return s.encoder;
  if (encoderFailed) return null;
  if (!encoderPromise) {
    encoderPromise = (async () => {
      const base = localBase();
      const net = netSignal(signal);
      const total = ENCODER_BYTES;
      const got = {};
      let done = 0;
      for (const f of ENCODER_FILES) {
        const r = await hfFetch({
          repo: ENC_REPO,
          file: localName(f.file, base),
          localBase: base,
          signal: net,
          onProgress(e) {
            onProgress && onProgress({
              file: f.file, loaded: done + (e.read || 0), total,
              p: Math.min(1, (done + (e.read || 0)) / total),
              host: e.host, rate: e.rate, phase: 'encoder',
            });
          },
        });
        got[f.key] = r.bytes;
        done += r.bytes.length;
      }
      onProgress && onProgress({ file: 'encoder', loaded: total, total, p: 1, phase: 'session' });
      const o = await ort();
      const dino = await buildEncoderSession(o, got.dino, 'dino');
      const vae = await buildEncoderSession(o, got.vae, 'vae');
      const matte = await buildEncoderSession(o, got.matte, 'matte');
      s.encoder = {
        dino, vae, matte, o,
        meta: JSON.parse(new TextDecoder().decode(got.emeta)),
      };
      return s.encoder;
    })()
      .catch((e) => {
        encoderFailed = true;
        s.encoderErr = e;
        console.warn('[flip] 编码器不可用', e);
        return null;
      })
      .finally(() => { encoderPromise = null; });
  }
  return encoderPromise;
}

// ------------------------------------------------------- 图片 → 条件

const CANVAS = 1024;

// ?in=N 只改「照片先缩到短边 N」这一步：matte / 连通域 / 裁切框都在 N 尺度上算，
// 最后仍画回 1024² 画布 —— DINO / VAE 的 ONNX 输入形状是 [1,3,1024,1024] 定死的，
// 输出 token 数也定死（4101 / 4096）。所以这不是"真的喂 512 给模型"，
// 而是"少采 4 倍细节、再放大回模型要的尺寸"。输出高斯数不变。
// 用来回答：把输入细节砍一半，对当前这套糊的链路是帮忙还是帮倒忙。
function inputSize() {
  const n = Number(new URLSearchParams(location.search).get('in'));
  return Number.isFinite(n) && n >= 256 && n <= CANVAS ? Math.round(n / 32) * 32 : CANVAS;
}
const IN_SIZE = inputSize();

function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function context2d(cv) {
  return cv.getContext('2d', { willReadFrequently: true });
}

/** 和 scripts/k3_encoder_distill.py 的 resize_min_side 对齐：短边 1024，边长吸到 /32。 */
function resizeDims(w, h, target = CANVAS) {
  const s = target / Math.min(w, h);
  return {
    w: Math.max(32, Math.round((w * s) / 32) * 32),
    h: Math.max(32, Math.round((h * s) / 32) * 32),
  };
}

/** 3×3 最小值滤波，等价上游的 MinFilter(3) 腐蚀。 */
function erodeAlpha(a, w, h, r = 1) {
  const out = new Uint8ClampedArray(a.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = 255;
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const v = a[yy * w + xx];
          if (v < m) m = v;
        }
      }
      out[y * w + x] = m;
    }
  }
  return out;
}

/**
 * 从 alpha 求正方形裁切框；空 alpha 就取居中最大正方形。
 *
 * 阈值 8 是给 RMBG 的软背景留的余量：老师用的 BiRefNet 背景直接是 0，k3 里
 * 用 alpha>0 求框；RMBG 在背景里留下 1–3/255 的薄雾，>0 会把框撑到整帧
 * （hf_0000 上 1.2 倍后 1843²，老师是 1255²）。阈值 8 之后两边 bbox 差 2–4 px。
 */
function squareCropBox(a, w, h, expand = 1.2, thr = 8) {
  let x0 = w;
  let x1 = -1;
  let y0 = h;
  let y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (a[y * w + x] > thr) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) {
    const side = Math.min(h, w);
    const sx = (w - side) / 2;
    const sy = (h - side) / 2;
    return [sx, sy, side, side].map((v) => Math.round(v));
  }
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const half = (Math.max(x1 - x0, y1 - y0) / 2) * expand;
  // 和 k3 的预处理一致：框心压在人物 bbox 上，超出画面的部分留黑，不往画面里夹。
  // 夹框会把人物推离训练时的位置：hf_0000 上横竖各偏 ~100 px，DINO cos 掉到 0.84。
  return [cx - half, cy - half, 2 * half, 2 * half].map((v) => Math.round(v));
}

/** RGBA ImageData → CHW float32，亮度可带缩放/偏移。 */
function toCHW(img, mul = 1, add = 0) {
  const { width: w, height: h, data } = img;
  const n = w * h;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const j = i * 4;
    out[i] = data[j] * mul + add;
    out[n + i] = data[j + 1] * mul + add;
    out[2 * n + i] = data[j + 2] * mul + add;
  }
  return out;
}

/**
 * 图片 → feature1 / feature2。返回 null 表示没有编码器。
 * progress(p) 的 p 是编码阶段内部的 0..1。
 */
export async function loadCondition(bitmap, { progress = null, signal = null } = {}) {
  const s = await initModels({ signal });
  const enc = s.encoder || (await ensureEncoder({
    signal,
    onProgress(e) {
      progress && progress(0.02 + 0.53 * Math.min(1, e.p || 0));
    },
  }));
  if (!enc) return null;
  progress && progress(0.58);
  const o = s.o;

  const { w: w2, h: h2 } = resizeDims(bitmap.width, bitmap.height, IN_SIZE);
  const src = makeCanvas(w2, h2);
  const sctx = context2d(src);
  // k3 用 LANCZOS 缩到短边 1024，浏览器里最接近的是 high（bicubic 级）
  sctx.imageSmoothingQuality = 'high';
  sctx.drawImage(bitmap, 0, 0, w2, h2);
  const img = sctx.getImageData(0, 0, w2, h2);
  const n = w2 * h2;

  // 输入自带 alpha 就不跑抠图；否则用 RMBG 估。
  const alpha = new Uint8ClampedArray(n);
  let intrinsic = false;
  for (let i = 0; i < n; i++) {
    alpha[i] = img.data[i * 4 + 3];
    if (alpha[i] < 250) intrinsic = true;
  }
  if (!intrinsic) {
    // RMBG-1.4 固定 1024² 输入，[0,1] 再减 0.5（mean=.5 std=1）；
    // 输出 alpha 是 1024²，双线性放回 w2×h2 再求裁切框。
    const MS = 1024;
    const msrc = makeCanvas(MS, MS);
    const mctx = context2d(msrc);
    mctx.drawImage(src, 0, 0, MS, MS);
    const input = toCHW(mctx.getImageData(0, 0, MS, MS), 1 / 255, -0.5);
    const out = await enc.matte.run({
      [enc.matte.inputs[0]]: new o.Tensor('float32', input, [1, 3, MS, MS]),
    });
    const a = out[enc.matte.outputs[0]].data;
    const acv = makeCanvas(MS, MS);
    const actx = context2d(acv);
    const aimg = actx.createImageData(MS, MS);
    for (let i = 0; i < MS * MS; i++) {
      const v = a[i] * 255;
      aimg.data[i * 4] = v;
      aimg.data[i * 4 + 1] = v;
      aimg.data[i * 4 + 2] = v;
      aimg.data[i * 4 + 3] = 255;
    }
    actx.putImageData(aimg, 0, 0);
    const acv2 = makeCanvas(w2, h2);
    const actx2 = context2d(acv2);
    actx2.drawImage(acv, 0, 0, w2, h2);
    const adata = actx2.getImageData(0, 0, w2, h2).data;
    for (let i = 0; i < n; i++) alpha[i] = adata[i * 4];
  }
  progress && progress(0.82);

  for (let i = 0; i < n; i++) img.data[i * 4 + 3] = alpha[i];
  sctx.putImageData(img, 0, 0);

  // 和 k3 的预处理一致：裁切框看腐蚀 1px 的 alpha（只用求框），合成用的是
  // 原始软 alpha；先裁 RGBA、放大到 1024²，再贴到黑底上。
  const er = erodeAlpha(alpha, w2, h2, 1);
  const [sx, sy, sw, sh] = squareCropBox(er, w2, h2, 1.2, 8);
  const crop = makeCanvas(Math.max(1, sw), Math.max(1, sh));
  const cropCtx = context2d(crop);
  cropCtx.drawImage(src, -sx, -sy);

  const rs = makeCanvas(CANVAS, CANVAS);
  const rctx = context2d(rs);
  rctx.imageSmoothingQuality = 'high';
  rctx.drawImage(crop, 0, 0, crop.width, crop.height, 0, 0, CANVAS, CANVAS);

  const cv = makeCanvas(CANVAS, CANVAS);
  const cctx = context2d(cv);
  cctx.fillStyle = '#000';
  cctx.fillRect(0, 0, CANVAS, CANVAS);
  cctx.drawImage(rs, 0, 0);
  const canvasImg = cctx.getImageData(0, 0, CANVAS, CANVAS);
  progress && progress(0.86);

  // 调试口子：把这一趟真正喂给 DiT 的东西留在 window 上。结果不像照片时，
  // 先看这两张，再看模型 —— __flipCond.png().canvas 是 1024² 黑底裁切，
  // .mask 是抠图 alpha（原始尺寸）。图里还有天空和石头，就是前半段的问题；
  // 如果是一张干净的人像贴在黑底上，那问题在后面。
  if (typeof window !== 'undefined') {
    const debugCanvas = () => {
      const c = document.createElement('canvas');
      c.width = CANVAS;
      c.height = CANVAS;
      c.getContext('2d').drawImage(cv, 0, 0);
      return c;
    };
    window.__flipCond = {
      width: w2, height: h2, intrinsic,
      crop: [sx, sy, sw, sh],
      alpha,
      png() {
        const mask = document.createElement('canvas');
        mask.width = w2;
        mask.height = h2;
        const mctx = mask.getContext('2d');
        const im = mctx.createImageData(w2, h2);
        for (let i = 0; i < w2 * h2; i++) {
          im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = alpha[i];
          im.data[i * 4 + 3] = 255;
        }
        mctx.putImageData(im, 0, 0);
        return { canvas: debugCanvas().toDataURL('image/png'),
                 mask: mask.toDataURL('image/png') };
      },
    };
  }

  const rgb = toCHW(canvasImg, 1 / 255, 0);
  const dinoOut = await enc.dino.run({
    image: new o.Tensor('float32', rgb, [1, 3, CANVAS, CANVAS]),
  });
  const f1 = dinoOut[enc.dino.outputs[0]].data;
  progress && progress(0.95);

  const vaeIn = new Float32Array(rgb.length);
  for (let i = 0; i < rgb.length; i++) vaeIn[i] = rgb[i] * 2 - 1;
  const vaeOut = await enc.vae.run({
    image: new o.Tensor('float32', vaeIn, [1, 3, CANVAS, CANVAS]),
  });
  const f2 = vaeOut[enc.vae.outputs[0]].data;
  progress && progress(1);

  return {
    feature1: f1,
    feature2: f2,
    width: w2,
    height: h2,
    intrinsicAlpha: intrinsic,
  };
}

// ------------------------------------------------------------------ 推理

/**
 * 采样网格。k12 学生是在原生网格 t=[1000, 900, 750, 500] 上蒸的：归一化后
 * 1 → 0.9 → 0.75 → 0.5 → 0，dt 0.1 / 0.15 / 0.25 / 0.5，无 CFG。图里的 t
 * 输入吃 1000×t，这里给的是归一化值。其他步数没有训练过的 schedule，
 * ?steps=N 只作对照；默认 4 步走原生网格。
 */
export function sampleSteps() {
  const n = Number(new URLSearchParams(location.search).get('steps'));
  return Number.isFinite(n) && n >= 1 && n <= 8 ? Math.round(n) : 4;
}

const NATIVE_GRID = [[1, 0.9], [0.9, 0.75], [0.75, 0.5], [0.5, 0]];

function stepGrid(n) {
  if (n === NATIVE_GRID.length) return NATIVE_GRID;
  const out = [];
  for (let i = 0; i < n; i++) out.push([1 - i / n, 1 - (i + 1) / n]);
  return out;
}

/**
 * 跑一遍：编码（可选）+ N 步采样 + 八叉树 + 高斯头。
 * 返回 buildGaussians 的结果，和 loadPLY 的形状一样，viewer 直接吃。
 */
export async function runInference({ bitmap = null, onProgress, signal } = {}) {
  const s = await initModels({ signal });
  const aborted = () => {
    if (signal && signal.aborted) throw new Error('aborted');
  };

  let f1 = null;
  let f2 = null;
  let encoded = false;

  if (bitmap) {
    const cond = await loadCondition(bitmap, {
      signal,
      progress(p) {
        onProgress && onProgress(0.18 * p, STAGES[0]);
      },
    });
    if (!cond) {
      const why = s.encoderErr ? (s.encoderErr.message || String(s.encoderErr)) : '未知原因';
      throw new Error(`编码器不可用：${why}`);
    }
    f1 = cond.feature1;
    f2 = cond.feature2;
    encoded = true;
  }

  // 只有 fix=1（主流程传 bitmap=null）才会用夹具条件。
  if (!encoded) {
    f1 = s.feature1.data;                                       // [1,4101,1280]
    const f2raw = s.feature2.data;
    f2 = new Float32Array(4096 * 128);                          // 图里补那 5 个零 token
    f2.set(f2raw.subarray(5 * 128, 5 * 128 + 4096 * 128));
  }

  const lat = new Float32Array(s.noise0.data);                  // [1,8192,16]
  const cam = new Float32Array(5);
  cam.set(s.noise1.data.subarray(0, 5));

  const startP = encoded ? 0.18 : 0;
  const sampleSpan = (1 - startP) * 0.72;
  const steps = stepGrid(sampleSteps());
  for (let i = 0; i < steps.length; i++) {
    aborted();
    const [t, tPrev] = steps[i];
    const dt = t - tPrev;
    const out = await s.dit.run({
      latent: new s.o.Tensor('float32', lat, [1, 8192, 16]),
      camera: new s.o.Tensor('float32', cam, [1, 1, 5]),
      feature1: new s.o.Tensor('float32', f1, [1, 4101, 1280]),
      feature2: new s.o.Tensor('float32', f2, [1, 4096, 128]),
      t: new s.o.Tensor('float32', new Float32Array([1000 * t]), [1]),
    });
    const vl = out.out_latent.data;
    const vc = out.out_camera.data;
    for (let k = 0; k < lat.length; k++) lat[k] -= vl[k] * dt;
    for (let k = 0; k < cam.length; k++) cam[k] -= vc[k] * dt;
    onProgress && onProgress(startP + sampleSpan * ((i + 1) / steps.length), STAGES[1]);
  }

  aborted();
  const { points } = await sampleOctree({
    session: s.octree,
    cond: lat,
    numPoints: 2048,
    level: 8,
    pad: 2048,
    onLevel(l) {
      const q = startP + sampleSpan;
      onProgress && onProgress(q + (1 - q) * (l.level / 8), STAGES[2]);
    },
  });

  aborted();
  const out = await s.gs.run({
    x: new s.o.Tensor('float32', points, [1, 2048, 3]),
    cond: new s.o.Tensor('float32', lat, [1, 8192, 16]),
  });
  const splat = buildGaussians({
    features: out.features.data, points, meta: s.meta, numPoints: 2048,
  });
  onProgress && onProgress(1, STAGES[2]);
  // 真实输入走到这里 encoded 一定是 true；false 只会来自 fix=1。
  splat.encoded = encoded;
  return splat;
}

export async function releaseEncoder() {
  const enc = S && S.encoder;
  if (!enc) return;
  await Promise.allSettled([enc.dino.release(), enc.vae.release(), enc.matte.release()]);
  S.encoder = null;
}
