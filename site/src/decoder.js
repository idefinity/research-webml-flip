// 解码那一半：八叉树采样循环 + 高斯头 + 拼成可渲染的数组。
//
// 上游的 OctreeProbabilityFixedlenDecoder.sample 是一段循环：每层把当前层的
// 父体坐标喂给 transformer，拿到 8 个孩子的概率，按 systematic 采样决定每个
// 孩子分到几个点，然后进下一层。循环里的 token 数是变的（第 1 层 1 个，
// 第 8 层最多 2048 个），还带采样，这两样 ONNX 都表达不了，所以循环搬到 JS。
//
// 图里只剩"一层"：固定 2048 个 token，多余的位子填什么都不会影响真正用到的那几
// 行 —— 因为八叉树那个 transformer 是 cross-only（ModulatedTransformerCrossOnlyBlock
// 没有 self-attention，每个 token 只看 cond，不看邻居）。这个"填什么无所谓"
// 的说法在导出脚本里被量过（padding_max_abs），不是推算出来的。
//
// 最后一步 _build_gaussians 是把 480 个通道切成 xyz / dc / scaling / rotation /
// opacity / offset_scale 六段，各自过一遍激活。常量（layout、扰动表、bias）
// 是 buffer 不是权重，ONNX 不会带，得单独从 decoder_meta.json 读。

import { ort } from './onnx.js';

const C0 = 0.28209479177387814;

// softplus(0.004) 的反函数，和上游 scale_bias 的算法一致
const SCALE_BIAS = 0.004 + Math.log(-Math.expm1(-0.004));   // ≈ -5.5194
// sigmoid(0.1) 的反函数
const OPACITY_BIAS = Math.log(0.1 / 0.9);                   // ≈ -2.1972

function softplus(x) {
  return x > 20 ? x : Math.log1p(Math.exp(x));
}
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

/** 上游 sample_probs 的 systematic 分支。probs: 8 个数，n: 这一行要分几个点。 */
export function sampleProbsSystematic(probs, n, rand = Math.random) {
  const out = new Int32Array(8);
  if (n <= 0) return out;

  let sum = 0;
  for (let i = 0; i < 8; i++) sum += Math.max(0, probs[i]);
  const p = new Float64Array(8);
  if (sum <= 0) for (let i = 0; i < 8; i++) p[i] = 1 / 8;
  else for (let i = 0; i < 8; i++) p[i] = Math.max(0, probs[i]) / sum;

  const cdf = new Float64Array(8);
  let acc = 0;
  for (let i = 0; i < 8; i++) {
    acc = Math.min(acc + p[i], 1 - 1e-12);
    cdf[i] = acc;
  }

  const u0 = rand() / n;                    // 每行一个随机数，和 torch 一致
  for (let j = 0; j < n; j++) {
    const u = Math.min(u0 + j / n, 1 - 1e-12);
    let idx = 0;
    while (idx < 7 && cdf[idx] < u) idx++;  // torch.searchsorted 右侧插入点
    out[idx]++;
  }
  return out;
}

function softmaxRow(logits, row, temperature) {
  const o = row * 8;
  let mx = -Infinity;
  for (let i = 0; i < 8; i++) mx = Math.max(mx, logits[o + i] / temperature);
  const e = new Float64Array(8);
  let s = 0;
  for (let i = 0; i < 8; i++) { e[i] = Math.exp(logits[o + i] / temperature - mx); s += e[i]; }
  for (let i = 0; i < 8; i++) e[i] /= s;
  return { probs: e, logProb: (i) => Math.log(e[i] + 1e-20) };
}

/**
 * 跑八叉树的 8 层。cond 是 DiT 出来的 latent，[8192, 16]。
 * 返回 { points: Float32Array[numPoints*3], levels: [...] }。
 */
export async function sampleOctree({
  session, cond, numPoints = 2048, level = 8, temperature = 1.0,
  pad = 2048, rand = Math.random, onLevel = null,
}) {
  let coords = [0, 0, 0];                 // 展平成 [n*3]，第 1 层只有一个根
  let counts = [numPoints];
  const levels = [];

  const x = new Float32Array(pad * 3);
  const lArr = new BigInt64Array(1);
  const l2Arr = new BigInt64Array(1);
  l2Arr[0] = BigInt(numPoints);

  for (let lv = 1; lv <= level; lv++) {
    const resP = 1 << (lv - 1);
    const res = 1 << lv;
    const n = counts.length;

    x.fill(0);
    for (let i = 0; i < n; i++) {
      x[i * 3] = (coords[i * 3] + 0.5) / resP;
      x[i * 3 + 1] = (coords[i * 3 + 1] + 0.5) / resP;
      x[i * 3 + 2] = (coords[i * 3 + 2] + 0.5) / resP;
    }
    lArr[0] = BigInt(res);

    const o = await ort();
    const feeds = {
      x: new o.Tensor('float32', x, [1, pad, 3]),
      l: new o.Tensor('int64', lArr, [1]),
      cond: new o.Tensor('float32', cond, [1, 8192, 16]),
    };
    // l2 只在"还有第二路 level embedder"的图里存在；用不上的输入会被导出器丢掉
    if (!session.inputs || session.inputs.includes('l2')) {
      feeds.l2 = new o.Tensor('int64', l2Arr, [1]);
    }
    const t0 = performance.now();
    const out = await session.run(feeds);
    const logits = out.logits.data;       // [1, pad, 8]
    const ms = performance.now() - t0;

    const nextCoords = [];
    const nextCounts = [];
    for (let i = 0; i < n; i++) {
      const { probs } = softmaxRow(logits, i, temperature);
      const got = sampleProbsSystematic(probs, counts[i], rand);
      for (let c = 0; c < 8; c++) {
        if (got[c] <= 0) continue;
        const ci = c & 1;
        const cj = (c >> 1) & 1;
        const ck = (c >> 2) & 1;
        nextCoords.push(coords[i * 3] * 2 + ci,
          coords[i * 3 + 1] * 2 + cj,
          coords[i * 3 + 2] * 2 + ck);
        nextCounts.push(got[c]);
      }
    }
    coords = nextCoords;
    counts = nextCounts;
    levels.push({ level: lv, parents: n, kept: counts.length, ms });
    onLevel && onLevel({ level: lv, parents: n, kept: counts.length, ms });
  }

  // 最后一步：按每个体素分到的点数展开，再在体素里随机抖一下
  const res = 1 << level;
  const points = new Float32Array(numPoints * 3);
  let w = 0;
  for (let i = 0; i < counts.length && w < numPoints; i++) {
    for (let k = 0; k < counts[i] && w < numPoints; k++) {
      points[w * 3] = (coords[i * 3] + rand()) / res;
      points[w * 3 + 1] = (coords[i * 3 + 1] + rand()) / res;
      points[w * 3 + 2] = (coords[i * 3 + 2] + rand()) / res;
      w++;
    }
  }
  if (w !== numPoints) {
    throw new Error(`八叉树只填了 ${w} 个点，要 ${numPoints}`);
  }
  return { points, levels };
}

/**
 * 高斯头 + _build_gaussians，合成渲染要的数组。
 * features: [numPoints, 480]；points: [numPoints, 3]（0..1）。
 * 输出的 16 个数一组的布局和 ply.js 一致，所以可以直接喂给 viewer。
 */
export function buildGaussians({ features, points, meta, numPoints = 2048 }) {
  const ng = meta.gaussians_per_point;
  const L = meta.layout;
  const pert = meta.points_offset_perturbation;   // [ng, 3]
  const base = meta.base_offset_scale;
  const lr = meta.rep_config.lr;
  const perturbe = meta.rep_config.perturbe_size;

  const count = numPoints * ng;
  const data = new Float32Array(count * 16);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];

  let o = 0;
  for (let t = 0; t < numPoints; t++) {
    const f = t * meta.out_channels;
    const px = points[t * 3];
    const py = points[t * 3 + 1];
    const pz = points[t * 3 + 2];

    for (let g = 0; g < ng; g++) {
      // offset
      const os = softplus(features[f + L._offset_scale.range[0] + g] + base);
      let ox = features[f + L._xyz.range[0] + g * 3] * lr._xyz + pert[g * 3];
      let oy = features[f + L._xyz.range[0] + g * 3 + 1] * lr._xyz + pert[g * 3 + 1];
      let oz = features[f + L._xyz.range[0] + g * 3 + 2] * lr._xyz + pert[g * 3 + 2];
      ox = Math.tanh(ox) * 0.5 * perturbe * os;
      oy = Math.tanh(oy) * 0.5 * perturbe * os;
      oz = Math.tanh(oz) * 0.5 * perturbe * os;

      // 上游 PLY 里那一步默认变换 [[1,0,0],[0,0,-1],[0,1,0]] → (x, z, -y)
      const x = ox + px;
      const y = oz + pz;
      const z = -(oy + py);

      if (x < min[0]) min[0] = x;
      if (y < min[1]) min[1] = y;
      if (z < min[2]) min[2] = z;
      if (x > max[0]) max[0] = x;
      if (y > max[1]) max[1] = y;
      if (z > max[2]) max[2] = z;

      const dc = L._features_dc.range[0] + g * 3;
      const sc = L._scaling.range[0] + g * 3;
      const rt = L._rotation.range[0] + g * 4;
      const op = L._opacity.range[0] + g;

      let qw = features[f + rt] * lr._rotation + 1;
      let qx = features[f + rt + 1] * lr._rotation;
      let qy = features[f + rt + 2] * lr._rotation;
      let qz = features[f + rt + 3] * lr._rotation;
      const qlen = Math.hypot(qw, qx, qy, qz) || 1;

      data[o] = x;
      data[o + 1] = y;
      data[o + 2] = z;
      data[o + 3] = sigmoid(features[f + op] * lr._opacity + OPACITY_BIAS);
      data[o + 4] = clamp01(0.5 + C0 * features[f + dc] * lr._features_dc);
      data[o + 5] = clamp01(0.5 + C0 * features[f + dc + 1] * lr._features_dc);
      data[o + 6] = clamp01(0.5 + C0 * features[f + dc + 2] * lr._features_dc);
      data[o + 7] = 0;
      data[o + 8] = softplus(features[f + sc] * lr._scaling + SCALE_BIAS);
      data[o + 9] = softplus(features[f + sc + 1] * lr._scaling + SCALE_BIAS);
      data[o + 10] = softplus(features[f + sc + 2] * lr._scaling + SCALE_BIAS);
      data[o + 11] = 0;
      data[o + 12] = qw / qlen;
      data[o + 13] = qx / qlen;
      data[o + 14] = qy / qlen;
      data[o + 15] = qz / qlen;
      o += 16;
    }
  }

  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const radius = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2 || 1;
  const fitRadius = 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
  const ys = new Float32Array(count);
  for (let i = 0; i < count; i++) ys[i] = data[i * 16 + 1];
  ys.sort();
  const floor = ys[Math.min(count - 1, Math.floor(count * 0.02))];

  return { count, data, min, max, center, radius, fitRadius, floor };
}
