// 三段 WGSL：加载动画、舞台背景、高斯泼溅。
// 都只用一个全屏三角形或实例化四边形，没有顶点缓冲。

import { FULLSCREEN_VS } from './gpu.js';

/* ------------------------------------------------------------------ 加载
   一格一格的"卡片"按进度翻面。翻过来的那一面亮，没翻的是暗的。
   进度从左上往右下扫，所以看一眼就知道走到哪了——这是唯一的加载指示，
   没有转圈的菊花。 */

export const LOADER_WGSL = /* wgsl */ `
struct Load { p: f32, t: f32, wpx: f32, hpx: f32 };
@group(0) @binding(0) var<uniform> u: Load;

fn hash21(p: vec2<f32>) -> f32 {
  var q = fract(p * vec2<f32>(123.34, 345.45));
  q = q + vec2<f32>(dot(q, q + vec2<f32>(45.32)));
  return fract(q.x * q.y);
}

fn sdBox(p: vec2<f32>, b: vec2<f32>) -> f32 {
  let d = abs(p) - b;
  return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

${FULLSCREEN_VS}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let res = vec2<f32>(u.wpx, u.hpx);
  let uv = fc.xy / res;
  let cell = 76.0;
  let grid = max(res / cell, vec2<f32>(1.0));
  let g = floor(fc.xy / cell);
  let f = fract(fc.xy / cell) - 0.5;

  // 没翻的只有一根极淡的边，翻过来的也只是微微发暖；
  // 真正亮的是"正在翻"的那一条波前。整块屏幕不该跟进度条抢注意力。
  let h = hash21(g + vec2<f32>(0.0, 7.0));
  let sweep = g.x / max(grid.x - 1.0, 1.0);
  let order = clamp(sweep * 0.76 + h * 0.24, 0.0, 1.0);
  let t = smoothstep(order, order + 0.05, u.p);
  let front = exp(-pow((order - u.p) / 0.055, 2.0));
  let ang = t * 3.14159265;
  let sx = max(abs(cos(ang)), 0.07);
  let q = vec2<f32>(f.x / sx, f.y);

  let d = sdBox(q, vec2<f32>(0.30, 0.30));
  let inside = 1.0 - smoothstep(-0.012, 0.012, d);
  let din = sdBox(q, vec2<f32>(0.285, 0.285));
  let ring = inside * smoothstep(-0.008, 0.008, din);

  var col = vec3<f32>(0.036, 0.034, 0.030) * (0.7 + 0.5 * abs(cos(ang)));
  col = col + vec3<f32>(0.085, 0.062, 0.030) * t;
  col = col + vec3<f32>(0.80, 0.58, 0.31) * front * 0.42;

  let edgeCol = mix(vec3<f32>(0.085, 0.082, 0.074), vec3<f32>(0.40, 0.31, 0.18), t);
  col = mix(col, edgeCol, ring * 0.72);
  col = col + vec3<f32>(0.9, 0.68, 0.38) * ring * front * 0.5;

  // 背景
  let r = length((uv - vec2<f32>(0.5)) * vec2<f32>(1.0, 1.35));
  var bg = vec3<f32>(0.048, 0.046, 0.039) * (0.30 + 0.70 * (1.0 - smoothstep(0.0, 0.9, r)));

  var outc = mix(bg, col, inside);

  let vig = 1.0 - 0.5 * smoothstep(0.3, 1.05, length(uv - vec2<f32>(0.5)));
  outc = outc * vig;
  let grain = (hash21(fc.xy + fract(u.t) * 61.0) - 0.5) * 0.01;
  return vec4<f32>(max(outc + grain, vec3<f32>(0.0)), 1.0);
}
`;

/* ------------------------------------------------------------------ 舞台
   转台 + 接触阴影 + 渐晕。全部按屏幕空间的椭圆算，
   因为相机只在水平面上绕，椭圆形状是可预测的。 */

export const BACKDROP_WGSL = /* wgsl */ `
struct U {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  cam: vec4<f32>,       // eye.xyz, tanHalfFovY
  viewport: vec4<f32>,  // w, h, 1/w, 1/h
  ring: vec4<f32>,      // cx, cy, rx, ry  （屏幕像素）
  misc: vec4<f32>,      // focal, 环内比例, 阴影强度, 曝光
};
@group(0) @binding(0) var<uniform> u: U;

${FULLSCREEN_VS}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let res = u.viewport.xy;
  let uv = fc.xy / res;

  // 背板：上浅下深，像一块打光的纸
  var col = mix(vec3<f32>(0.085, 0.081, 0.069), vec3<f32>(0.030, 0.029, 0.024),
                smoothstep(0.05, 0.95, uv.y));

  // 转台：一块微微暖起来的台面，比背景亮一点，好让暗部的高斯有边界
  let c = u.ring.xy;
  let rad = max(u.ring.zw, vec2<f32>(1.0));
  let dr = (fc.xy - c) / rad;
  let rr = length(dr);

  let floorMask = smoothstep(1.02, 0.97, rr);
  col = mix(col, col * vec3<f32>(1.34, 1.30, 1.16), floorMask);

  // 接触阴影：一圈宽的 + 一块贴地的实心，东西才像坐在台子上
  let e = dr * vec2<f32>(1.0, 1.55);
  let wide = exp(-1.5 * dot(e, e));
  let core = exp(-9.0 * dot(e, e));
  let sh = clamp((wide * 0.55 + core * 0.55) * u.misc.z, 0.0, 0.94);
  col = mix(col, vec3<f32>(0.004, 0.004, 0.003), sh);

  // 环线：外侧一道细边，内侧一道更淡的
  let band = smoothstep(0.01, 0.0, abs(rr - 1.0));
  col = mix(col, vec3<f32>(0.52, 0.48, 0.40), band * 0.55);
  let band2 = smoothstep(0.006, 0.0, abs(rr - 0.94));
  col = mix(col, vec3<f32>(0.28, 0.26, 0.22), band2 * 0.4);

  // 渐晕
  let v = length((uv - vec2<f32>(0.5)) * vec2<f32>(1.0, 1.25));
  col = col * (1.0 - 0.66 * smoothstep(0.30, 1.05, v));

  return vec4<f32>(col * u.misc.w, 1.0);
}
`;

/* ------------------------------------------------------------------ 高斯

   每个高斯一个实例，四个顶点拼一个四边形。
   协方差走标准 EWA：Σ' = J W Σ Wᵀ Jᵀ，然后 2×2 特征分解出两个轴。
   混合是前乘 alpha、从远到近——顺序由 CPU 的计数排序给。 */

export const SPLAT_WGSL = /* wgsl */ `
struct U {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  cam: vec4<f32>,
  viewport: vec4<f32>,
  ring: vec4<f32>,
  misc: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> splats: array<vec4<f32>>;   // 16 floats each, 4 vec4
@group(0) @binding(2) var<storage, read> order: array<u32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) opacity: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var out: VSOut;
  let base = order[ii] * 4u;
  let a = splats[base];        // xyz, opacity
  let b = splats[base + 1u];   // rgb
  let s = splats[base + 2u];   // scale (已取 exp)
  let q = splats[base + 3u];   // quat wxyz

  let p = a.xyz;

  // 四元数 → 旋转（按列给，WGSL 的 mat3x3 构造按列）
  let qw = q.x; let qx = q.y; let qy = q.z; let qz = q.w;
  let c0 = vec3<f32>(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy));
  let c1 = vec3<f32>(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx));
  let c2 = vec3<f32>(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));

  let m0 = c0 * s.x;
  let m1 = c1 * s.y;
  let m2 = c2 * s.z;

  // Σ 的三列（对称）
  let sig0 = vec3<f32>(dot(m0, m0), dot(m1, m0), dot(m2, m0));
  let sig1 = vec3<f32>(dot(m0, m1), dot(m1, m1), dot(m2, m1));
  let sig2 = vec3<f32>(dot(m0, m2), dot(m1, m2), dot(m2, m2));

  let tv = u.view * vec4<f32>(p, 1.0);
  let z = -tv.z;

  let clip = u.viewProj * vec4<f32>(p, 1.0);
  if (z < 0.03 || clip.w <= 0.0) {
    out.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);   // 丢掉
    out.local = vec2<f32>(0.0);
    out.color = vec3<f32>(0.0);
    out.opacity = 0.0;
    return out;
  }

  // J = [[f/z, 0, -f·x/z²], [0, f/z, -f·y/z²]]，W 取视矩阵的左上 3×3
  let focal = u.misc.x;
  let invz = 1.0 / z;
  let jx = vec3<f32>(focal * invz, 0.0, -focal * tv.x * invz * invz);
  let jy = vec3<f32>(0.0, focal * invz, -focal * tv.y * invz * invz);

  let w0 = u.view[0].xyz;
  let w1 = u.view[1].xyz;
  let w2 = u.view[2].xyz;
  let t0 = vec3<f32>(dot(jx, w0), dot(jx, w1), dot(jx, w2));
  let t1 = vec3<f32>(dot(jy, w0), dot(jy, w1), dot(jy, w2));

  // C = T · Σ，cov = C · Tᵀ
  let cr0 = vec3<f32>(dot(t0, sig0), dot(t0, sig1), dot(t0, sig2));
  let cr1 = vec3<f32>(dot(t1, sig0), dot(t1, sig1), dot(t1, sig2));
  var ca = dot(cr0, t0) + 0.3;   // 低通，防止亚像素高斯闪烁
  let cb = dot(cr0, t1);
  var cc = dot(cr1, t1) + 0.3;

  let mid = 0.5 * (ca + cc);
  let det = ca * cc - cb * cb;
  let rad = sqrt(max(mid * mid - det, 0.0));
  let l1 = mid + rad;
  let l2 = max(mid - rad, 0.0);

  var v1 = vec2<f32>(1.0, 0.0);
  if (abs(l1 - ca) > 1e-6) {
    v1 = normalize(vec2<f32>(cb, l1 - ca));
  }
  let v2 = vec2<f32>(-v1.y, v1.x);

  var r1 = 2.0 * sqrt(l1);
  var r2 = 2.0 * sqrt(l2);

  // 贴着视线看向一个薄面时，2×2 协方差会炸出一大片四边形。
  // 限一下半径，同时把局部坐标一起缩，形状才不会被拉长。
  let maxR = max(u.viewport.y * 0.28, 8.0);
  let k1 = min(1.0, maxR / max(r1, 1e-4));
  let k2 = min(1.0, maxR / max(r2, 1e-4));
  r1 = r1 * k1;
  r2 = r2 * k2;

  let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
  let off = v1 * (corner.x * r1) + v2 * (corner.y * r2);
  let ndcOff = off * u.viewport.zw * 2.0;

  out.pos = vec4<f32>(clip.xy + ndcOff * clip.w, clip.z, clip.w);
  out.local = corner * 2.0;
  out.color = b.rgb;
  out.opacity = a.w;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.local, in.local);
  let alpha = in.opacity * exp(-0.5 * r2);
  if (alpha < 0.004) { discard; }
  return vec4<f32>(in.color * alpha, alpha);
}
`;
