// 阶段 3：高斯泼溅。
//
// 一个 uniform、两块 storage：一块放高斯（每个 16 个 float），一块放排序索引。
// 每帧 CPU 只做一件重活——按视深做一次计数排序，O(n)，六万五千个点约 2 ms。
// 画法是标准 EWA：视空间协方差投影成 2×2，特征分解出两个轴，每根轴 2σ。
// 没有深度缓冲，顺序就是正确性。

import { makeModule } from './gpu.js';
import { BACKDROP_WGSL, SPLAT_WGSL } from './shaders.js';

const UNIFORM_BYTES = 192;
const NEAR = 0.02;
const FOV = (34 * Math.PI) / 180;

/* ----------------------------------------------------------------- 数学 */

function perspective(fovy, aspect, near, far, out = new Float32Array(16)) {
  const f = 1 / Math.tan(fovy / 2);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);
  out[11] = -1;
  out[14] = (far * near) / (near - far);
  return out;
}

function lookAt(eye, target, up, out = new Float32Array(16)) {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];
  let l = Math.hypot(zx, zy, zz) || 1;
  zx /= l; zy /= l; zz /= l;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz) || 1;
  xx /= l; xy /= l; xz /= l;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

function mul(a, b, out = new Float32Array(16)) {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

/* ----------------------------------------------------------------- 主体 */

export async function createViewer(gpu, canvas, opts = {}) {
  const { device } = gpu;
  const format = navigator.gpu.getPreferredCanvasFormat();

  const [splatModule, backModule] = await Promise.all([
    makeModule(device, SPLAT_WGSL, 'splat'),
    makeModule(device, BACKDROP_WGSL, 'backdrop'),
  ]);

  const uniform = device.createBuffer({
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });

  const splatPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex: { module: splatModule, entryPoint: 'vs' },
    fragment: {
      module: splatModule,
      entryPoint: 'fs',
      targets: [{
        format,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-strip' },
  });

  const backPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: backModule, entryPoint: 'vs' },
    fragment: {
      module: backModule,
      entryPoint: 'fs',
      targets: [{
        format,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const backBind = device.createBindGroup({
    layout: backPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniform } }],
  });

  const ctx = canvas.getContext('webgpu');
  ctx.configure({ device, format, alphaMode: 'opaque' });

  /* ---- 状态 ---- */

  let model = null;
  let splatBuf = null;
  let orderBuf = null;
  let splatBind = null;

  let order = null;
  let depth = null;
  let counts = null;

  const cam = {
    az: 0,
    el: 0.18,
    dist: 3,
    target: [0, 0, 0],
    radius: 1,
    fit: 1,
    floor: -1,
  };
  const home = { az: 3.35, el: 0.18, dist: 3 };

  let running = false;
  let dirty = true;
  let raf = 0;
  let lastInteract = performance.now();
  let lastFrame = performance.now();
  let fps = 0;
  let onAz = opts.onAzimuth || (() => {});
  let onStats = opts.onStats || (() => {});
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const view = new Float32Array(16);
  const proj = new Float32Array(16);
  const vp = new Float32Array(16);
  const uniformData = new Float32Array(UNIFORM_BYTES / 4);

  function fitDistance() {
    const w = Math.max(1, canvas.clientWidth);
    const h = Math.max(1, canvas.clientHeight);
    const hFov = 2 * Math.atan(Math.tan(FOV / 2) * (w / h));
    const fov = Math.min(FOV, hFov);
    return (cam.fit / Math.sin(fov / 2)) * 1.08;
  }

  function setSplat(m) {
    model = m;
    const bytes = model.count * 16 * 4;

    if (splatBuf && splatBuf.size < bytes) { splatBuf.destroy(); splatBuf = null; }
    if (!splatBuf) {
      splatBuf = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      splatBind = null;
    }
    device.queue.writeBuffer(splatBuf, 0, model.data);

    if (orderBuf && orderBuf.size < model.count * 4) { orderBuf.destroy(); orderBuf = null; }
    if (!orderBuf) {
      orderBuf = device.createBuffer({
        size: model.count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      splatBind = null;
    }

    if (!splatBind) {
      splatBind = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: splatBuf } },
          { binding: 2, resource: { buffer: orderBuf } },
        ],
      });
    }

    order = new Uint32Array(model.count);
    depth = new Float32Array(model.count);
    counts = new Uint32Array(65536);

    cam.target = [model.center[0], (model.min[1] + model.max[1]) / 2, model.center[2]];
    cam.radius = model.radius;
    cam.fit = model.fitRadius || model.radius;
    cam.floor = model.min[1];
    home.dist = cam.dist = fitDistance();
    cam.az = home.az;
    cam.el = home.el;
    dirty = true;
    onAz(0);
    if (typeof window !== 'undefined') window.__flipCam = cam;   // 调试用
  }

  function sort() {
    const { count, data } = model;
    const v = view;

    // 视线方向的深度：z = -(view * p).z
    let minD = Infinity;
    let maxD = -Infinity;
    for (let i = 0; i < count; i++) {
      const o = i * 16;
      const d = -(v[2] * data[o] + v[6] * data[o + 1] + v[10] * data[o + 2] + v[14]);
      depth[i] = d;
      if (d < minD) minD = d;
      if (d > maxD) maxD = d;
    }
    const span = maxD - minD || 1;
    const scale = 65535 / span;

    counts.fill(0);
    for (let i = 0; i < count; i++) {
      const k = ((maxD - depth[i]) * scale) | 0;   // 远的 k 小 → 先画
      depth[i] = k;
      counts[k]++;
    }
    let sum = 0;
    for (let k = 0; k < 65536; k++) { const c = counts[k]; counts[k] = sum; sum += c; }
    for (let i = 0; i < count; i++) order[counts[depth[i]]++] = i;

    device.queue.writeBuffer(orderBuf, 0, order);
  }

  function ringScreen(w, h) {
    // 转台圆心（地面）投到屏幕，再沿相机右方向量一个半径
    const base = [cam.target[0], cam.floor, cam.target[2]];
    const eye = eyePos();
    let fx = cam.target[0] - eye[0];
    let fy = cam.target[1] - eye[1];
    let fz = cam.target[2] - eye[2];
    const fl = Math.hypot(fx, fz) || 1;
    // 视线的水平投影绕 y 轴转 90° 就是右方向
    const rx = -fz / fl;
    const rz = fx / fl;
    const R = cam.radius * 1.18;

    const project = (p) => {
      const cx = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
      const cy = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
      const cw = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
      const iw = cw !== 0 ? 1 / cw : 0;
      return [(cx * iw * 0.5 + 0.5) * w, (0.5 - cy * iw * 0.5) * h];
    };

    const c0 = project(base);
    const c1 = project([base[0] + rx * R, base[1], base[2] + rz * R]);
    const rxp = Math.max(8, Math.hypot(c1[0] - c0[0], c1[1] - c0[1]));
    const ryp = Math.max(6, rxp * Math.sin(Math.max(cam.el, 0.05)));
    return { cx: c0[0], cy: c0[1], rx: rxp, ry: ryp };
  }

  function eyePos() {
    const ce = Math.cos(cam.el);
    return [
      cam.target[0] + cam.dist * ce * Math.sin(cam.az),
      cam.target[1] + cam.dist * Math.sin(cam.el),
      cam.target[2] + cam.dist * ce * Math.cos(cam.az),
    ];
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      dirty = true;
    }
  }

  function frame(now) {
    if (!running) return;
    raf = requestAnimationFrame(frame);
    if (!model) return;

    // 空闲自转：像展台上那盏灯，慢慢转
    if (!reduced && now - lastInteract > 2600) {
      cam.az += (now - lastFrame) * 0.00012;
      dirty = true;
    }
    const dt = now - lastFrame;
    lastFrame = now;

    if (!dirty) return;
    resize();
    if (canvas.width < 2) return;

    const w = canvas.width;
    const h = canvas.height;
    const eye = eyePos();
    lookAt(eye, cam.target, [0, 1, 0], view);
    perspective(FOV, w / h, NEAR, cam.dist * 4 + cam.radius * 4, proj);
    mul(proj, view, vp);

    sort();

    const ring = ringScreen(w, h);
    const focal = (0.5 * h) / Math.tan(FOV / 2);

    uniformData.set(vp, 0);
    uniformData.set(view, 16);
    uniformData[32] = eye[0]; uniformData[33] = eye[1]; uniformData[34] = eye[2]; uniformData[35] = 0;
    uniformData[36] = w; uniformData[37] = h; uniformData[38] = 1 / w; uniformData[39] = 1 / h;
    uniformData[40] = ring.cx; uniformData[41] = ring.cy; uniformData[42] = ring.rx; uniformData[43] = ring.ry;
    uniformData[44] = focal; uniformData[45] = 0.94; uniformData[46] = 0.8; uniformData[47] = 1.14;
    device.queue.writeBuffer(uniform, 0, uniformData);

    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0.05, g: 0.049, b: 0.043, a: 1 },
        storeOp: 'store',
      }],
    });

    pass.setPipeline(backPipeline);
    pass.setBindGroup(0, backBind);
    pass.draw(3);

    pass.setPipeline(splatPipeline);
    pass.setBindGroup(0, splatBind);
    pass.draw(4, model.count);

    pass.end();
    device.queue.submit([enc.finish()]);

    dirty = false;

    fps = fps ? fps * 0.9 + (1000 / Math.max(dt, 1)) * 0.1 : 1000 / Math.max(dt, 1);
    onStats({ fps, count: model.count });

    const deg = ((cam.az * 180) / Math.PI + 360) % 360;
    onAz(deg);
  }

  /* ---- 交互 ---- */

  let dragging = false;
  let px = 0;
  let py = 0;

  function down(e) {
    dragging = true;
    px = e.clientX;
    py = e.clientY;
    lastInteract = performance.now();
    canvas.setPointerCapture(e.pointerId);
  }
  function move(e) {
    if (!dragging) return;
    const dx = e.clientX - px;
    const dy = e.clientY - py;
    px = e.clientX;
    py = e.clientY;
    cam.az -= dx * 0.0072;
    cam.el = Math.max(-0.15, Math.min(0.62, cam.el + dy * 0.0038));
    lastInteract = performance.now();
    dirty = true;
  }
  function up(e) {
    dragging = false;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  }
  function wheel(e) {
    e.preventDefault();
    const base = fitDistance();
    cam.dist = Math.max(base * 0.45, Math.min(base * 3.2, cam.dist * Math.exp(e.deltaY * 0.0012)));
    lastInteract = performance.now();
    dirty = true;
  }
  function reset() {
    cam.az = home.az;
    cam.el = home.el;
    cam.dist = fitDistance();
    lastInteract = performance.now();
    dirty = true;
  }

  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('wheel', wheel, { passive: false });
  canvas.addEventListener('dblclick', reset);

  const ro = new ResizeObserver(() => { dirty = true; });
  ro.observe(canvas);

  return {
    setSplat,
    reset,
    cam,
    redraw() { dirty = true; },
    start() {
      if (running) return;
      running = true;
      lastFrame = performance.now();
      lastInteract = performance.now();
      dirty = true;
      raf = requestAnimationFrame(frame);
    },
    stop() {
      running = false;
      cancelAnimationFrame(raf);
    },
  };
}
