// 阶段 0：权重加载。
// 这段是整条链路里唯一可以"随便用 GPU"的时刻，所以给它一个真的 GPU 动画，
// 而不是一条 CSS 进度条。像素层面的成本：一次全屏三角 + 三十来条 ALU。

import { makeModule } from './gpu.js';
import { LOADER_WGSL } from './shaders.js';

export async function createLoader(gpu, canvas) {
  const { device } = gpu;
  const module = await makeModule(device, LOADER_WGSL, 'loader');

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: {
      module,
      entryPoint: 'fs',
      targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const uniform = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniform } }],
  });

  const data = new Float32Array([0, 0, 1, 1]);
  let progress = 0;
  let running = false;
  let raf = 0;
  let t0 = performance.now();
  let dpr = 1;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function frame(now) {
    if (!running) return;
    resize();
    data[0] = progress;
    data[1] = (now - t0) / 1000;
    data[2] = canvas.width;
    data[3] = canvas.height;
    device.queue.writeBuffer(uniform, 0, data);

    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: ctx.getCurrentTexture().createView(),
          loadOp: 'clear',
          clearValue: { r: 0.05, g: 0.049, b: 0.041, a: 1 },
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
    raf = requestAnimationFrame(frame);
  }

  return {
    setProgress(p) {
      progress = Math.max(0, Math.min(1, p));
    },
    start() {
      if (running) return;
      running = true;
      t0 = performance.now();
      raf = requestAnimationFrame(frame);
    },
    stop() {
      running = false;
      cancelAnimationFrame(raf);
    },
    destroy() {
      this.stop();
    },
  };
}
