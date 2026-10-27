import { acquireGPU } from './gpu.js';
import { createLoader } from './loader.js';
import { createThink } from './think.js';
import { createViewer } from './viewer.js';
import { loadPLY } from './ply.js';
import {
  MODE, TOTAL_MB, SAMPLES, FALLBACK_PLY, runInference, tierAt,
} from './pipeline.js';

const $ = (id) => document.getElementById(id);

const el = {
  boot: $('stage-boot'),
  drop: $('stage-drop'),
  think: $('stage-think'),
  view: $('stage-view'),
  bootCanvas: $('boot-canvas'),
  bootFill: $('boot-fill'),
  bootBytes: $('boot-bytes'),
  bootPhase: $('boot-phase'),
  dropZone: $('drop'),
  samples: $('samples'),
  file: $('file'),
  thinkCanvas: $('think-canvas'),
  thinkFill: $('think-fill'),
  thinkPct: $('think-pct'),
  thinkPhase: $('think-phase'),
  thinkRes: $('think-res'),
  azFill: $('az-fill'),
  azVal: $('az-val'),
  viewCanvas: $('view-canvas'),
  viewHint: $('view-hint'),
  replay: $('btn-replay'),
  metaText: $('meta-text'),
  footGpu: $('foot-gpu'),
  footMode: $('foot-mode'),
};

const STAGE_ELS = { boot: el.boot, drop: el.drop, think: el.think, view: el.view };

let gpu = null;
let loader = null;
let viewer = null;
let think = null;
let runToken = 0;

function setStage(name) {
  document.body.dataset.stage = name;
  for (const [key, node] of Object.entries(STAGE_ELS)) {
    node.classList.toggle('is-active', key === name);
  }
}

/* ------------------------------------------------------------------ 启动 */

async function boot() {
  document.body.dataset.booted = '1';
  el.footMode.textContent = MODE === 'demo'
    ? '演示管线：推理尚未接线，结果来自预烘焙 PLY'
    : '本地推理 · 全部在浏览器内';

  gpu = await acquireGPU();
  document.body.dataset.gpu = gpu ? 'ready' : 'none';

  if (gpu) {
    el.metaText.textContent = gpu.label;
    el.footGpu.textContent = `${gpu.label} · ${gpu.features.includes('shader-f16') ? 'f16' : 'no-f16'}`;
    try {
      loader = await createLoader(gpu, el.bootCanvas);
      loader.start();
    } catch (err) {
      console.error(err);
      document.body.dataset.gpu = 'none';
      el.metaText.textContent = 'WebGPU 初始化失败';
    }
  } else {
    el.metaText.textContent = '这个浏览器没有 WebGPU';
    el.footGpu.textContent = 'WebGPU 不可用';
  }

  await loadWeights();
  loader && loader.stop();
  setStage('drop');
}

/** 分档加载。字节是真的（来自 manifest 与参数量表），耗时是演示的。 */
function loadWeights() {
  const DUR = 4200;
  const t0 = performance.now();
  return new Promise((resolve) => {
    function tick() {
      const raw = Math.min(1, (performance.now() - t0) / DUR);
      const p = 1 - Math.pow(1 - raw, 3);
      if (loader) loader.setProgress(p);
      el.bootFill.style.width = `${(p * 100).toFixed(1)}%`;
      el.bootBytes.textContent = `${(p * TOTAL_MB).toFixed(1)} MB / ${TOTAL_MB} MB`;
      el.bootPhase.textContent = tierAt(p).name;
      if (raw < 1) requestAnimationFrame(tick);
      else setTimeout(resolve, 260);
    }
    requestAnimationFrame(tick);
  });
}

/* ------------------------------------------------------------------ 投图 */

function buildSamples() {
  for (const s of SAMPLES) {
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = s.label;
    b.addEventListener('click', () => {
      fetch(s.image)
        .then((r) => r.blob())
        .then((blob) => createImageBitmap(blob))
        .then((bmp) => start(bmp, s.ply))
        .catch((e) => fail(e));
    });
    li.appendChild(b);
    el.samples.appendChild(li);
  }
}

el.dropZone.addEventListener('click', () => el.file.click());
el.dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.file.click(); }
});

el.file.addEventListener('change', () => {
  const f = el.file.files && el.file.files[0];
  if (!f) return;
  acceptBlob(f);
  el.file.value = '';
});

el.dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  el.dropZone.classList.add('is-over');
});
el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('is-over'));
el.dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  el.dropZone.classList.remove('is-over');
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) acceptBlob(f);
});

window.addEventListener('paste', (e) => {
  if (document.body.dataset.stage !== 'drop') return;
  for (const item of e.clipboardData?.items || []) {
    if (item.type.startsWith('image/')) {
      const f = item.getAsFile();
      if (f) { acceptBlob(f); break; }
    }
  }
});

async function acceptBlob(blob) {
  try {
    const bmp = await createImageBitmap(blob);
    // 任何输入都落到同一份预烘焙结果上——页脚已经写明。
    start(bmp, FALLBACK_PLY);
  } catch (err) {
    fail(err);
  }
}

/* ------------------------------------------------------------------ 推理 */

function start(bitmap, plyUrl) {
  const token = ++runToken;

  if (gpu) el.metaText.textContent = gpu.label;
  setStage('think');
  if (!think) think = createThink(el.thinkCanvas);
  think.setImage(bitmap);
  think.draw(0);
  el.thinkFill.style.width = '0%';
  el.thinkPct.textContent = '0%';
  el.thinkRes.textContent = '…';

  // PLY 在"解码"之前就该到，所以提前拉。它占的是网络，不是 GPU。
  const modelPromise = loadPLY(plyUrl);
  modelPromise.catch(() => {});

  runInference({
    onProgress(p, stage) {
      if (token !== runToken) return;
      think.draw(p);
      el.thinkFill.style.width = `${(p * 100).toFixed(1)}%`;
      el.thinkPct.textContent = `${Math.round(p * 100)}%`;
      el.thinkPhase.textContent = stage.label;
      const { w, h } = think.size;
      el.thinkRes.textContent = `${w} × ${h}`;
    },
    signal: { get aborted() { return token !== runToken; } },
  })
    .then(() => modelPromise)
    .then(async (model) => {
      if (token !== runToken) return;
      await enterView(model);
    })
    .catch((err) => {
      if (token !== runToken) return;
      fail(err);
    });
}

async function enterView(model) {
  setStage('view');
  el.viewHint.textContent = MODE === 'demo'
    ? `${model.count.toLocaleString()} 高斯 · 演示结果`
    : `${model.count.toLocaleString()} 高斯`;

  if (!gpu) {
    const note = document.createElement('p');
    note.className = 'view-fallback';
    note.textContent = '这个浏览器没有 WebGPU，画不出高斯。结果文件已经生成。';
    el.view.appendChild(note);
    return;
  }

  try {
    if (!viewer) {
      viewer = await createViewer(gpu, el.viewCanvas, {
        onAzimuth(deg) {
          el.azFill.style.width = `${Math.min(100, (deg / 360) * 100)}%`;
          el.azVal.textContent = `${Math.round(deg)}°`;
        },
        onStats({ fps, count }) {
          el.viewHint.textContent = `${count.toLocaleString()} 高斯 · ${fps.toFixed(0)} fps`;
        },
      });
    }
    viewer.setSplat(model);
    viewer.start();
    window.__flipView = viewer;   // 调试入口：__.cam.az = 1.57; __.redraw()
  } catch (err) {
    fail(err);
  }
}

/* ------------------------------------------------------------------ 重来 */

el.replay.addEventListener('click', () => {
  runToken++;
  viewer && viewer.stop();
  setStage('drop');
  const note = el.view.querySelector('.view-fallback');
  if (note) note.remove();
});

function fail(err) {
  console.error(err);
  el.metaText.textContent = `出错了：${err.message || err}`;
  runToken++;
  loader && loader.stop();
  setStage('drop');
}

/* ------------------------------------------------------------------ go */

buildSamples();
setStage('boot');
boot().catch(fail);
