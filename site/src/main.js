import { acquireGPU } from './gpu.js';
import { createLoader } from './loader.js';
import { createThink } from './think.js';
import { createViewer } from './viewer.js';
import { loadPLY } from './ply.js';
import { streamWeights, weightUrls, verifyLine, tensorAt } from './weights.js';
import {
  MODE, SAMPLES, FALLBACK_PLY, runInference,
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
  bootTiers: $('boot-tiers'),
  bootChecks: $('boot-checks'),
  bootRate: $('boot-rate'),
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

/* --------------------------------------------------------------- 权重加载 */

// 卡片扫完至少留这么久。本机读 200 MB 只要两三百毫秒，不设下限就只剩一眨。
const BOOT_MIN_SWEEP = 1500;

/** 桌面留下字节（下一步要喂给 GPU），手机只数不存。 */
function canRetain() {
  const mem = navigator.deviceMemory;
  if (mem && mem < 8) return false;
  return !matchMedia('(pointer: coarse)').matches;
}

/**
 * 真的读 assets/k2n/int4.bin。显示的是字节，不是时间：
 * 进度条走多快取决于盘和网络，走完了就是读完了。
 */
function loadWeights() {
  const urls = weightUrls();
  const t0 = performance.now();
  const st = { p: 0, read: 0, total: 0, tensor: '', rate: 0, stats: null };
  let shown = 0;
  let done = false;
  let failed = false;
  let ledger = null;
  let failure = null;

  const stream = streamWeights({
    manifestUrl: urls.manifest,
    binUrl: urls.bin,
    retain: canRetain(),
    onProgress(e) { Object.assign(st, e); },
  }).then((r) => r, (e) => { failure = e; return null; });

  return new Promise((resolve) => {
    function paint() {
      // 真实进度是上限：显示的字节数永远不会超过已经读到的字节数。
      // 本机读 200 MB 只要两百毫秒，所以再给一个最慢的扫过速度，否则动画一闪就没了。
      const floorP = Math.min(1, (performance.now() - t0) / BOOT_MIN_SWEEP);
      const real = failure ? 1 : st.p;
      const target = Math.min(real, Math.max(shown, floorP));
      shown += (target - shown) * 0.14;

      if (loader) loader.setProgress(shown);
      el.bootFill.style.width = `${(shown * 100).toFixed(1)}%`;

      if (st.total) {
        const mb = (shown >= 0.999 ? st.total : shown * st.total) / 1e6;
        el.bootBytes.textContent = `${mb.toFixed(1)} MB / ${(st.total / 1e6).toFixed(1)} MB`;
      } else {
        el.bootBytes.textContent = '—';
      }

      // 扫到哪个张量就报哪个。读完了还要扫一小会儿，这时候按显示位置回报，
      // 免得条子才过半就写"校验完成"。
      let phase = st.tensor;
      if (done && ledger && st.total) {
        const seg = shown >= 0.999 ? null : tensorAt(ledger, shown * st.total);
        phase = seg ? seg.name : (failed ? '权重不在本地' : '校验完成');
      }
      el.bootPhase.textContent = phase || '连接';
      el.bootRate.textContent = st.rate > 0 ? `${st.rate.toFixed(0)} MB/s` : '—';
    }

    function tick() {
      paint();
      if (done && shown > 0.999) {
        paint();
        setTimeout(resolve, failure ? 900 : 620);
        return;
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    stream.then((r) => {
      if (!r) {
        // 权重不在本地（发布时 200 MB 不进 git）—— 进度退化成占位，页面照样能走。
        document.body.dataset.weights = 'missing';
        el.bootTiers.textContent = '权重不在本地 · 占位进度';
        el.bootChecks.textContent = '跳过校验';
        console.warn('weight stream failed:', failure && failure.message);
        failed = true;
        done = true;
        return;
      }

      const L = r.ledger;
      ledger = L;
      document.body.dataset.weights = r.clean && !r.short ? 'ok' : 'check';
      el.bootTiers.textContent =
        `DiT 学生 · ${L.int4.n} 张量化 + ${L.fp16.n} 张 fp16 · ${(L.int4.params / 1e6).toFixed(0)} M 参数`;
      el.bootChecks.textContent = verifyLine(r);
      el.bootPhase.textContent = r.short ? '读短了' : '校验完成';
      window.__flipWeights = r;   // 调试：__.bytes 就是那 209 MB
      done = true;
    });
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
