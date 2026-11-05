import { acquireGPU } from './gpu.js';
import { createLoader } from './loader.js';
import { createThink } from './think.js';
import { createViewer } from './viewer.js';
import {
  MODE, STAGES, SAMPLES, runInference, initModels, modelsReady,
  ARTIFACT_BYTES,
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
  viewBanner: $('view-banner'),
  dropError: $('drop-error'),
  replay: $('btn-replay'),
  metaText: $('meta-text'),
  footGpu: $('foot-gpu'),
  footMode: $('foot-mode'),
};

const STAGE_ELS = { boot: el.boot, drop: el.drop, think: el.think, view: el.view };

// ?fix=1 跳过编码器，直接喂夹具条件：把「照片 → 条件」和「DiT + 解码器」分开看。
const FIXTURE = new URLSearchParams(location.search).get('fix') === '1';

let gpu = null;
let loader = null;
let viewer = null;
let think = null;
let runToken = 0;
let modelError = null;
// 结果标签的尾巴（本机推理 / 夹具条件 / 演示结果）。viewer 每秒报一次 fps，
// 那一次重写标签，所以真话得存在这里，不能现场再猜。
let lastLabel = '';

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
    : '正在准备本机推理';

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

  await preload();
  loader && loader.stop();

  el.footMode.textContent = modelsReady()
    ? '本机推理 · 编码器按需加载 · 4 步 int4 DiT + 解码器在浏览器里'
    : '模型没加载成功 · 不会生成结果';
  if (!modelsReady()) {
    showDropError('模型没加载成功', modelError || new Error('未知原因'));
  }
  setStage('drop');
}

/* --------------------------------------------------------------- 权重加载 */

// 卡片扫完至少留这么久。本机读 200 MB 只要两三百毫秒，不设下限就只剩一眨。
const BOOT_MIN_SWEEP = 1500;

/**
 * 真的把 4 个图 + 4 个夹具读进来（默认 HF，?local=../w/ 读本地），
 * 进度条走的是字节数。失败不挡路：后面退回预烘焙 PLY。
 */
function preload() {
  const t0 = performance.now();
  const st = { file: '', loaded: 0, total: ARTIFACT_BYTES, rate: 0, phase: '' };
  let shown = 0;
  let done = false;
  let failure = null;
  let host = '';

  const job = initModels({
    onProgress(e) { Object.assign(st, e); if (e.host) host = e.host; },
  }).then(() => {}, (e) => { failure = e; });

  return new Promise((resolve) => {
    function paint() {
      // 真实进度是上限：显示的字节数永远不会超过已经读到的字节数。
      const floorP = Math.min(1, (performance.now() - t0) / BOOT_MIN_SWEEP);
      const real = failure ? 1 : (st.total ? st.loaded / st.total : 0);
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

      el.bootPhase.textContent = failure ? '模型不可用'
        : st.phase === 'session' ? `创建会话 · ${st.file}`
          : st.phase === 'done' && st.p >= 1 ? '就绪'
            : st.file || '连接';
      el.bootRate.textContent = st.rate > 0 ? `${st.rate.toFixed(0)} MB/s` : '—';
      el.bootTiers.textContent = 'DiT 245 MB · 解码器 208 MB · 条件夹具 24 MB';
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

    job.then(() => {
      done = true;
      if (failure) {
        modelError = failure;
        document.body.dataset.weights = 'missing';
        el.bootChecks.textContent = '模型取不到 · 不会生成结果';
        console.warn('model preload failed:', failure.message);
      } else {
        document.body.dataset.weights = 'ok';
        el.bootChecks.textContent = `${host === 'local' ? '本地文件' : host} · 3 个图 + 4 个夹具 · 已进 Cache`;
      }
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
        .then((bmp) => start(bmp))
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
    start(bmp);
  } catch (err) {
    fail(err);
  }
}

/* ------------------------------------------------------------------ 推理 */

function start(bitmap) {
  const token = ++runToken;
  el.dropError.hidden = true;

  if (gpu) el.metaText.textContent = gpu.label;
  setStage('think');
  if (!think) think = createThink(el.thinkCanvas);
  think.setImage(bitmap);
  think.draw(0);
  el.thinkFill.style.width = '0%';
  el.thinkPct.textContent = '0%';
  el.thinkRes.textContent = '…';

  const signal = { get aborted() { return token !== runToken; } };
  const onProgress = (p, stage) => {
    if (token !== runToken) return;
    think.draw(p);
    el.thinkFill.style.width = `${(p * 100).toFixed(1)}%`;
    el.thinkPct.textContent = `${Math.round(p * 100)}%`;
    el.thinkPhase.textContent = stage.label;
    const { w, h } = think.size;
    el.thinkRes.textContent = `${w} × ${h}`;
  };

  // 模型没就绪就直接失败：不拿预烘焙结果冒充这一趟的输出。
  const job = modelsReady()
    ? runInference({ bitmap: FIXTURE ? null : bitmap, onProgress, signal })
    : Promise.reject(modelError || new Error('模型没有加载成功'));

  job
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
  window.__flipSplat = model;   // 调试入口：__.count / __.data / __.encoded
  // encoded 是这一趟到底有没有把照片变成条件。真实输入必须为 true；
  // 只有显式的 fix=1 诊断模式才会是 false（bitmap 被主流程置空）。
  const usedPhoto = model.encoded === true;
  const label = usedPhoto ? '本机推理' : '夹具条件';
  lastLabel = label;
  el.viewHint.textContent = `${model.count.toLocaleString()} 高斯 · ${label}`;
  if (!usedPhoto) {
    el.viewBanner.hidden = false;
    el.viewBanner.textContent = FIXTURE
      ? '诊断模式 fix=1 · 故意忽略照片，条件来自夹具'
      : '这一趟没有编码照片 · 条件来自夹具';
  } else {
    el.viewBanner.hidden = true;
  }
  window.__flipRun = {
    usedPhoto, fix: FIXTURE, count: model.count,
    dit: new URLSearchParams(location.search).get('dit') || 'default',
  };
  el.footMode.textContent = usedPhoto
    ? '本机推理 · 编码器已接入 · 4 步 int4 DiT + 解码器在浏览器里'
    : '本机推理 · 条件来自夹具（fix=1）';

  if (!gpu) {
    const note = document.createElement('p');
    note.className = 'view-fallback';
    note.textContent = '这个浏览器没有 WebGPU，无法显示这批高斯。';
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
          el.viewHint.textContent = `${count.toLocaleString()} 高斯 · ${fps.toFixed(0)} fps`
            + (lastLabel ? ` · ${lastLabel}` : '');
        },
      });
    }
    // decoder 已经产出 viewer 坐标系（Y 向上），不再翻 y；
    // 正面视角由离线 EWA 转台实测（az=2.0 是脸部，3.35 是侧后方）
    viewer.setSplat(model, { raw: true, az: 2.0, el: 0.18 });
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

function showDropError(title, err) {
  const msg = err && (err.message || err);
  el.dropError.hidden = false;
  el.dropError.textContent = `${title}：${msg || '未知原因'}`;
}

function fail(err, title = '生成失败') {
  console.error(err);
  el.metaText.textContent = `出错了：${err.message || err}`;
  showDropError(title, err);
  runToken++;
  loader && loader.stop();
  setStage('drop');
}

/* ------------------------------------------------------------------ go */

buildSamples();
setStage('boot');
boot().catch((e) => fail(e, '启动失败'));
