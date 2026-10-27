// 管线。现在只有 demo 一条，但接口是按真管线切的：
//   loadWeights({ onProgress })        → 分档下载，字节数是真的
//   runInference({ bitmap, onProgress }) → 预处理 / 编码 / 采样 / 解码
//
// 接真推理时只改这个文件：onProgress 的语义不变（0..1 的全局进度 + 当前阶段），
// UI 一行都不用动。MODE 会写进页脚，别让观众以为结果真的由这张图生成。

export const MODE = 'demo';

// 分档字节数取自 k2m 打出来的 manifest 与 PLAN 的参数量表（DiT 按 g32 的 209 MB）。
export const TIERS = [
  { name: 'DINOv3 ViT-H/16+', mb: 433, note: '编码器，体积最大的一块' },
  { name: 'Flux2 VAE 编码器', mb: 18, note: '进 latent' },
  { name: 'DiT（学生，int4 g32）', mb: 209, note: '采样主干' },
  { name: '八叉树解码器', mb: 26, note: '8 层采样' },
  { name: '高斯解码器', mb: 122, note: '65K 高斯属性' },
  { name: '抠图（浏览器侧替换）', mb: 114, note: '前景分离' },
];

export const TOTAL_MB = TIERS.reduce((a, t) => a + t.mb, 0);

export const STAGES = [
  { key: 'prep', label: '预处理 · 抠图', from: 0.0, to: 0.1, ms: 700 },
  { key: 'encode', label: '编码 · DINOv3 + VAE', from: 0.1, to: 0.38, ms: 2100 },
  { key: 'sample', label: '采样 · 4 步无 CFG', from: 0.38, to: 0.84, ms: 4200 },
  { key: 'decode', label: '解码 · 八叉树 + 高斯', from: 0.84, to: 1.0, ms: 1500 },
];

// 演示阶段可以按倍数快进（?speed=4），方便截图和自动化检查。
const SPEED = (() => {
  const v = Number(new URLSearchParams(location.search).get('speed') || '1');
  return Number.isFinite(v) && v > 0 ? v : 1;
})();

// 示例：图和 PLY 是同一张输入跑出来的（仓库里唯一一对）。
export const SAMPLES = [
  {
    id: 'house',
    label: '石屋',
    image: '../assets/k2b/input.png',
    ply: '../assets/k2g/building_stone_house_t20.ply',
    prebaked: true,
  },
];

/** 预烘焙结果：任何输入都会落到它上面。 */
export const FALLBACK_PLY = SAMPLES[0].ply;

export function stageAt(p) {
  for (let i = STAGES.length - 1; i >= 0; i--) {
    if (p >= STAGES[i].from) return STAGES[i];
  }
  return STAGES[0];
}

export function tierAt(p) {
  let acc = 0;
  const target = p * TOTAL_MB;
  for (const t of TIERS) {
    acc += t.mb;
    if (target <= acc) return t;
  }
  return TIERS[TIERS.length - 1];
}

/** 演示用的推理：只推进时间，不动数据。 */
export async function runInference({ onProgress, signal } = {}) {
  const total = STAGES.reduce((a, s) => a + s.ms, 0) / SPEED;
  const t0 = performance.now();
  return new Promise((resolve, reject) => {
    function tick() {
      if (signal && signal.aborted) return reject(new Error('aborted'));
      const elapsed = performance.now() - t0;
      const p = Math.min(1, elapsed / total);
      onProgress && onProgress(p, stageAt(p));
      if (p < 1) requestAnimationFrame(tick);
      else resolve({ mode: MODE });
    }
    requestAnimationFrame(tick);
  });
}
