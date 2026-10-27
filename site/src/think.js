// 阶段 2：思考。
//
// 这个阶段的全部画面都是 Canvas2D，一帧都不提交给 WebGPU——推理正在用整个队列。
// "降级"不是装饰：分辨率阶梯和色阶都在往上爬，观众看到的是模型在一点点补齐细节，
// 而不是一个和进度无关的菊花。像素本身就是进度条。

const LADDER = [20, 28, 38, 52, 70, 94, 126];

const BAYER4 = [
  0, 8, 2, 10,
  12, 4, 14, 6,
  3, 11, 1, 9,
  15, 7, 13, 5,
];

export function createThink(canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  let bitmap = null;
  let aspect = 1;
  let lastKey = '';
  let lut = null;
  let lutKey = '';

  // 调色板 LUT：每个灰阶量化到 L 档，加一点有序抖动。
  function buildLut(levels) {
    const key = String(levels);
    if (lutKey === key) return lut;
    lut = new Uint8ClampedArray(256 * 16);
    for (let b = 0; b < 16; b++) {
      for (let v = 0; v < 256; v++) {
        const n = v / 255;
        const d = BAYER4[b] / 16 - 0.5;
        const q = Math.round(n * (levels - 1) + d * 0.55);
        lut[b * 256 + v] = Math.round((Math.max(0, Math.min(levels - 1, q)) / (levels - 1)) * 255);
      }
    }
    lutKey = key;
    return lut;
  }

  function setImage(bmp) {
    bitmap = bmp;
    aspect = bmp.width / bmp.height;
    // 竖图不让它撑破视口
    const clampA = Math.max(0.62, Math.min(1.7, aspect));
    canvas.parentElement.style.aspectRatio = String(clampA);
    lastKey = '';
  }

  function draw(progress) {
    if (!bitmap) return;

    const p = Math.max(0, Math.min(1, progress));
    const step = Math.min(LADDER.length - 1, Math.floor(p * LADDER.length));
    const w = LADDER[step];
    const h = Math.max(8, Math.round(w / aspect));
    const levels = 5 + Math.round(p * 3);

    const key = `${w}x${h}:${levels}`;
    if (key === lastKey) return;
    lastKey = key;

    canvas.width = w;
    canvas.height = h;

    // 先按目标尺寸重采样（这里开平滑，缩小时才不会丢细节），
    // 再用 LUT 量化成有限的几档颜色。
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(bitmap, 0, 0, w, h);

    const img = ctx.getImageData(0, 0, w, h);
    const px = img.data;
    const table = buildLut(levels);
    for (let y = 0; y < h; y++) {
      const row = (y & 3) << 2;
      for (let x = 0; x < w; x++) {
        const b = row + (x & 3);
        const o = (y * w + x) * 4;
        px[o] = table[b * 256 + px[o]];
        px[o + 1] = table[b * 256 + px[o + 1]];
        px[o + 2] = table[b * 256 + px[o + 2]];
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  return { setImage, draw, get size() { return { w: canvas.width, h: canvas.height }; } };
}
