// 权重入口。真的把 assets/k2n/int4.bin 读进来 —— 209,276,416 字节，
// 和 manifest 的账逐字节对得上，所以这段进度条报的是字节，不是时间。
//
// 这一层只做三件事：
//   1. 流式读字节，按 manifest 的偏移说出"现在到哪个张量了"；
//   2. 每个张量读完之后抽查它的 scale（有限、为正）和码值块（不是一整片空）；
//   3. 把账目交出去：多少张量、多少参数、多少 MB 码值 / 多少 MB 尺度。
//
// 不反量化，也不碰 GPU。反量化在内核里，GPU 留给推理。

const LOCAL = {
  manifest: '../assets/k2n/manifest.json',
  bin: '../assets/k2n/int4.bin',
};

/** ?weights=<url> / ?manifest=<url> 可以换源，本地默认读 assets/k2n。 */
export function weightUrls() {
  const q = new URLSearchParams(location.search);
  return {
    manifest: q.get('manifest') || LOCAL.manifest,
    bin: q.get('weights') || LOCAL.bin,
  };
}

// half → float 查表。DataView.getFloat16 太新，查表既快又不挑浏览器。
const HALF = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 5.960464477539063e-8
      : e === 31 ? (m ? NaN : s * Infinity)
        : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

const MB = 1e6;

/** 把 manifest 折成可以显示、可以查偏移的账。 */
export function summarize(man) {
  const segs = [];
  const checks = [];
  const int4 = { n: 0, params: 0, qBytes: 0, sBytes: 0, scales: 0 };
  const fp16 = { n: 0, params: 0, bytes: 0 };

  for (const t of man.tensors) {
    if (t.kind === 'int4') {
      int4.n++;
      int4.params += t.n * t.k;
      int4.qBytes += t.bytes_q;
      int4.sBytes += t.bytes_s;
      int4.scales += t.bytes_s >> 1;
      segs.push({ start: t.offset_q, end: t.offset_q + t.bytes_q, name: t.name, kind: 'code' });
      segs.push({ start: t.offset_s, end: t.offset_s + t.bytes_s, name: t.name, kind: 'scale' });
      checks.push({
        name: t.name, kind: 'int4',
        end: Math.max(t.offset_q + t.bytes_q, t.offset_s + t.bytes_s),
        scale: [t.offset_s, t.bytes_s],
        code: [t.offset_q, t.bytes_q],
      });
    } else {
      fp16.n++;
      fp16.params += t.bytes >> 1;
      fp16.bytes += t.bytes;
      segs.push({ start: t.offset, end: t.offset + t.bytes, name: t.name, kind: 'fp16' });
      checks.push({
        name: t.name, kind: 'fp16',
        end: t.offset + t.bytes,
        scale: [t.offset, t.bytes],
      });
    }
  }

  segs.sort((a, b) => a.start - b.start);
  checks.sort((a, b) => a.end - b.end);

  return {
    bytes: man.bytes,
    bits: man.bits,
    block: man.block_size,
    tensors: man.tensors.length,
    int4,
    fp16,
    segs,
    checks,
  };
}

// 抽样点必须按元素宽度对齐：fp16 落在偶数字节，u32 word 落在 4 的倍数上。
// 差一个字节就会把 0.0123 读成 -14.24 —— 而且只有一半的点会出错，看起来像"数据坏了"。
function pickOffsets(off, bytes, count, align) {
  const span = Math.floor((bytes - align) / align);
  if (span < 0) return new Int32Array(0);
  const n = Math.min(count, span + 1);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const k = n === 1 ? 0 : Math.round((i * span) / (n - 1));
    out[i] = off + k * align;
  }
  return out;
}

/**
 * 流式读权重，边读边验。
 * onProgress({ read, total, p, tensor, rate, stats }) —— p 是真字节比例。
 */
export async function streamWeights({ manifestUrl, binUrl, onProgress, signal, retain = true } = {}) {
  const t0 = performance.now();

  const mres = await fetch(manifestUrl, { signal });
  if (!mres.ok) throw new Error(`manifest ${mres.status}`);
  const man = await mres.json();
  const sum = summarize(man);

  const res = await fetch(binUrl, { signal });
  if (!res.ok) throw new Error(`权重 ${res.status}`);
  const total = Number(res.headers.get('content-length')) || sum.bytes;
  if (!res.body) throw new Error('这个浏览器不支持流式响应');

  // 209 MB 一次分配。分配不到就退化成"只数字节"——进度照样是真的。
  let buf = null;
  if (retain) {
    try { buf = new Uint8Array(Math.max(total, sum.bytes)); } catch { buf = null; }
  }
  const view = buf ? new DataView(buf.buffer) : null;

  const stats = {
    tensors: 0,
    scales: 0, scalesBad: 0,
    words: 0, wordsBlank: 0,
    fp16: 0, fp16Bad: 0,
    minScale: Infinity, maxScale: 0,
  };

  function verify(c) {
    if (!view) return;
    stats.tensors++;
    if (c.kind === 'int4') {
      for (const off of pickOffsets(c.scale[0], c.scale[1], 12, 2)) {
        if (off + 2 > view.byteLength) continue;
        const v = HALF[view.getUint16(off, true)];
        stats.scales++;
        if (!(v > 0) || !Number.isFinite(v)) stats.scalesBad++;
        else {
          if (v < stats.minScale) stats.minScale = v;
          if (v > stats.maxScale) stats.maxScale = v;
        }
      }
      for (const off of pickOffsets(c.code[0], c.code[1], 8, 4)) {
        if (off + 4 > view.byteLength) continue;
        const w = view.getUint32(off, true);
        stats.words++;
        if (w === 0 || w === 0xffffffff) stats.wordsBlank++;
      }
    } else {
      let any = false;
      for (const off of pickOffsets(c.scale[0], c.scale[1], 6, 2)) {
        if (off + 2 > view.byteLength) continue;
        const v = HALF[view.getUint16(off, true)];
        stats.fp16++;
        if (!Number.isFinite(v)) stats.fp16Bad++;
        if (v !== 0) any = true;
      }
      if (!any) stats.fp16Bad++;
    }
  }

  const reader = res.body.getReader();
  let read = 0;
  let cp = 0;
  let last = 0;
  let si = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && read + value.length <= buf.length) buf.set(value, read);
    read += value.length;

    while (si + 1 < sum.segs.length && sum.segs[si + 1].start <= read) si++;
    const seg = sum.segs[si];

    while (cp < sum.checks.length && sum.checks[cp].end <= read) verify(sum.checks[cp++]);

    const now = performance.now();
    if (now - last > 60) {
      last = now;
      const secs = (now - t0) / 1000;
      onProgress && onProgress({
        read,
        total,
        p: total ? read / total : 0,
        tensor: seg ? seg.name : '',
        part: seg ? seg.kind : '',
        rate: secs > 0 ? read / MB / secs : 0,
        stats,
      });
    }
  }

  while (cp < sum.checks.length) verify(sum.checks[cp++]);

  const secs = (performance.now() - t0) / 1000;
  const clean = stats.scalesBad === 0 && stats.wordsBlank === 0 && stats.fp16Bad === 0;

  const result = {
    manifest: man,
    ledger: sum,
    read,
    total,
    short: read !== total,
    verified: !!view,
    stats,
    clean,
    secs,
    rate: secs > 0 ? read / MB / secs : 0,
    bytes: buf,
  };
  onProgress && onProgress({
    read: total, total, p: 1, tensor: '', part: 'done',
    rate: result.rate, stats,
  });
  return result;
}

/** 某个字节偏移落在哪个张量上。加载页用它在扫过时点亮张量名。 */
export function tensorAt(ledger, byte) {
  const segs = ledger.segs;
  let lo = 0;
  let hi = segs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const s = segs[mid];
    if (byte < s.start) hi = mid - 1;
    else if (byte >= s.end) lo = mid + 1;
    else return s;
  }
  return null;
}

/** 一句话的验货结论，加载页最后一行显示它。 */
export function verifyLine(r) {
  if (!r.verified) return '只数了字节（内存不够留存）';
  const s = r.stats;
  const parts = [
    `尺度 ${s.scales + s.fp16} 抽样`,
    `${s.scalesBad + s.fp16Bad} 异常`,
    `${s.wordsBlank} 空块`,
  ];
  return parts.join(' · ');
}
