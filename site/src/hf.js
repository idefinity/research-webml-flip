// Hugging Face 取文件。huggingface.co 先试，不行就换 hf-mirror.com。
//
// 为什么要有 fallback：这台机器到 huggingface.co 是 502（系统代理直接拒掉 CONNECT），
// 而 hf-mirror.com 是通的。浏览器里没有代理可绕，所以两个域名都试一遍，
// 谁先给出字节谁就是今天的源。页面右上角会写出用的是哪一个，
// 免得"加载慢"被当成"模型慢"。
//
// 另外两件事是顺手做掉的：
//   - 大文件走并发 Range。镜像站 accept-ranges: bytes，294 MB 单流要十几分钟，
//     8 路并发能压到一两分钟；
//   - 下载完进 Cache API。第二次打开是本地读，不重复掏流量。
//     Cache 放不下就静默跳过 —— 缓存失败不该让推理跑不起来。

export const HF_HOSTS = ['https://huggingface.co', 'https://hf-mirror.com'];

const CACHE_NAME = 'flip-weights-v1';
const PROBE_TIMEOUT_MS = 6000;

export function resolveUrl(host, repo, file, revision = 'main') {
  return `${host}/${repo}/resolve/${revision}/${file}`;
}

function supportsCache() {
  return typeof caches !== 'undefined' && !!caches.open;
}

async function cacheGet(key) {
  if (!supportsCache()) return null;
  try {
    const c = await caches.open(CACHE_NAME);
    const hit = await c.match(key);
    if (!hit) return null;
    return new Uint8Array(await hit.arrayBuffer());
  } catch {
    return null;
  }
}

async function cachePut(key, bytes) {
  if (!supportsCache()) return false;
  try {
    const c = await caches.open(CACHE_NAME);
    await c.put(key, new Response(bytes.slice().buffer, {
      headers: { 'content-type': 'application/octet-stream' },
    }));
    return true;
  } catch {
    return false;
  }
}

function combine(parts, total) {
  const out = new Uint8Array(total);
  for (const { start, data } of parts) out.set(data, start);
  return out;
}

/**
 * 问一个源：文件多大、支不支持 Range。答不上来的源直接换下一个。
 * 用 Range: bytes=0-0 而不是 HEAD —— resolve 端点会 302，HEAD 跟随之后
 * 有些 CDN 不给 Content-Length，一个字节的 GET 不会。
 */
async function probe(url, timeout = PROBE_TIMEOUT_MS) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout);
  try {
    const r = await fetch(url, {
      headers: { Range: 'bytes=0-0' },
      signal: ctl.signal,
      redirect: 'follow',
      mode: 'cors',
      cache: 'no-store',
    });
    // 200 而不是 206：这个源不支持 Range，但仍然能用（单流）。
    if (!r.ok && r.status !== 206) return { ok: false, status: r.status };
    const cr = r.headers.get('content-range') || '';
    const total = cr && cr.indexOf('/') > 0
      ? Number(cr.split('/').pop())
      : Number(r.headers.get('content-length') || 0);
    return {
      ok: true,
      total,
      ranged: r.status === 206 && total > 0,
      finalUrl: r.url,
    };
  } catch (e) {
    return { ok: false, status: 0, error: String(e && e.message || e) };
  } finally {
    clearTimeout(timer);
  }
}

/** 一路 Range 请求，失败从断点续，不从零开始。 */
async function fetchRange(url, start, end, tries = 4) {
  let got = null;
  for (let i = 0; i < tries; i++) {
    const off = start + (got ? got.length : 0);
    if (off >= end) break;
    try {
      const r = await fetch(url, {
        headers: { Range: `bytes=${off}-${end - 1}` },
        mode: 'cors',
        cache: 'no-store',
      });
      if (!r.ok && r.status !== 206) throw new Error('HTTP ' + r.status);
      const chunk = new Uint8Array(await r.arrayBuffer());
      got = got ? concat(got, chunk) : chunk;
    } catch (e) {
      if (i === tries - 1) throw e;
      await new Promise((s) => setTimeout(s, 400 * (i + 1)));
    }
  }
  return { start, data: got };
}

function concat(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

/**
 * 取一个文件。
 * onProgress({ read, total, p, host, phase, rate }) —— read 是真字节数。
 * 返回 { bytes, host, total, fromCache, secs, rate, ranged }。
 */
export async function hfFetch({
  repo,
  file,
  revision = 'main',
  hosts = HF_HOSTS,
  localBase = '',
  parts = 8,
  minRangedMB = 12,
  useCache = true,
  onProgress = null,
  signal = null,
} = {}) {
  // localBase 存在的场合只问本地：这一台机器的浏览器两个 HF 域名都到不了
  // （代理白名单里没有），要在本机验证管线就得先把文件拉到本地目录，
  // 再让同一段代码从那里读。走这条路时不进 Cache，免得把本地文件当成缓存。
  if (localBase) {
    const url = `${localBase.replace(/\/$/, '')}/${file}`;
    const t0 = performance.now();
    const r = await fetch(url, { signal });
    if (!r.ok) throw new Error(`${url} HTTP ${r.status}`);
    const total = Number(r.headers.get('content-length')) || 0;
    const bytes = new Uint8Array(await r.arrayBuffer());
    const secs = (performance.now() - t0) / 1000;
    onProgress && onProgress({ read: bytes.length, total: bytes.length, p: 1,
      host: 'local', phase: 'done', rate: bytes.length / 1e6 / Math.max(secs, 1e-6) });
    return { bytes, host: 'local', total: bytes.length, fromCache: false,
      secs, rate: bytes.length / 1e6 / Math.max(secs, 1e-6), ranged: false };
  }

  const key = `hf:${repo}/${revision}/${file}`;
  const t0 = performance.now();

  if (useCache) {
    const hit = await cacheGet(key);
    if (hit && hit.length) {
      const secs = (performance.now() - t0) / 1000;
      onProgress && onProgress({ read: hit.length, total: hit.length, p: 1,
        host: 'cache', phase: 'cache', rate: hit.length / 1e6 / Math.max(secs, 1e-6) });
      return { bytes: hit, host: 'cache', total: hit.length, fromCache: true,
        secs, rate: hit.length / 1e6 / Math.max(secs, 1e-6), ranged: false };
    }
  }

  const errors = [];
  for (const host of hosts) {
    if (signal && signal.aborted) throw new Error('aborted');
    const url = resolveUrl(host, repo, file, revision);
    onProgress && onProgress({ read: 0, total: 0, p: 0, host, phase: 'probe' });

    const info = await probe(url);
    if (!info.ok) {
      errors.push(`${host}: ${info.status || info.error}`);
      continue;
    }

    const total = info.total || 0;
    const wantParts = info.ranged && total > minRangedMB * 1e6 ? parts : 1;
    const chunk = Math.ceil(total / wantParts);
    const spans = [];
    for (let i = 0; i < total; i += chunk) {
      spans.push([i, Math.min(i + chunk, total)]);
    }

    let read = 0;
    const done = [];
    const started = performance.now();
    let last = 0;
    const tick = (phase) => {
      const now = performance.now();
      if (now - last < 80) return;
      last = now;
      const secs = (now - started) / 1000;
      onProgress && onProgress({
        read, total, p: total ? read / total : 0, host, phase,
        rate: read / 1e6 / Math.max(secs, 1e-6),
      });
    };

    try {
      // 并发有上限：一次开 8 路以上，镜像站会给 429，反而更慢。
      const queue = spans.slice();
      const workers = new Array(Math.min(wantParts, 8)).fill(0).map(async () => {
        for (;;) {
          const span = queue.shift();
          if (!span) return;
          const part = await fetchRange(url, span[0], span[1]);
          done.push(part);
          read += part.data.length;
          tick('download');
        }
      });
      await Promise.all(workers);

      if (read !== total) throw new Error(`short read ${read} of ${total}`);
      const bytes = wantParts === 1 ? done[0].data : combine(done, total);
      const secs = (performance.now() - t0) / 1000;
      const cached = await cachePut(key, bytes);
      onProgress && onProgress({ read: total, total, p: 1, host, phase: cached ? 'cached' : 'done',
        rate: total / 1e6 / Math.max(secs, 1e-6) });
      return { bytes, host, total, fromCache: false, secs,
        rate: total / 1e6 / Math.max(secs, 1e-6), ranged: wantParts > 1 };
    } catch (e) {
      errors.push(`${host}: ${String(e && e.message || e)}`);
      // 这一个源失败了，字节丢掉，换下一个源从头来
      read = 0;
      done.length = 0;
    }
  }

  throw new Error(`取不到 ${repo}/${file}（${errors.join(' | ')}）`);
}

/** 小文件专用：JSON。走同一个 fallback，失败同样把两个源都说出来。 */
export async function hfJSON(repo, file, opts = {}) {
  const { bytes, host } = await hfFetch({ repo, file, ...opts });
  return { json: JSON.parse(new TextDecoder().decode(bytes)), host, bytes };
}
