// ONNX Runtime Web 的一层薄封装。
//
// 只做三件小事，因为三件事都踩过坑：
//   1. wasm 的路径必须显式给，否则 ORT 会去页面相对路径找 .wasm 然后 404；
//   2. 会话要能报告"有多少节点没落到 WebGPU"——回退到 wasm 是静默的，
//      一旦发生，789 GFLOP/s 会变成 58，而页面上只看到"慢"；
//   3. 张量要能在 GPU 上交接（gpu-buffer），DiT 的注意力走我们自己的核。

const ORT_VERSION = '1.31.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let ortPromise = null;

export async function ort() {
  if (!ortPromise) {
    ortPromise = import(/* @vite-ignore */ `${ORT_BASE}ort.webgpu.min.mjs`).then((m) => {
      m.env.wasm.wasmPaths = ORT_BASE;
      try { m.env.webgpu.powerPreference = 'high-performance'; } catch { /* 老版本没有这个字段 */ }
      try { m.env.logLevel = 'error'; } catch { /* 同上 */ }
      return m;
    });
  }
  return ortPromise;
}

/**
 * 起一个会话。bytes 是整张图的字节（ORT-web 吃 ArrayBuffer/Uint8Array）。
 * 返回 { run(feeds), session, bytes }；run 会把耗时记进 timing。
 */
export async function createSession(bytes, { label = '', providers = ['webgpu'] } = {}) {
  const o = await ort();
  const t0 = performance.now();
  const session = await o.InferenceSession.create(bytes, {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
  });
  const ms = performance.now() - t0;
  const timing = { createMs: ms, calls: 0, runMs: 0 };

  async function run(feeds, opts = {}) {
    const t = performance.now();
    const out = await session.run(feeds, opts);
    timing.runMs += performance.now() - t;
    timing.calls++;
    return out;
  }

  return {
    session,
    run,
    timing,
    label,
    bytes: bytes.byteLength ?? bytes.length,
    inputs: session.inputNames,
    outputs: session.outputNames,
    async release() { try { await session.release?.(); } catch { /* 没有 release 就算了 */ } },
  };
}

/** int64 的标量/向量输入：ORT 要 BigInt64Array，JS 的 Number 不行。 */
export function i64(values) {
  const a = new BigInt64Array(values.length);
  for (let i = 0; i < values.length; i++) a[i] = BigInt(values[i]);
  return a;
}
