// binary_little_endian PLY → 渲染用的紧凑数组。
// 只认 3DGS 那一套属性；多余的（法线、高阶 SH）跳过但算进 stride。

const C0 = 0.28209479177387814;

export async function loadPLY(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`PLY ${res.status} ${url}`);

  const total = Number(res.headers.get('content-length')) || 0;
  let buf;
  if (!total || !res.body) {
    buf = await res.arrayBuffer();
    onProgress && onProgress(1);
  } else {
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress && onProgress(got / total);
    }
    const out = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    buf = out.buffer;
  }

  return parsePLY(buf);
}

export function parsePLY(buffer) {
  const bytes = new Uint8Array(buffer);

  // 头部是 ASCII，逐字节找 end_header
  let headEnd = -1;
  const probe = new TextDecoder('latin1').decode(bytes.subarray(0, Math.min(bytes.length, 4000)));
  for (const mark of ['end_header\n', 'end_header\r\n']) {
    const i = probe.indexOf(mark);
    if (i >= 0) { headEnd = i + mark.length; break; }
  }
  if (headEnd < 0) throw new Error('不是合法的 PLY：找不到 end_header');

  const head = probe.slice(0, headEnd);
  if (!head.includes('binary_little_endian')) throw new Error('只支持 binary_little_endian');

  let count = 0;
  const props = [];
  let inVertex = false;
  for (const raw of head.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.startsWith('element ')) {
      const [, name, n] = line.split(/\s+/);
      inVertex = name === 'vertex';
      if (inVertex) count = Number(n);
    } else if (line.startsWith('property ') && inVertex) {
      const parts = line.split(/\s+/);
      // property <type> <name>   |   property list ...
      if (parts[1] === 'list') continue;
      props.push({ type: parts[1], name: parts[2] });
    }
  }
  if (!count) throw new Error('PLY 里没有顶点');

  const sizeOf = (t) => {
    if (t === 'float' || t === 'float32' || t === 'int' || t === 'int32' || t === 'uint') return 4;
    if (t === 'double' || t === 'float64') return 8;
    if (t === 'uchar' || t === 'uint8' || t === 'char' || t === 'int8') return 1;
    if (t === 'short' || t === 'ushort' || t === 'int16' || t === 'uint16') return 2;
    return 4;
  };
  const stride = props.reduce((a, p) => a + sizeOf(p.type), 0);
  const need = headEnd + stride * count;
  if (bytes.length < need) throw new Error(`PLY 截断：要 ${need} 字节，只有 ${bytes.length}`);

  const at = Object.create(null);
  let off = 0;
  for (const p of props) { at[p.name] = { off, type: p.type }; off += sizeOf(p.type); }
  for (const k of ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']) {
    if (!(k in at)) throw new Error(`PLY 缺少属性 ${k}`);
  }

  // 全部是 4 字节浮点时走快路径：直接按 Float32Array 对齐读
  const allFloat = props.every((p) => sizeOf(p.type) === 4);
  const dv = allFloat ? null : new DataView(buffer);

  // 为了让快路径能对齐，先把正文拷一份到独立 buffer
  const body = bytes.subarray(headEnd);
  let f32 = null;
  if (allFloat) {
    const copy = new Uint8Array(count * stride);
    copy.set(body.subarray(0, count * stride));
    f32 = new Float32Array(copy.buffer);
  }
  const fdiv = stride / 4;

  const reads = {
    f(name, i) {
      const p = at[name];
      if (allFloat) return f32[i * fdiv + p.off / 4];
      return dv.getFloat32(headEnd + i * stride + p.off, true);
    },
  };

  const data = new Float32Array(count * 16);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];

  for (let i = 0; i < count; i++) {
    const x = reads.f('x', i);
    const y = reads.f('y', i);
    const z = reads.f('z', i);

    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;

    const o = i * 16;
    data[o] = x;
    data[o + 1] = y;
    data[o + 2] = z;
    data[o + 3] = 1 / (1 + Math.exp(-reads.f('opacity', i)));

    data[o + 4] = clamp01(0.5 + C0 * reads.f('f_dc_0', i));
    data[o + 5] = clamp01(0.5 + C0 * reads.f('f_dc_1', i));
    data[o + 6] = clamp01(0.5 + C0 * reads.f('f_dc_2', i));
    data[o + 7] = 0;

    data[o + 8] = Math.exp(reads.f('scale_0', i));
    data[o + 9] = Math.exp(reads.f('scale_1', i));
    data[o + 10] = Math.exp(reads.f('scale_2', i));
    data[o + 11] = 0;

    let qw = reads.f('rot_0', i);
    let qx = reads.f('rot_1', i);
    let qy = reads.f('rot_2', i);
    let qz = reads.f('rot_3', i);
    const len = Math.hypot(qw, qx, qy, qz) || 1;
    data[o + 12] = qw / len;
    data[o + 13] = qx / len;
    data[o + 14] = qy / len;
    data[o + 15] = qz / len;
  }

  const center = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const radius = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2 || 1;
  // 装取景框用的是包围球（半对角线），只用最长边会让纵深方向出画。
  const fitRadius = 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;

  // 台面高度取 2% 分位而不是最小值：几个飘在下面的 floater 能把整台子拉低，
  // 模型就悬在半空。
  const ys = new Float32Array(count);
  for (let i = 0; i < count; i++) ys[i] = data[i * 16 + 1];
  ys.sort();
  const floor = ys[Math.min(count - 1, Math.floor(count * 0.02))];

  return { count, data, min, max, center, radius, fitRadius, floor };
}

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
