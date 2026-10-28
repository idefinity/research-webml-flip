// .npy 读进来 → Float32Array。
//
// Kaggle 那边存的是 numpy 的默认格式，浏览器里没有 numpy，而这个格式简单到
// 值得自己读：一个 128 字节以内的 ASCII 头，后面紧跟着裸数据。
// 只支持我们真正会写出来的那几种 dtype，遇到别的就报错，不要悄悄按 f32 解。

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

const DTYPES = {
  '<f4': { size: 4, read: (dv, o) => dv.getFloat32(o, true) },
  '<f8': { size: 8, read: (dv, o) => dv.getFloat64(o, true) },
  '<f2': { size: 2, read: half },
  '<i8': { size: 8, read: (dv, o) => Number(dv.getBigInt64(o, true)) },
  '<i4': { size: 4, read: (dv, o) => dv.getInt32(o, true) },
  '|u1': { size: 1, read: (dv, o) => dv.getUint8(o) },
  '<u4': { size: 4, read: (dv, o) => dv.getUint32(o, true) },
};

function half(dv, o) {
  const h = dv.getUint16(o, true);
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 5.960464477539063e-8;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export function parseNPY(buf) {
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < MAGIC.length; i++) {
    if (u8[i] !== MAGIC[i]) throw new Error('不是 .npy');
  }
  const major = u8[6];
  const headerLen = major === 1 ? u8[8] | (u8[9] << 8) : u8[8] | (u8[9] << 8) | (u8[10] << 16) | (u8[11] << 24);
  const headStart = major === 1 ? 10 : 12;
  const head = new TextDecoder('latin1').decode(u8.subarray(headStart, headStart + headerLen));

  const descr = /'descr'\s*:\s*'([^']+)'/.exec(head);
  const shapeM = /'shape'\s*:\s*\(([^)]*)\)/.exec(head);
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(head);
  if (!descr || !shapeM) throw new Error('npy 头里没有 descr/shape: ' + head.slice(0, 120));
  const dtype = DTYPES[descr[1]];
  if (!dtype) throw new Error('暂不支持的 dtype ' + descr[1]);
  const shape = shapeM[1].trim() === '' ? []
    : shapeM[1].split(',').map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n));
  if (fortran && fortran[1] === 'True' && shape.length > 1) {
    throw new Error('fortran_order 的 npy 暂时不读');
  }

  const start = headStart + headerLen;
  const n = shape.reduce((a, b) => a * b, 1);
  const dv = new DataView(buf);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dtype.read(dv, start + i * dtype.size);
  return { data: out, shape, dtype: descr[1] };
}
