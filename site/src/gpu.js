// 一块设备，两个阶段共用。
// 加载动画要它，结论页要它；中间的推理阶段一次都不提交，
// 让出整个队列——这是这个应用唯一硬性的性能约定。

let pending = null;

export async function acquireGPU() {
  if (pending) return pending;
  pending = (async () => {
    // ?nogpu=1 是给降级路径留的开关：没有它，这条分支只能靠换浏览器才测得到。
    if (new URLSearchParams(location.search).get('nogpu') === '1') return null;
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;

    // 默认 16 KB 的 workgroup storage 会让大 tile 的 pipeline 静默失效
    // （创建成 invalid，dispatch 什么也不做）。这里直接要满。
    const limits = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxBufferSize: limits.maxBufferSize,
      },
    });

    device.addEventListener('uncapturederror', (e) => {
      console.error('[gpu]', e.error.message);
    });

    const info = adapter.info || {};
    return {
      adapter,
      device,
      label: [info.vendor, info.architecture].filter(Boolean).join(' ') || 'unknown GPU',
      features: [...adapter.features],
    };
  })();
  return pending;
}

/** 编译一个 shader module，编译错误直接抛出来——静默的 invalid pipeline 是最贵的 bug。 */
export async function makeModule(device, code, label) {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    const text = errors.map((e) => `${label} ${e.lineNum}:${e.linePos} ${e.message}`).join('\n');
    throw new Error(text);
  }
  return module;
}

/** 全屏三角形的顶点着色器，三个页面都用得着。 */
export const FULLSCREEN_VS = /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}
`;
