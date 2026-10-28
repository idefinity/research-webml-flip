// Drive a page in headless Chrome over CDP and print what it recorded.
//
// Why not playwright: this machine's Chrome cannot start its sandbox
// ("Operation not permitted"), so playwright's launch dies before the page
// does. CDP against a manually started Chrome works, and node 22 ships a
// WebSocket client, so there is nothing to install.
//
//   node scripts/browser/cdp_run.mjs http://127.0.0.1:8899/webgpu/hf-probe.html [ms]
//
// The page is expected to set window.__done = true and (optionally)
// window.__rows / window.__log. Both are printed.
const url = process.argv[2] || 'http://127.0.0.1:8899/';
const budgetMs = Number(process.argv[3] || 600000);
const PORT = Number(process.argv[4] || 9333);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function targets() {
  const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
  return await r.json();
}

class Cdp {
  constructor(ws) { this.ws = ws; this.id = 0; this.waiting = new Map(); this.logs = []; }
  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((res, rej) => this.waiting.set(id, { res, rej }));
  }
  onMessage(raw) {
    const msg = JSON.parse(raw);
    if (msg.id && this.waiting.has(msg.id)) {
      const { res, rej } = this.waiting.get(msg.id);
      this.waiting.delete(msg.id);
      msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
      return;
    }
    if (msg.method === 'Runtime.consoleAPICalled') {
      this.logs.push(msg.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
    }
    if (msg.method === 'Runtime.exceptionThrown') {
      const d = msg.params.exceptionDetails;
      this.logs.push('EXCEPTION ' + (d.exception?.description || d.text));
    }
  }
  async evalJS(expr) {
    const r = await this.send('Runtime.evaluate', {
      expression: expr, returnByValue: true, awaitPromise: true,
    });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval failed');
    return r.result.value;
  }
}

async function main() {
  // Attach by URL, not by "the first page": this box usually has an older
  // Chrome running probe pages of its own, and picking the wrong tab makes a
  // finished run look like a timeout.
  const list = await targets();
  let page = list.find((t) => t.type === 'page' && t.url && t.url.startsWith(url));
  if (!page) {
    const created = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(url)}`,
      { method: 'PUT' });
    const info = await created.json();
    await sleep(500);
    const again = await targets();
    page = again.find((t) => t.id === info.id) ||
           again.find((t) => t.type === 'page' && t.url && t.url.startsWith(url));
  }
  if (!page) throw new Error('no page target for ' + url);
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  const cdp = new Cdp(ws);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  ws.onmessage = (e) => cdp.onMessage(e.data);

  await cdp.send('Runtime.enable');
  await cdp.send('Page.enable');
  // editing a module and re-running must not serve last week's bytes
  await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
  await cdp.send('Log.enable').catch(() => {});
  await cdp.send('Page.navigate', { url });
  console.log('navigated', url);

  const t0 = Date.now();
  let done = false;
  while (Date.now() - t0 < budgetMs) {
    await sleep(2000);
    try {
      done = await cdp.evalJS('window.__done === true');
    } catch { /* page still loading */ }
    if (done) break;
  }
  const secs = (Date.now() - t0) / 1000;
  console.log(done ? `page finished in ${secs.toFixed(1)} s` : `TIMEOUT after ${secs.toFixed(1)} s`);

  const rows = await cdp.evalJS('window.__rows || []').catch(() => []);
  for (const [k, v] of rows) console.log(String(k).padEnd(26), v);
  const logTxt = await cdp.evalJS(
    'document.getElementById("out")?.textContent || (window.__log||"")').catch(() => '');
  if (logTxt.trim()) console.log('--- page log ---\n' + logTxt.trim());
  if (cdp.logs.length) console.log('--- console ---\n' + cdp.logs.slice(-12).join('\n'));
  ws.close();
  process.exit(done ? 0 : 1);
}

main().catch((e) => { console.error('runner failed:', e.message); process.exit(2); });
