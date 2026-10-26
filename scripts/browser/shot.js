// Screenshot one splat PLY from one camera angle, with real waiting.
// Usage: node shot.js <ply> <az> <out> [el] [pad] [fov] [c] [r]
//   c = "x,y,z" centre, r = radius. Supplied by shoot.py from the PLY itself,
//   because Spark's SplatMesh does not expose a stable Box3 early enough.
const { chromium } = require("playwright");

const [, , ply, az, out, el = "0", pad = "1.45", fov = "40", c = "", r = "0"] = process.argv;
const qs = new URLSearchParams({ ply, az, el, pad, fov });
if (c) qs.set("c", c);
if (r && r !== "0") qs.set("r", r);

(async () => {
  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--enable-unsafe-swiftshader", "--hide-scrollbars"],
  });
  const page = await browser.newPage({ viewport: { width: 900, height: 900 }, deviceScaleFactor: 1 });
  const logs = [];
  page.on("console", (m) => { if (m.type() === "error") logs.push(m.text()); });
  page.on("pageerror", (e) => logs.push("pageerror: " + e.message));

  await page.goto(`http://127.0.0.1:8765/index.html?${qs}`, { waitUntil: "load", timeout: 60000 });
  let title = "";
  try {
    await page.waitForFunction(() => document.title === "READY", null, { timeout: 60000 });
    title = "READY";
  } catch (e) {
    title = await page.title();
  }
  const msg = await page.evaluate(() => document.getElementById("msg").textContent);
  await page.screenshot({ path: out });
  console.log(JSON.stringify({ ply, az, title, msg, logs: logs.slice(0, 6) }));
  await browser.close();
})();
