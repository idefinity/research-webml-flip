// Load a WebGPU bench page, wait for it to finish, print what it says.
//
// Usage: node run_page.js <url> [--unsafe]
//   --unsafe adds the flag that exposes experimental WebGPU features
//   (chromium-experimental-subgroup-matrix needs it on current Chrome).
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const url = args.find(a => !a.startsWith("--"));
const unsafe = args.includes("--unsafe");

(async () => {
  const launch = {
    channel: "chrome",
    headless: true,
    args: ["--hide-scrollbars", "--use-angle=metal"],
  };
  if (unsafe) launch.args.push("--enable-unsafe-webgpu");

  const browser = await chromium.launch(launch);
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const errs = [];
  page.on("pageerror", (e) => errs.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });

  await page.goto(url, { waitUntil: "load", timeout: 60000 });
  try {
    await page.waitForFunction(() => document.title === "READY", null, { timeout: 120000 });
  } catch (e) {
    errs.push("timeout waiting for READY");
  }
  const text = await page.evaluate(() => document.getElementById("out")?.textContent || "");
  console.log(text.trim());
  if (errs.length) console.log("console: " + errs.slice(0, 4).join(" | "));
  await browser.close();
})();
