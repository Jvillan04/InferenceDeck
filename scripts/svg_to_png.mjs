// Convert docs/screenshots/*.svg (from scripts/screenshots.py) to PNG with Playwright's Chromium.
//   node scripts/svg_to_png.mjs
// The SVGs reference a web font; it is stripped so the local monospace font is used and nothing is fetched.
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const dir = join(dirname(fileURLToPath(import.meta.url)), "..", "docs", "screenshots");

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 2 });
await page.route("**/*", (route) => (route.request().url().startsWith("data:") ? route.continue() : route.abort()));
for (const name of readdirSync(dir).filter((f) => f.endsWith(".svg"))) {
  const svg = readFileSync(join(dir, name), "utf8").replace(/@font-face[^}]*}/g, "");
  await page.setContent(`<html><body style="margin:0;background:transparent">${svg}</body></html>`);
  await page.locator("svg").first().screenshot({ path: join(dir, name.replace(/\.svg$/, ".png")), omitBackground: true });
  console.log(`wrote ${name.replace(/\.svg$/, ".png")}`);
}
await browser.close();
