// Drives the demo in headless Chromium: settle, slice, hose, screenshots.
import { chromium } from 'playwright';

const browser = await chromium.launch({
  channel: 'chrome',
  args: ['--use-angle=gl', '--enable-unsafe-swiftshader', '--no-sandbox'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(String(e)));

await page.goto('http://localhost:8741/');
await page.waitForTimeout(3500); // cube drops and settles
await page.screenshot({ path: '/tmp/t1_settled.png' });

// slice tool: two strokes across the cube (screen center-ish)
await page.keyboard.press('2');
for (const x of [580, 700]) {
  await page.mouse.move(x, 200);
  await page.mouse.down();
  for (let y = 200; y <= 620; y += 30) {
    await page.mouse.move(x + (y - 200) * 0.05, y);
    await page.waitForTimeout(16);
  }
  await page.mouse.up();
  await page.waitForTimeout(300);
}
await page.waitForTimeout(1500);
await page.screenshot({ path: '/tmp/t2_sliced.png' });

// grab a piece and toss it
await page.keyboard.press('1');
await page.mouse.move(700, 430);
await page.mouse.down();
await page.mouse.move(850, 250, { steps: 20 });
await page.waitForTimeout(300);
await page.mouse.up();
await page.waitForTimeout(1200);
await page.screenshot({ path: '/tmp/t3_tossed.png' });

// hose: spray at the pieces for 6 seconds
await page.keyboard.press('3');
await page.mouse.move(640, 450);
await page.mouse.down();
await page.waitForTimeout(6000);
await page.mouse.up();
await page.waitForTimeout(2500);
await page.screenshot({ path: '/tmp/t4_hosed.png' });

const stats = await page.textContent('#stats');
console.log('STATS:', stats);
console.log('ERRORS:', errors.length ? errors.join('\n') : 'none');
await browser.close();
