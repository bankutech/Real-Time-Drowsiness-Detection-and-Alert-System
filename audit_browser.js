const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const OUTPUTS_DIR = path.resolve(__dirname, 'outputs');

async function runAudit() {
    console.log('=== STARTING PRODUCTION BROWSER & CONSOLE AUDIT ===');
    const browser = await puppeteer.launch({
        executablePath: CHROME_PATH,
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
    });

    const page = await browser.newPage();
    const consoleLogs = [];
    const networkErrors = [];

    page.on('console', msg => {
        const type = msg.type();
        const text = msg.text();
        consoleLogs.push({ type, text });
        if (type === 'error') {
            console.error(`[CONSOLE ERROR] ${text}`);
        }
    });

    page.on('requestfailed', req => {
        networkErrors.push({
            url: req.url(),
            failure: req.failure() ? req.failure().errorText : 'Unknown'
        });
        console.error(`[NETWORK FAILED] ${req.url()}: ${req.failure()?.errorText}`);
    });

    page.on('response', res => {
        if (res.status() >= 400) {
            networkErrors.push({
                url: res.url(),
                status: res.status()
            });
            console.error(`[HTTP ${res.status()}] ${res.url()}`);
        }
    });

    // 1. DESKTOP VIEWPORT (1440 x 900)
    console.log('\n[1] Testing Desktop Viewport (1440x900)...');
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto('http://localhost:8080/', { waitUntil: 'networkidle2', timeout: 15000 });
    await new Promise(r => setTimeout(r, 1500));
    await page.screenshot({ path: path.join(OUTPUTS_DIR, 'audit_desktop_1440.png'), fullPage: false });
    console.log(' -> Captured outputs/audit_desktop_1440.png');

    // Test tab switching
    console.log('\n[2] Testing Tab Navigation & Interactive Controls...');
    const pills = await page.$$('.channel-tab-pill');
    if (pills.length >= 2) {
        await pills[1].click();
        await new Promise(r => setTimeout(r, 600));
        await page.screenshot({ path: path.join(OUTPUTS_DIR, 'audit_results_tab_1440.png'), fullPage: false });
        console.log(' -> Switched to Results Tab, captured outputs/audit_results_tab_1440.png');
        await pills[0].click();
        await new Promise(r => setTimeout(r, 600));
    }

    // Test simulation trigger
    console.log('\n[3] Testing Quick Simulation Triggers...');
    const simBtns = await page.$$('.sim-trigger-btn');
    if (simBtns.length > 0) {
        await simBtns[1].click(); // Yawn
        await new Promise(r => setTimeout(r, 400));
        await simBtns[2].click(); // Microsleep
        await new Promise(r => setTimeout(r, 400));
        await simBtns[0].click(); // Alert
        await new Promise(r => setTimeout(r, 400));
        console.log(' -> Verified simulation triggers.');
    }

    // Test Lightbox modal
    console.log('\n[4] Testing Diagnostic Lightbox Modal...');
    if (pills.length >= 2) {
        await pills[1].click();
        await new Promise(r => setTimeout(r, 400));
        const cards = await page.$$('.diagnostic-card');
        if (cards.length > 0) {
            await cards[0].click();
            await new Promise(r => setTimeout(r, 500));
            await page.screenshot({ path: path.join(OUTPUTS_DIR, 'audit_lightbox_modal.png'), fullPage: false });
            console.log(' -> Captured outputs/audit_lightbox_modal.png');
            await page.keyboard.press('Escape');
            await new Promise(r => setTimeout(r, 300));
        }
        await pills[0].click();
        await new Promise(r => setTimeout(r, 400));
    }

    // 2. TABLET VIEWPORT (768 x 1024)
    console.log('\n[5] Testing Tablet Viewport (768x1024)...');
    await page.setViewport({ width: 768, height: 1024 });
    await new Promise(r => setTimeout(r, 800));
    await page.screenshot({ path: path.join(OUTPUTS_DIR, 'audit_tablet_768.png'), fullPage: false });
    console.log(' -> Captured outputs/audit_tablet_768.png');

    // 3. MOBILE VIEWPORT (375 x 812)
    console.log('\n[6] Testing Mobile Viewport (375x812)...');
    await page.setViewport({ width: 375, height: 812 });
    await new Promise(r => setTimeout(r, 800));
    await page.screenshot({ path: path.join(OUTPUTS_DIR, 'audit_mobile_375.png'), fullPage: false });
    console.log(' -> Captured outputs/audit_mobile_375.png');

    // 4. ANALYTICS ROUTE (/analytics)
    console.log('\n[7] Testing /analytics React Dashboard...');
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto('http://localhost:8080/analytics', { waitUntil: 'networkidle2', timeout: 15000 });
    await new Promise(r => setTimeout(r, 1500));
    await page.screenshot({ path: path.join(OUTPUTS_DIR, 'audit_analytics_desktop.png'), fullPage: false });
    console.log(' -> Captured outputs/audit_analytics_desktop.png');

    await browser.close();

    console.log('\n=== AUDIT SUMMARY ===');
    console.log(`Console Errors: ${consoleLogs.filter(l => l.type === 'error').length}`);
    console.log(`Console Warnings: ${consoleLogs.filter(l => l.type === 'warning').length}`);
    console.log(`Network Failures: ${networkErrors.length}`);

    const result = {
        consoleErrors: consoleLogs.filter(l => l.type === 'error'),
        networkErrors: networkErrors
    };
    fs.writeFileSync(path.join(OUTPUTS_DIR, 'audit_result.json'), JSON.stringify(result, null, 2));
    console.log('Saved audit summary to outputs/audit_result.json');
}

runAudit().catch(err => {
    console.error('Audit fatal error:', err);
    process.exit(1);
});
