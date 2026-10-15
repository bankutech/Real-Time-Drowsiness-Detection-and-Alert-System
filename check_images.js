const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'templates', 'index.html'), 'utf8');
const regex = /src=["'](\/outputs\/[^"']+)["']/g;
let match;
let missing = 0;
while ((match = regex.exec(html)) !== null) {
    const url = match[1];
    const diskPath = path.join(__dirname, url.replace(/^\//, ''));
    const exists = fs.existsSync(diskPath);
    console.log(`${exists ? '[OK]' : '[MISSING]'} ${url} -> ${diskPath}`);
    if (!exists) missing++;
}

console.log(`\nTotal Missing Images: ${missing}`);
if (missing > 0) process.exit(1);
