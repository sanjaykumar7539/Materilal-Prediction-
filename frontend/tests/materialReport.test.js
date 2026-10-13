import { test } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MaterialMessage, splitMaterialReport } from '../src/materialReport.js';
import { buildRequest } from '../src/api.js';

const report = 'CAMPUSMIND MATERIALS INTELLIGENCE\n\n# Copper\n\n## 01. Material Overview\n\nReviewed description.\n\n## 02. Complete Material Properties\n\n| Property | Value |\n| --- | --- |\n| Example | 10 |\n\n## 06. AI Property Prediction\n\nQualitative only.\n\n## 12. Final Engineering Recommendation\n\nTest first.';

test('material reports have accessible native expandable sections and real tables', () => {
  const html = renderToStaticMarkup(React.createElement(MaterialMessage, { text: report }));
  assert.equal((html.match(/<details/g) || []).length, 4);
  assert.equal((html.match(/open=""/g) || []).length, 3);
  assert.match(html, /<summary>/);
  assert.match(html, /<table>/);
  assert.match(html, /<th>Property<\/th>/);
  assert.match(html, /Qualitative only/);
  assert.match(html, /aria-label="Material data table"/);
});

test('ordinary replies and malformed section envelopes stay ordinary Markdown', () => {
  assert.equal(splitMaterialReport('## 01. Material Overview'), null);
  assert.equal(splitMaterialReport(report + '\n\n## 01. Duplicate\n\nBad order'), null);
  const html = renderToStaticMarkup(React.createElement(MaterialMessage, { text: 'Direct answer.' }));
  assert.doesNotMatch(html, /<details/);
});

test('rendering does not execute untrusted HTML', () => {
  const html = renderToStaticMarkup(React.createElement(MaterialMessage, { text: report + '<script>alert(1)</script>' }));
  assert.doesNotMatch(html, /<script>/);
});

test('prior user identity hint does not cross failed turns or new conversations', () => {
  const settings = { domain: 'materials', tokens: 512, temperature: 0.7 };
  assert.equal(buildRequest([{ role: 'user', content: 'Copper' }, { role: 'assistant', content: 'Reviewed report' }, { role: 'user', content: 'improve it' }], settings).previous_material, 'Copper');
  assert.equal(buildRequest([{ role: 'user', content: 'Copper', failed: true }, { role: 'user', content: 'improve it' }], settings).previous_material, undefined);
  assert.equal(buildRequest([{ role: 'user', content: 'improve it' }], settings).previous_material, undefined);
});
