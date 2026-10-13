import { test } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import Markdown from 'react-markdown';
import { markdownOptions } from '../src/markdown.js';

test('renders the nested material template and LaTeX fractions', () => {
  const text = '### 1. General Overview\n\nHypothetical fixture.\n\n### 2. Engineering Profile\n\n* **Example property:**\n  * **Knowledge:** Fixture definition.\n  * **Typical Value / Range:** 10–20\n  * **Unit / Scale:** Fixture scale\n  * **Formula / Equation:** $X = \\frac{a}{b}$';
  const html = renderToStaticMarkup(React.createElement(Markdown, markdownOptions, text));
  assert.match(html, /<h3>1\. General Overview<\/h3>/);
  assert.match(html, /<h3>2\. Engineering Profile<\/h3>/);
  assert.equal((html.match(/<ul>/g) || []).length, 2);
  assert.match(html, /class="katex"/);
  assert.match(html, /<mfrac>/);
});
