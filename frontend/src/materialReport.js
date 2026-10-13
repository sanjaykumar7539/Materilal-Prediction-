import React from 'react';
import Markdown from 'react-markdown';
import { markdownOptions } from './markdown.js';

// Split only our explicit report envelope. Ordinary answers retain their original rendering.
export function splitMaterialReport(text) {
  if (!text.startsWith('CAMPUSMIND MATERIALS INTELLIGENCE\n\n# ')) return null;
  const headings = [...text.matchAll(/^## (0[1-9]|1[0-2])\. (.+)$/gm)];
  if (!headings.length) return null;
  const numbers = headings.map(match => Number(match[1]));
  if (new Set(numbers).size !== numbers.length || numbers.some((n, i) => i > 0 && n <= numbers[i - 1])) return null;
  return {
    intro: text.slice(0, headings[0].index),
    sections: headings.map((match, i) => ({
      number: Number(match[1]), title: match[2],
      body: text.slice(match.index + match[0].length, headings[i + 1]?.index ?? text.length).trim(),
    })),
  };
}

const h = React.createElement;
const components = {
  table: ({ node, ...props }) => h('div', { className: 'report-table', role: 'region', 'aria-label': 'Material data table', tabIndex: 0 }, h('table', props)),
};
const markdown = text => h(Markdown, { ...markdownOptions, components }, text);

export function MaterialMessage({ text }) {
  const report = splitMaterialReport(text);
  if (!report) return markdown(text);
  return h('div', { className: 'material-report' },
    h('div', { className: 'report-intro' }, markdown(report.intro)),
    ...report.sections.map(section => h('details', {
      className: 'report-section', key: section.number, open: [1, 2, 12].includes(section.number),
    }, h('summary', null, h('span', { className: 'report-number' }, String(section.number).padStart(2, '0')), section.title),
    h('div', { className: 'report-section-body' }, markdown(section.body)))),
  );
}
