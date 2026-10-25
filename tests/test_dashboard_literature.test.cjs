const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync('dashboard/app.js', 'utf8');

function node(tag) {
  return {tag, children: [], append(...items) { this.children.push(...items); }};
}
const context = vm.createContext({document: {
  createElement: node,
  createTextNode: value => ({textContent: value}),
}});
vm.runInContext(source.slice(source.indexOf('function el('), source.indexOf('function rows(')), context);

test('literature links open the maintained dashboard review', () => {
  const parent = node('div');
  context.inline(parent, '[Literature review](literature-review.md)');
  const anchor = parent.children.find(item => item.tag === 'a');
  assert.equal(anchor.href, '#literature');
});

test('primary references remain external links, not served repository paths', () => {
  const parent = node('div');
  context.inline(parent, '[Primary source](https://arxiv.org/abs/2512.24601)');
  const anchor = parent.children.find(item => item.tag === 'a');
  assert.equal(anchor.href, 'https://arxiv.org/abs/2512.24601');
  assert.equal(anchor.rel, 'noreferrer');
});
