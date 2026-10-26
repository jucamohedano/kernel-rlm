const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync('dashboard/app.js', 'utf8');

function node(tag) {
  return {
    tag, children: [], listeners: {},
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    get lastElementChild() { return this.children.at(-1); },
    setAttribute(name, value) { this[name] = value; },
    addEventListener(name, callback) { this.listeners[name] = callback; },
    scrollIntoView(options) { this.scrolled = options; },
    focus() { this.focused = true; },
  };
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

function descendants(parent, tag) {
  return parent.children.flatMap(child => [
    ...(child.tag === tag ? [child] : []),
    ...(child.children ? descendants(child, tag) : []),
  ]);
}
function textContent(parent) {
  return (parent.textContent || '') + (parent.children || []).map(textContent).join('');
}

test('literature opens topic introductions and collapses complete paper records', () => {
  const topics = node('nav');
  const article = context.literatureMarkdown(`# Review
Introductory context.
## Verification
Read correctness before speed.
### First paper
[Primary source](https://example.org/paper) · 2026
**Limits:** Finite checks only.
### Second paper
**Project connection:** Distinguish training from evaluation.
## Training
Keep evaluation independent.`, topics);
  assert.equal(descendants(article, 'h2').length, 2);
  assert.equal(descendants(article, 'details').length, 2);
  const [first, second] = descendants(article, 'details');
  assert.equal(first.open, undefined);
  assert.equal(first.children[0].tag, 'summary');
  assert.equal(textContent(first.children[0]), 'First paper');
  assert.match(textContent(first), /2026.*Limits:.*Finite checks only/s);
  assert.match(textContent(second), /Project connection:/);
  assert.equal(first.children[1].children[0].tag, 'p');
  const firstTopic = article.children.find(child => child.tag === 'section');
  assert.match(textContent(firstTopic.children[1]), /Read correctness before speed/);
  assert.equal(topics.children.length, 2);
  assert.equal(topics.children[0].tag, 'button');
  topics.children[1].listeners.click();
  const target = descendants(article, 'h2')[1];
  assert.equal(target.scrolled.block, 'start');
  assert.equal(target.focused, true);
  assert.equal(topics.children[1].href, undefined);
});

test('heading-like code remains code and does not create paper entries or topics', () => {
  const topics = node('nav');
  const article = context.literatureMarkdown('## Topic\n```md\n### Not a paper\n## Not a topic\n```', topics);
  assert.equal(topics.children.length, 1);
  assert.equal(descendants(article, 'details').length, 0);
  assert.equal(descendants(article, 'pre').length, 1);
});

test('only exact trusted local figures load and preserve accessible captions', () => {
  for (const path of ['/literature/research-map.svg', '/literature/train-evaluate.svg', '/literature/rlm-figure-2.png', '/literature/kernelbench-figure-1.png', '/literature/rao-reward.svg', '/literature/rao-credit-ablation.svg', '/literature/rao-delegation-ablation.svg']) {
    const article = context.markdown(`![Architecture and attribution](${path})`);
    const [image] = descendants(article, 'img');
    assert.equal(image.src, path);
    assert.equal(image.alt, 'Architecture and attribution');
    const [fullSize] = descendants(article, 'a');
    assert.equal(fullSize.href, path);
    assert.equal(fullSize.target, '_blank');
    assert.equal(fullSize.rel, 'noopener');
    assert.match(fullSize['aria-label'], /Open full-size figure/);
    assert.equal(textContent(descendants(article, 'figcaption')[0]), image.alt);
  }
  for (const path of ['https://example.org/image.svg', '//example.org/image.png',
    '/literature/../server.py', '/literature/research-map.svg?remote=1',
    '/literature/unknown.svg', '/literature/harness-context.svg', 'javascript:alert(1)', 'data:image/svg+xml,x']) {
    const article = context.markdown(`![Blocked](${path})`);
    assert.equal(descendants(article, 'img').length, 0);
    assert.match(textContent(article), /Blocked/);
  }
});

test('raw HTML and untrusted links remain text, including in paper titles', () => {
  const topics = node('nav');
  const article = context.literatureMarkdown('## Topic\n### <img src=x onerror=alert(1)>\n<script>alert(1)</script>\n[unsafe](javascript:evil)', topics);
  assert.equal(descendants(article, 'img').length, 0);
  assert.equal(descendants(article, 'script').length, 0);
  assert.match(textContent(article), /<script>alert\(1\)<\/script>/);
  assert.equal(descendants(article, 'a')[0].href, undefined);
});
