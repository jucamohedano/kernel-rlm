const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync('dashboard/app.js', 'utf8');
const start = source.indexOf('function el(');
const end = source.indexOf('function rows(', start);
const created = [];
function node() { const item = {children: [], append(...items) { this.children.push(...items); }}; created.push(item); return item; }
const context = vm.createContext({document: {createElement: node}});
vm.runInContext(source.slice(start, end), context);

test('run aggregate preserves unknown coverage instead of summing null as zero', () => {
  const unknown = {sum: null, recorded: 0, total: null};
  const partial = {sum: 7, recorded: 1, total: 2};
  const result = context.aggregateRunUsage([
    {metrics: {token_usage: {root: {input_tokens: unknown}}}},
    {metrics: {token_usage: {root: {input_tokens: partial}}}},
  ]);
  assert.deepEqual(JSON.parse(JSON.stringify(result.root.input_tokens)), {sum: 7, recorded: 1, total: null});
});

test('token panel labels unknown separately from explicit zero', () => {
  const panel = context.tokenPanel('usage', {
    input_tokens: {sum: null, recorded: 0, total: null},
    output_tokens: {sum: 0, recorded: 1, total: 1},
    total_tokens: {sum: null, recorded: 0, total: 1},
  });
  const labels = created.map(item => item.textContent);
  assert.ok(labels.includes('Unknown'));
  assert.ok(labels.includes('0'));
  assert.ok(labels.includes('0 calls recorded · coverage unknown'));
  assert.ok(labels.includes('1/1 calls recorded · complete'));
});

test('run aggregate sums recorded elapsed calls with null-safe coverage', () => {
  const partial = {sum: 12.5, recorded: 2, total: 3};
  const result = context.aggregateRunUsage([
    {metrics: {token_usage: {root: {elapsed_seconds: partial}}}},
    {metrics: {token_usage: {root: {elapsed_seconds: {sum: 4, recorded: 1, total: 1}}}}},
  ]);
  assert.deepEqual(JSON.parse(JSON.stringify(result.root.elapsed_seconds)), {sum: 16.5, recorded: 3, total: 4});
  const unknown = context.aggregateRunUsage([
    {metrics: {token_usage: {children: {elapsed_seconds: {sum: null, recorded: 0, total: null}}}}},
  ]);
  assert.deepEqual(JSON.parse(JSON.stringify(unknown.children.elapsed_seconds)), {sum: null, recorded: 0, total: null});
});
