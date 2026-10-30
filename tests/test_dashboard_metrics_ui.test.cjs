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
const usageStart = source.indexOf("function tokenUsageLabel(");
const usageEnd = source.indexOf("function renderChildCalls(", usageStart);
vm.runInContext(source.slice(usageStart, usageEnd), context);
const historyStart = source.indexOf("function experimentHistoryContext(");
const historyEnd = source.indexOf("async function experimentsPage(", historyStart);
vm.runInContext(source.slice(historyStart, historyEnd), context);

test("history context is limited to the matching experiment group", () => {
  assert.equal(context.experimentHistoryContext("New base phase"), null);
  const historical = context.experimentHistoryContext("L1 post-training");
  assert.equal(historical.className, "experiment-context");
  assert.ok(historical.children.some(child => child.href === "#writing/first-campaign"));
});

test('run aggregate preserves unknown coverage instead of summing null as zero', () => {
  const unknown = {sum: null, recorded: 0, total: null};
  const partial = {sum: 7, recorded: 1, total: 2};
  const result = context.aggregateRunUsage([
    {metrics: {token_usage: {root: {input_tokens: unknown}}}},
    {metrics: {token_usage: {root: {input_tokens: partial}}}},
  ]);
  assert.deepEqual(JSON.parse(JSON.stringify(result.root.input_tokens)), {sum: 7, recorded: 1, total: null});
});

test('per-call usage distinguishes output cap from context window and keeps missing caps unknown', () => {
  const label = context.tokenUsageLabel({prompt_tokens: 12, completion_tokens: 3}, {
    output_tokens: {limit: 8, source: "effective"},
    context_tokens: {limit: 100, source: "config.model_context_tokens"},
  }).textContent;
  assert.match(label, /Output used\/cap: 3 \/ 8 \(effective\)/);
  assert.match(label, /Context used\/window: 15 \/ 100 \(recorded\)/);
  const unknown = context.tokenUsageLabel({completion_tokens: 0}, {}).textContent;
  assert.match(unknown, /Output used\/cap: 0 \/ Unknown/);
  assert.match(unknown, /Context used\/window: Unknown \/ Unknown/);
});

test('aggregate output used/cap requires matching complete coverage', () => {
  const complete = context.tokenPanel('usage', {
    output_tokens: {sum: 12, recorded: 2, total: 2},
    output_cap_tokens: {sum: 20, recorded: 2, total: 2},
  });
  assert.ok(created.some(item => item.textContent === 'Output used / sum of per-call caps: 12 / 20 tokens (2/2 calls)'));
  context.tokenPanel('usage', {
    output_tokens: {sum: 5, recorded: 1, total: 2},
    output_cap_tokens: {sum: 10, recorded: 1, total: 2},
  });
  assert.ok(created.some(item => item.textContent === 'Output used / sum of per-call caps: 5 / Unknown'));
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
