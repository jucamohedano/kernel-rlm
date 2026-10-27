const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync('dashboard/app.js', 'utf8');
const start = source.indexOf('function readableChildText(');
const end = source.indexOf('function renderChildCalls(', start);
const context = vm.createContext({});
vm.runInContext(source.slice(start,end),context);
test('child response displays content without transport token arrays', () => {
  const response = {prompt_token_ids: [123,456], choices: [{message: {content:'Readable reply', reasoning:'private reasoning'}, token_ids:[789]}]};
  assert.equal(context.readableChildText(response),'Readable reply');
  assert.equal(context.readableChildText({prompt_token_ids:[123]}),null);
  assert.equal(context.readableChildText([{type:'text',text:'first'},{type:'image',data:'ignored'},{type:'text',text:'second'}]),'first\nsecond');
});

test('iterative child turns render raw replies and executed observations separately', () => {
  const start = source.indexOf('function renderChildCalls(');
  const end = source.indexOf('function renderTurnTimeline(', start);
  const context = vm.createContext({
    el: (_tag, _className, value='') => ({children:[], textContent:value, append(...items){this.children.push(...items);}}),
    panel: (title, value) => ({title, value, append(){}}),
    readableChildText: value => typeof value === 'string' ? value : null,
  });
  vm.runInContext(source.slice(start,end),context);
  const parent={children:[],append(...items){this.children.push(...items);}};
  context.renderChildCalls([{
    child_invocation_id:'child-1', parent_turn_id:'root-1', mode:'compile_repair',
    status:'failed', completion_tokens:12, child_total_tokens:4096,
    turns:[{step:1,finish_reason:'stop',observed_completion_tokens:12,
      raw_reply:{content:'<tool_call>literal'},format_error:'malformed',outcome:'malformed_action'},
      {step:2,outcome:'check_executed',parsed_action:{action:'check'},
        observation:{compiled:true}}],
  }],parent);
  assert.equal(parent.children.some(child=>child.title==='Error'), false);
  function texts(node){return [node.textContent,node.title,node.value,...(node.children||[]).flatMap(texts)].filter(Boolean).join(" ");}
  const serialized=texts(parent);
  assert.match(serialized,/Malformed action · not executed/);
  assert.match(serialized,/<tool_call>literal/);
  assert.match(serialized,/Compiler\/check observation/);
});
