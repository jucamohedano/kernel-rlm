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
