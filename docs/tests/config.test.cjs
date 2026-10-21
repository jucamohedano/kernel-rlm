const assert = require('node:assert/strict');
const test = require('node:test');
const config = require('../next.config.js');

test('static docs use the independent project repository path', () => {
  assert.equal(config.basePath, '/kernel-rlm');
  assert.equal(config.output, 'export');
});

test('static hosting does not require a Next image server or route rewrites', () => {
  assert.equal(config.images.unoptimized, true);
  assert.equal(config.trailingSlash, true);
});
