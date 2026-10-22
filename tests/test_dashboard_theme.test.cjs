const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync('dashboard/theme.js', 'utf8');
const storageKey = 'kernel-rlm-theme';

function loadTheme({dark = false, storage = new Map(), blocked = false} = {}) {
  const root = {dataset: {}};
  const button = {attributes: {}, setAttribute(name, value) { this.attributes[name] = value; }};
  const preference = {matches: dark, addEventListener(event, callback) { this.change = callback; }};
  const document = {
    documentElement: root,
    addEventListener(event, callback) { this.ready = callback; },
    querySelector(selector) { assert.equal(selector, '#theme-toggle'); return button; },
  };
  button.addEventListener = (event, callback) => { button.click = callback; };
  vm.runInNewContext(source, {
    document,
    window: {
      matchMedia(query) { assert.equal(query, '(prefers-color-scheme: dark)'); return preference; },
      localStorage: {
        getItem(key) { if (blocked) throw new Error('Storage disabled'); return storage.get(key) ?? null; },
        setItem(key, value) { if (blocked) throw new Error('Storage disabled'); storage.set(key, value); },
      },
    },
  });
  return {root, button, preference, ready: () => document.ready()};
}

test('system preference sets the initial palette before the button exists', () => {
  for (const dark of [false, true]) {
    const storage = new Map();
    const page = loadTheme({dark, storage});
    assert.equal(page.root.dataset.theme, dark ? 'dark' : 'light');
    page.ready();
    assert.equal(page.button.attributes['aria-pressed'], String(dark));
    page.preference.change({matches: !dark});
    assert.equal(page.root.dataset.theme, dark ? 'light' : 'dark');
    assert.equal(page.button.attributes['aria-pressed'], String(!dark));
    assert.equal(storage.size, 0, 'system preference is not stored as an override');
  }
});

test('toggle updates accessible state and persists both choices across reloads', () => {
  const storage = new Map();
  const page = loadTheme({storage});
  page.ready();
  page.button.click();
  assert.equal(page.root.dataset.theme, 'dark');
  assert.equal(page.button.attributes['aria-pressed'], 'true');
  assert.equal(storage.get(storageKey), 'dark');
  const reloaded = loadTheme({dark: false, storage});
  reloaded.ready();
  assert.equal(reloaded.root.dataset.theme, 'dark');
  assert.equal(reloaded.button.attributes['aria-pressed'], 'true');
  reloaded.preference.change({matches: false});
  assert.equal(reloaded.root.dataset.theme, 'dark', 'manual override wins over system changes');
  reloaded.button.click();
  assert.equal(reloaded.button.attributes['aria-pressed'], 'false');
  assert.equal(storage.get(storageKey), 'light');
  const lightReload = loadTheme({dark: true, storage});
  lightReload.ready();
  assert.equal(lightReload.root.dataset.theme, 'light', 'saved light overrides a dark OS');
  assert.equal(lightReload.button.attributes['aria-pressed'], 'false');
});

test('invalid saved choices use the system default until the user toggles', () => {
  const storage = new Map([[storageKey, 'invalid']]);
  const page = loadTheme({dark: true, storage});
  assert.equal(page.root.dataset.theme, 'dark');
  page.ready();
  page.preference.change({matches: false});
  assert.equal(page.root.dataset.theme, 'light');
  page.button.click();
  page.preference.change({matches: false});
  assert.equal(page.root.dataset.theme, 'dark');
  assert.equal(storage.get(storageKey), 'dark');
});

test('blocked storage does not stop initialization or manual toggling', () => {
  const page = loadTheme({dark: true, blocked: true});
  assert.equal(page.root.dataset.theme, 'dark');
  page.ready();
  page.button.click();
  assert.equal(page.root.dataset.theme, 'light');
  assert.equal(page.button.attributes['aria-pressed'], 'false');
  page.preference.change({matches: true});
  assert.equal(page.root.dataset.theme, 'light');
});
