document.querySelectorAll('[data-tab]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-tab]').forEach(tab => tab.setAttribute('aria-selected', String(tab === button)));
    document.querySelectorAll('.panel').forEach(panel => { panel.hidden = panel.id !== button.dataset.tab; });
    const url = new URL(location.href);
    if (url.searchParams.has('run')) {
      url.searchParams.set('tab', button.dataset.tab);
      history.replaceState(null, '', url);
    }
  });
});
document.querySelector('#trajectory')?.addEventListener('change', event => {
  const url = new URL(location.href);
  url.searchParams.set('trajectory', event.target.value);
  location.href = url;
});
document.querySelector('#expand')?.addEventListener('change', event => {
  document.querySelectorAll('#trace details').forEach(detail => { detail.open = event.target.checked; });
});
function filterRuns() {
  const query = (document.querySelector('#run-search')?.value || '').toLowerCase();
  const status = document.querySelector('#run-status')?.value || '';
  const runs = [...document.querySelectorAll('[data-run]')];
  runs.forEach(run => {
    run.hidden = !run.textContent.toLowerCase().includes(query) || Boolean(status && !run.querySelector(`.pill.${status}`));
  });
  const counter = document.querySelector('#run-count');
  if (counter) counter.textContent = `${runs.filter(run => !run.hidden).length} of ${runs.length} shown`;
}
document.querySelector('#run-search')?.addEventListener('input', filterRuns);
document.querySelector('#run-status')?.addEventListener('change', filterRuns);
filterRuns();
document.querySelector('#experiment-search')?.addEventListener('input', event => {
  const query = event.target.value.toLowerCase();
  document.querySelectorAll('[data-experiment]').forEach(item => {
    item.hidden = !item.textContent.toLowerCase().includes(query);
  });
});
document.querySelector('#collapse')?.addEventListener('click', () => {
  document.querySelectorAll('details').forEach(detail => { detail.open = false; });
});
document.querySelector('#expand-charts')?.addEventListener('click', event => {
  const expanded = document.querySelector('.workspace').classList.toggle('wide-charts');
  event.target.setAttribute('aria-expanded', String(expanded));
  event.target.textContent = expanded ? 'Back to trace' : 'Expand charts';
});
