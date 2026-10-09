document.querySelectorAll('[data-tab]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-tab]').forEach(tab => tab.setAttribute('aria-selected', String(tab === button)));
    document.querySelectorAll('.panel').forEach(panel => { panel.hidden = panel.id !== button.dataset.tab; });
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
document.querySelector('#run-search')?.addEventListener('input', event => {
  const query = event.target.value.toLowerCase();
  const runs = [...document.querySelectorAll('[data-run]')];
  runs.forEach(run => { run.hidden = !run.textContent.toLowerCase().includes(query); });
  document.querySelector('#run-count').textContent = `${runs.filter(run => !run.hidden).length} of ${runs.length} runs shown`;
});
