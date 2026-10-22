(() => {
  const storageKey = "kernel-rlm-theme";
  const root = document.documentElement;
  const preference = window.matchMedia("(prefers-color-scheme: dark)");
  let savedTheme;
  try { savedTheme = window.localStorage.getItem(storageKey); } catch {}
  let manual = savedTheme === "dark" || savedTheme === "light";
  let toggle;

  const apply = (theme) => {
    root.dataset.theme = theme;
    toggle?.setAttribute("aria-pressed", String(theme === "dark"));
  };
  // Run before the stylesheet so the first paint uses the chosen palette.
  apply(manual ? savedTheme : preference.matches ? "dark" : "light");
  preference.addEventListener("change", (event) => {
    if (!manual) apply(event.matches ? "dark" : "light");
  });
  document.addEventListener("DOMContentLoaded", () => {
    toggle = document.querySelector("#theme-toggle");
    apply(root.dataset.theme);
    toggle.addEventListener("click", () => {
      const theme = root.dataset.theme === "dark" ? "light" : "dark";
      manual = true;
      apply(theme);
      // Storage can be disabled in private or restricted browser contexts.
      try { window.localStorage.setItem(storageKey, theme); } catch {}
    });
  });
})();
