(() => {
  const root = document.documentElement;
  const body = document.body;
  const collection = document.querySelector("#collection");
  const minWidth = 220;
  const defaultWidth = 260;
  const storageKey = "kernel-rlm-collection-width";
  let handle = null;
  let stopDrag = null;

  const maximumWidth = () => Math.max(minWidth, window.innerWidth - 210 - 420);
  const setWidth = (width) => {
    const next = Math.min(maximumWidth(), Math.max(minWidth, width));
    root.style.setProperty("--collection-width", `${next}px`);
    try { window.localStorage.setItem(storageKey, String(next)); } catch {}
  };
  let initialWidth = defaultWidth;
  try {
    const savedWidth = Number(window.localStorage.getItem(storageKey));
    if (Number.isFinite(savedWidth) && savedWidth >= minWidth) initialWidth = savedWidth;
  } catch {}
  setWidth(initialWidth);

  const finishDrag = () => {
    if (!stopDrag) return;
    stopDrag();
    stopDrag = null;
    handle?.classList.remove("is-dragging");
  };
  const removeHandle = () => {
    finishDrag();
    handle?.remove();
    handle = null;
  };
  const syncHandle = () => {
    const enabled = body.classList.contains("has-collection") && window.innerWidth > 700;
    if (!enabled) { removeHandle(); return; }
    if (handle || !collection) return;
    handle = document.createElement("div");
    handle.className = "collection-resize-handle";
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-label", "Resize workspace items pane");
    handle.setAttribute("aria-orientation", "vertical");
    handle.tabIndex = 0;
    collection.after(handle);
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      const startX = event.clientX;
      const startWidth = collection.getBoundingClientRect().width;
      handle.classList.add("is-dragging");
      const move = (moveEvent) => setWidth(startWidth + moveEvent.clientX - startX);
      const end = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", end);
        window.removeEventListener("pointercancel", end);
        stopDrag = null;
        handle?.classList.remove("is-dragging");
      };
      stopDrag = end;
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", end, { once: true });
      window.addEventListener("pointercancel", end, { once: true });
    });
    handle.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      setWidth(collection.getBoundingClientRect().width + (event.key === "ArrowRight" ? 16 : -16));
    });
  };

  new MutationObserver(syncHandle).observe(body, { attributes: true, attributeFilter: ["class"] });
  window.addEventListener("resize", () => {
    setWidth(parseFloat(root.style.getPropertyValue("--collection-width")) || defaultWidth);
    syncHandle();
  });
  syncHandle();
})();
