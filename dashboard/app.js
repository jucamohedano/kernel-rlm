const content = document.querySelector("#content");
const collection = document.querySelector("#collection");
const heading = document.querySelector("#heading");
let renderId = 0;
let pendingRequest;
const text = value => value == null ? "Unavailable" : typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
const statusLabel = value => value ? String(value).replaceAll("_", " ") : "Not recorded";
const metric = value => typeof value === "number" ? value.toFixed(3) : text(value);
const yesNo = value => typeof value === "boolean" ? (value ? "Yes" : "No") : "Unavailable";

function el(tag, cls, value) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (value !== undefined) node.textContent = value;
  return node;
}
function link(label, href, cls = "") {
  const node = el("a", cls, label);
  node.href = href;
  return node;
}
function route(run, trajectory, tab = "trajectory") {
  return "#trajectories/" + encodeURIComponent(run) + (trajectory == null ? "" : "/" + encodeURIComponent(trajectory) + "/" + tab);
}
async function api(path, signal) {
  const response = await fetch("/api/" + path, {signal});
  if (!response.ok) throw new Error(response.status === 404
    ? "This record is not available in the selected local evidence collection."
    : `The evidence request failed (${response.status}).`);
  return response.json();
}
function panel(title, value) {
  const node = el("section", "panel");
  node.append(el("h2", "panel-title", title));
  if (Array.isArray(value)) {
    const list = el("ul", "note-list");
    for (const item of value) { const li = el("li"); li.append(markdown(text(item))); list.append(li); }
    node.append(list);
  } else if (value != null) node.append(markdown(text(value)));
  return node;
}
function categoryTag(category) {
  const tag = el("span", "category-tag", category || "Unclassified");
  tag.dataset.category = category || "Unclassified";
  return tag;
}
function experimentTags(parent, experiments = []) {
  for (const ex of experiments) {
    const a = link("", "#experiments/" + encodeURIComponent(ex.id), "experiment-tag");
    a.append(categoryTag(ex.category), el("span", "", ex.title || ex.id));
    parent.append(a);
  }
}
function badge(value) {
  const state = String(value || "");
  const style = /closed|concluded|validated|finished|success/.test(state) ? "success" : /fail|error/.test(state) ? "failure" : /paused|defer/.test(state) ? "paused" : "neutral";
  return el("span", "status-pill " + style, statusLabel(value));
}
function raw(parent, value, title = "Raw record") {
  const details = el("details", "raw-record");
  details.append(el("summary", "", title), el("pre", "", text(value)));
  parent.append(details);
}
function codeBlock(value, language = "") {
  const block = el("div", "code-frame");
  const bar = el("div", "code-label", language || "Code");
  const pre = el("pre", "numbered-code");
  const code = el("code");
  String(value).split("\n").forEach(line => code.append(el("span", "code-line", line || " ")));
  pre.append(code);
  block.append(bar, pre);
  return block;
}
function inline(parent, source) {
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]*\))/g;
  let end = 0;
  for (const match of source.matchAll(pattern)) {
    parent.append(document.createTextNode(source.slice(end, match.index)));
    const token = match[0];
    if (token.startsWith("`")) parent.append(el("code", "", token.slice(1, -1)));
    else if (token.startsWith("**")) parent.append(el("strong", "", token.slice(2, -2)));
    else {
      const [, label, target] = /^\[([^\]]+)\]\(([^)]*)\)$/.exec(token);
      const docs = {"README.md":"README", "first-campaign.md":"first-campaign", "protocol.md":"protocol", "literature-review.md":"literature-review", "artifact-publication.md":"artifact-publication", "experiment-tracking.md":"experiment-tracking"};
      const anchor = el("a", "", label);
      if (target === "literature-review.md") anchor.href = "#literature";
      else if (docs[target]) anchor.href = "#writing/" + docs[target];
      else if (target.endsWith("/dashboard/guide.html")) anchor.href = "/archive/guide";
      else if (/^https?:\/\//i.test(target)) { anchor.href = target; anchor.rel = "noreferrer"; }
      else anchor.append(document.createTextNode(" (" + target + ")"));
      parent.append(anchor);
    }
    end = match.index + token.length;
  }
  parent.append(document.createTextNode(source.slice(end)));
}
const literatureImages = new Set([
  "/literature/kernelbench-figure-1.png",
  "/literature/lit-13-schematic.svg",
  "/literature/lit-14-schematic.svg",
  "/literature/lit-15-schematic.svg",
  "/literature/lit-16-figure-1.svg",
  "/literature/lit-17-schematic.svg",
  "/literature/lit-18-schematic.svg",
  "/literature/lit-19-figure-4.svg",
  "/literature/lit-2-schematic.svg",
  "/literature/lit-20-schematic.svg",
  "/literature/lit-21-schematic.svg",
  "/literature/lit-22-figure-3.svg",
  "/literature/lit-23-figure-3-left.png",
  "/literature/lit-24-schematic.svg",
  "/literature/lit-25-schematic.svg",
  "/literature/lit-2507.11948-figure-1.png",
  "/literature/lit-2509.14279-figure-2.png",
  "/literature/lit-26-schematic.svg",
  "/literature/lit-2602.05885-schematic.svg",
  "/literature/lit-2602.24286-schematic.svg",
  "/literature/lit-2603.21465-schematic.svg",
  "/literature/lit-2606.20128-figure-2.png",
  "/literature/lit-2607.16241-figure-1.svg",
  "/literature/lit-2608.03391-figure-1.png",
  "/literature/lit-27-schematic.svg",
  "/literature/lit-28-figure-2.svg",
  "/literature/lit-29-figure-1.png",
  "/literature/lit-3-figure-1.png",
  "/literature/rao-credit-ablation.svg",
  "/literature/rao-delegation-ablation.svg",
  "/literature/rao-reward.svg",
  "/literature/research-map.svg",
  "/literature/rlm-figure-2.png",
  "/literature/train-evaluate.svg",
]);
function markdown(source) {
  const article = el("div", "markdown");
  let code = null, language = "", list = null, paragraph = [];
  const flush = () => {
    if (paragraph.length) { const p = el("p"); inline(p, paragraph.join(" ")); article.append(p); paragraph = []; }
  };
  for (const line of String(source || "").split(/\r?\n/)) {
    if (code !== null) {
      if (/^\s*```/.test(line)) { article.append(codeBlock(code.join("\n"), language)); code = null; }
      else code.push(line);
      continue;
    }
    const fence = /^\s*```(.*)$/.exec(line);
    if (fence) { flush(); list = null; code = []; language = fence[1].trim(); continue; }
    if (!line.trim()) { flush(); list = null; continue; }
    const image = /^!\[([^\]]*)\]\(([^)]*)\)\s*$/.exec(line);
    if (image) {
      flush(); list = null;
      const [, alt, path] = image;
      if (literatureImages.has(path)) {
        const figure = el("figure", "literature-figure");
        const img = el("img"); img.src = path; img.alt = alt; img.loading = "lazy";
        const fullSize = link("", path, "literature-image-link");
        fullSize.target = "_blank"; fullSize.rel = "noopener";
        fullSize.setAttribute("aria-label", "Open full-size figure: " + alt + " (new tab)"); fullSize.append(img);
        figure.append(fullSize, el("figcaption", "", alt)); article.append(figure);
      } else article.append(el("p", "", line));
      continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    const item = /^\s*(?:[-*]|\d+\.)\s+(.*)$/.exec(line);
    if (h) { flush(); list = null; const node = el("h" + Math.min(h[1].length + 1, 5)); inline(node, h[2]); article.append(node); }
    else if (item) { flush(); if (!list) { list = el("ul"); article.append(list); } const node = el("li"); inline(node, item[1]); list.append(node); }
    else { list = null; paragraph.push(line); }
  }
  flush();
  if (code !== null) article.append(codeBlock(code.join("\n"), language));
  return article;
}
function literatureMarkdown(source, topics, onNavigate = () => {}) {
  const article = el("article", "literature-article");
  topics.replaceChildren();
  let parent = article, lines = [], fenced = false, topicIndex = 0;
  const flush = () => {
    if (lines.length) { parent.append(markdown(lines.join("\n"))); lines = []; }
  };
  for (const line of String(source || "").replace(/^#\s+[^\n]+\n?/, "").split(/\r?\n/)) {
    if (/^\s*```/.test(line)) fenced = !fenced;
    const heading = !fenced && /^(#{2,3})\s+(.*)$/.exec(line);
    if (!heading) { lines.push(line); continue; }
    flush();
    if (heading[1] === "##") {
      const section = el("section", "literature-topic");
      const title = el("h2"); inline(title, heading[2]);
      title.id = "literature-topic-" + topicIndex++; title.tabIndex = -1;
      section.append(title); article.append(section); parent = section;
      const button = el("button", "literature-topic-link", heading[2]); button.type = "button";
      button.addEventListener("click", () => { const target = onNavigate(title) || title; target.scrollIntoView({block: "start"}); target.focus({preventScroll: true}); });
      topics.append(button);
    } else {
      const details = el("details", "literature-paper");
      const summary = el("summary"); inline(summary, heading[2]);
      details.append(summary);
      const section = article.lastElementChild;
      if (section?.className === "literature-topic") section.append(details);
      else article.append(details);
      parent = details;
    }
  }
  flush();
  return article;
}
function notice(message) { content.append(el("div", "warning", message)); }
function tokenPanel(title, usage) {
  const node = el("section", "panel token-panel");
  node.append(el("h2", "panel-title", title));
  const grid = el("div", "token-grid");
  for (const [key, label] of [["input_tokens", "Input"], ["output_tokens", "Output"], ["total_tokens", "Total"]]) {
    const metric = usage?.[key];
    const value = metric?.sum == null ? "Unknown" : metric.sum.toLocaleString();
    const coverage = metric ? metric.total == null ? `${metric.recorded} calls recorded · coverage unknown` : `${metric.recorded}/${metric.total} calls recorded${metric.recorded < metric.total ? " · partial sum" : " · complete"}` : "Not recorded";
    const item = el("div", "token-stat"); item.append(el("span", "muted", label), el("strong", "", value), el("small", "muted", coverage)); grid.append(item);
  }
  node.append(grid);
  const elapsed = usage?.elapsed_seconds;
  const elapsedText = elapsed?.sum == null ? "Elapsed sum unavailable" : `${elapsed.sum.toFixed(3)} s summed across ${elapsed.recorded}/${elapsed.total ?? "?"} calls (not wall time)`;
  node.append(el("small", "muted", elapsedText));
  return node;
}
function aggregateRunUsage(traces) {
  const aggregate = {root: {}, children: {}, all: {}};
  for (const group of ["root", "children", "all"]) for (const key of ["input_tokens", "output_tokens", "total_tokens", "elapsed_seconds"]) {
    const entries = traces.map(t => t.metrics?.token_usage?.[group]?.[key]).filter(Boolean);
    const known = entries.filter(entry => entry.sum != null);
    aggregate[group][key] = {sum: known.length ? known.reduce((sum, entry) => sum + entry.sum, 0) : null, recorded: entries.reduce((sum, entry) => sum + entry.recorded, 0), total: entries.every(entry => entry.total != null) ? entries.reduce((sum, entry) => sum + entry.total, 0) : null};
  }
  return aggregate;
}
function rows(parent, records, target) {
  for (const record of records) {
    const row = link("", target(record), "run-list-row");
    row.append(el("strong", "run-name", record.title || record.id),
      el("span", "run-count", record.detail || (record.trajectory_count == null ? "No episode export" : `${record.trajectory_count} trajectories`)), badge(["running", "paused"].includes(record.status) ? "saved: " + record.status : record.status));
    parent.append(row);
  }
}
function experimentNav(experiments, selected, {view = "experiments", onSelect} = {}) {
  document.body.classList.add("has-collection");
  collection.hidden = false;
  collection.replaceChildren(el("div", "collection-heading", "L1 simple kernels · Stage 1"));
  collection.append(el("p", "collection-context", "Training, fixes and supporting work"));
  const search = el("input", "collection-search");
  search.type = "search"; search.placeholder = "Search records…"; search.setAttribute("aria-label", "Search stage records");
  const filter = el("select", "collection-filter");
  filter.setAttribute("aria-label", "Filter record category");
  const categories = [...new Set(experiments.map(ex => ex.category).filter(Boolean))];
  for (const category of ["All categories", ...categories]) {
    const option = el("option", "", category); option.value = category === "All categories" ? "" : category; filter.append(option);
  }
  const list = el("div", "experiment-list");
  const update = () => {
    list.replaceChildren();
    const matches = experiments.filter(ex => (!filter.value || ex.category === filter.value) &&
      [ex.title, ex.id, ex.category, ex.campaign, ex.scope_note, ex.intent, ex.conclusion].filter(Boolean).join(" ").toLowerCase().includes(search.value.toLowerCase()));
    for (const ex of matches) {
      const a = link("", "#" + view + "/" + encodeURIComponent(ex.id), "experiment-row" + (selected === ex.id ? " selected" : ""));
      a.dataset.experimentId = ex.id;
      if (selected === ex.id) a.setAttribute("aria-current", "page");
      if (onSelect) a.addEventListener("click", event => {
        if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();onSelect(ex.id);
      });
      a.append(categoryTag(ex.category), el("strong", "", ex.title), el("small", "", `${ex.runs.length} workloads`));
      list.append(a);
    }
    if (!matches.length) list.append(el("p", "muted", "No matching records."));
  };
  search.addEventListener("input", update); filter.addEventListener("change", update);
  collection.append(search, filter, list); update();
  return id => {
    selected = id;
    const selectedRecord = experiments.find(ex => ex.id === id);
    if (selectedRecord && ((filter.value && selectedRecord.category !== filter.value) ||
        ![selectedRecord.title, selectedRecord.id, selectedRecord.category, selectedRecord.campaign, selectedRecord.scope_note, selectedRecord.intent, selectedRecord.conclusion].filter(Boolean).join(" ").toLowerCase().includes(search.value.toLowerCase()))) {
      search.value = "";filter.value = "";
    }
    update();
    const row = [...list.children].find(node => node.dataset.experimentId === id);
    if (row) {
      const bounds = row.getBoundingClientRect(), pane = collection.getBoundingClientRect();
      if (bounds.top < pane.top || bounds.bottom > pane.bottom) collection.scrollTop += bounds.top - pane.top - pane.height / 2 + bounds.height / 2;
    }
  };
}
async function experimentsPage(id, signal, ticket) {
  const data = await api("experiments", signal);
  if (ticket !== renderId) return;
  const items = data.experiments || [];
  const selected = items.find(ex => ex.id === id) || (!id ? items[0] : null);
  experimentNav(items, selected?.id);
  if (!selected) { notice("Experiment not found."); return; }
  heading.textContent = selected.title;
  const actions = el("div", "detail-actions");
  actions.append(categoryTag(selected.category), el("span", "muted", `Stage 1 · saved status: ${statusLabel(selected.status)}`));
  content.append(actions);
  const context = el("div", "campaign-context");
  context.append(el("p", "", "The project began with Modal trials and moved to CloudRift after the compute grant. These records cover training, setup checks, applied fixes and pilot branches—not independent proof of improvement."), link("Read the stage history →", "#writing/first-campaign"));
  content.append(context);
  if (selected.scope_note) content.append(el("p", "scope-note", selected.scope_note));
  content.append(panel("Intent", selected.intent), panel("Conclusion", selected.conclusion));
  const details = el("details", "experiment-notes"); details.append(el("summary", "", "Change, limitations and recorded decision"));
  details.append(panel("Change", selected.change), panel("Limitations", selected.limitations), panel("Recorded decision — not a recommendation", selected.decision)); content.append(details);
  content.append(link("View lineage →", "#lineage/" + encodeURIComponent(selected.id), "back"));
  content.append(el("h2", "section-title", "Workloads"));
  content.append(el("p", "muted", "A workload is a saved execution run. Each can contain several tasks and multiple trajectory attempts per task."));
  const linked = el("div", "linked-run-list"); content.append(linked);
  const runData = await api("runs", signal);
  if (ticket !== renderId) return;
  const byId = new Map(runData.runs.map(run => [run.id, run]));
  for (const id of selected.runs) {
    const run = byId.get(id) || {id, status:"evidence unavailable"};
    const card = el("article", "workload-card");
    card.append(el("h3", "", run.title || run.id), el("p", "muted", run.id), badge("saved: " + statusLabel(run.status)));
    const count = run.trajectory_count == null ? "No local trajectory export" : `${run.trajectory_count} trajectory attempts`;
    card.append(el("p", "", count), link("View workload trajectories →", route(run.id), "back"));
    linked.append(card);
  }
  if (!selected.runs.length) linked.append(el("p", "muted", "No execution was recorded for this preparation."));
}
function modelLabel(t) {
  return t.model ? `${t.model}${t.model_scope === "base" ? " · base model" : ""}` : "Model unavailable";
}
function trajectoryCard(t, fallbackRun) {
  const runId = t.run?.id || fallbackRun;
  const card = el("article", "trajectory-card");
  const title = t.title || t.task || t.id;
  card.append(link(title, route(runId, t.id), "trajectory-title"));
  const meta = el("p", "trajectory-meta", `${t.task || "Task unavailable"} · Attempt ${t.attempt ?? "—"}`);
  card.append(meta);raw(card,t.id,"Episode ID");
  const tags = el("div", "detail-actions"); experimentTags(tags, t.experiments); card.append(tags);
  card.append(el("p", "muted", `${t.benchmark || "Benchmark unavailable"} · ${modelLabel(t)}`));
  const outcomes = el("div", "detail-actions");
  outcomes.append(badge(t.completion), badge(t.outcome.correct === true ? "Correct" : t.outcome.correct === false ? "Incorrect" : "Verification unavailable"), el("span", "muted", `Reward ${metric(t.outcome.reward)}`));
  card.append(outcomes, el("p", "trajectory-meta", `Workload: ${runId}`));
  return card;
}
function trajectoryList(records, parent, fallbackRun) {
  const controls = el("div", "writing-toolbar");
  const search = el("input", "search"); search.type="search"; search.placeholder="Search task, experiment, model or trajectory ID…"; search.setAttribute("aria-label","Search trajectories");
  const filter = el("select", "collection-filter"); filter.setAttribute("aria-label", "Filter trajectory completion");
  for (const name of ["All attempts", ...new Set(records.map(t=>t.completion).filter(Boolean))]) { const option=el("option","",name);option.value=name==="All attempts"?"":name;filter.append(option); }
  const list = el("div", "trajectory-list"); const count = el("p", "muted");
  let limit=40;
  const more=el("button","","Show 40 more");
  const update=()=>{
    const matches=records.filter(t=>(!filter.value||t.completion===filter.value)&&[t.title,t.task,t.id,t.model,t.run?.id,...(t.experiments||[]).map(ex=>ex.title)].join(" ").toLowerCase().includes(search.value.toLowerCase()));
    list.replaceChildren(...matches.slice(0,limit).map(t=>trajectoryCard(t,fallbackRun)));
    count.textContent=`${matches.length} matching attempts · showing ${Math.min(limit,matches.length)}`;more.hidden=matches.length<=limit;
  };
  search.addEventListener("input",()=>{limit=40;update();});filter.addEventListener("change",()=>{limit=40;update();});more.addEventListener("click",()=>{limit+=40;update();});
  controls.append(search,filter);parent.append(controls,count,list,more);update();
}
async function trajectoriesPage(signal,ticket) {
  content.append(el("p","muted","Individual trajectory attempts across saved workloads. A task can have several attempts. Reused episodes remain linked to each scoring workload; they are not new samples."));
  const loading=el("p","loading","Reading local trajectory summaries…");content.append(loading);
  const data=await api("trajectories",signal);if(ticket!==renderId)return;loading.remove();
  if(data.warnings?.length) {
    const notes=el("details","trajectory-extra");notes.append(el("summary","",`${data.warnings.length} workload records without episode exports`));
    for(const warning of data.warnings)notes.append(el("p","muted",warning));content.append(notes);
  }
  trajectoryList(data.trajectories,content);
}
async function lineagePage(focus,signal,ticket) {
  const data=await api("experiments",signal);if(ticket!==renderId)return;
  heading.textContent="L1 simple kernels · Stage 1 lineage";
  content.append(el("p","scope-note","Recorded research follow-ups—not checkpoint inheritance. The Modal and native TRL trainer lineages are separate; pilot branches are not cumulative training progress."));
  const byId=new Map(data.experiments.map(ex=>[ex.id,ex]));
  if (!byId.has(focus)) focus = data.experiments[0]?.id;
  const syncList = experimentNav(data.experiments, focus, {view:"lineage", onSelect: id => selectExperiment(id)});
  const levels=new Map();
  function level(id,visited=new Set()) {
    if(levels.has(id))return levels.get(id);
    if(visited.has(id))return 0;
    const next=new Set(visited);next.add(id);
    const parents=(byId.get(id)?.follows_experiments||[]).filter(p=>byId.has(p));
    const depth=parents.length?1+Math.max(...parents.map(p=>level(p,next))):0;levels.set(id,depth);return depth;
  }
  const slots=new Map(),positions=new Map();
  for(const ex of [...data.experiments].reverse()) {
    const depth=level(ex.id),slot=slots.get(depth)||0;slots.set(depth,slot+1);positions.set(ex.id,{x:30+depth*330,y:30+slot*205});
  }
  const width=60+(Math.max(...levels.values())+1)*330,height=60+Math.max(...slots.values())*205;
  const toolbar=el("div","writing-toolbar"),zoomOut=el("button","","−"),zoomIn=el("button","","+"),reset=el("button","","100%"),label=el("span","muted",`${data.experiments.length} records · drag background or scroll to pan`);
  zoomOut.setAttribute("aria-label","Zoom out lineage");zoomIn.setAttribute("aria-label","Zoom in lineage");
  toolbar.append(zoomOut,zoomIn,reset,label);
  const viewport=el("div","lineage-viewport"),extent=el("div","lineage-extent"),graph=el("div","lineage-graph");graph.style.width=width+"px";graph.style.height=height+"px";
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("width",width);svg.setAttribute("height",height);svg.classList.add("lineage-edges");
  for(const ex of data.experiments)for(const parent of ex.follows_experiments||[]) {
    if(!positions.has(parent))continue;const a=positions.get(parent),b=positions.get(ex.id);
    const path=document.createElementNS(svg.namespaceURI,"path");path.setAttribute("d",`M${a.x+280},${a.y+75} C${a.x+310},${a.y+75} ${b.x-30},${b.y+75} ${b.x},${b.y+75}`);svg.append(path);
  }
  graph.append(svg);
  const cards = new Map();
  for(const ex of data.experiments) {
    const pos=positions.get(ex.id),card=el("article","lineage-node"+(focus===ex.id?" selected":""));card.style.left=pos.x+"px";card.style.top=pos.y+"px";
    const select = el("button", "lineage-select", ex.title);
    select.setAttribute("aria-pressed", String(focus === ex.id));
    select.addEventListener("click", () => selectExperiment(ex.id));
    card.append(categoryTag(ex.category),select,el("small","muted",`${ex.runs.length} workloads · saved: ${statusLabel(ex.status)}`),link("Open experiment →","#experiments/"+encodeURIComponent(ex.id),"lineage-open"));
    card.addEventListener("click", event => {if (!event.target.closest("a,button")) selectExperiment(ex.id);});
    card.dataset.experimentId=ex.id;card.title=ex.scope_note||ex.title;cards.set(ex.id,card);graph.append(card);
  }
  let zoom=1;
  const scale=()=>{graph.style.transform=`scale(${zoom})`;extent.style.width=width*zoom+"px";extent.style.height=height*zoom+"px";reset.textContent=Math.round(zoom*100)+"%";};
  zoomOut.addEventListener("click",()=>{zoom=Math.max(.35,zoom-.15);scale();});zoomIn.addEventListener("click",()=>{zoom=Math.min(1.6,zoom+.15);scale();});reset.addEventListener("click",()=>{zoom=1;scale();});
  let drag=null;
  viewport.addEventListener("pointerdown",e=>{if(e.button!==0||e.target.closest("a,button,.lineage-node"))return;drag={x:e.clientX,y:e.clientY,left:viewport.scrollLeft,top:viewport.scrollTop};viewport.setPointerCapture(e.pointerId);});
  viewport.addEventListener("pointermove",e=>{if(drag){viewport.scrollLeft=drag.left+drag.x-e.clientX;viewport.scrollTop=drag.top+drag.y-e.clientY;}});
  viewport.addEventListener("pointerup",()=>{drag=null;});viewport.addEventListener("pointercancel",()=>{drag=null;});
  extent.append(graph);viewport.append(extent);content.append(toolbar,viewport);
  content.append(el("p","muted","Lines mean “follow-up”; read each record for exact checkpoint and replay provenance. Categories describe purpose, not success."));scale();
  function selectExperiment(id) {
    if (!positions.has(id)) return;
    focus=id;
    for (const [key,card] of cards) {
      card.classList.toggle("selected",key===id);
      card.querySelector(".lineage-select").setAttribute("aria-pressed",String(key===id));
    }
    syncList(id);
    const position=positions.get(id);
    viewport.scrollLeft=Math.max(0,(position.x+140)*zoom-viewport.clientWidth/2);
    viewport.scrollTop=Math.max(0,(position.y+75)*zoom-viewport.clientHeight/2);
    history.replaceState(null,"","#lineage/"+encodeURIComponent(id));
  }
  if(focus)selectExperiment(focus);
}
function eventTitle(event,index) {
  if(event.source==="kernel_subcalls")return "Child assignments and responses";
  if(event.source==="kernel_child_progress")return "Child progress";
  const role=event.role||"event";
  if(role==="system")return "System instructions";
  if(role==="user")return "User request / observation";
  if(role==="tool")return "REPL output";
  const names=(event.tool_calls||[]).map(call=>(call.function||call).name||"tool");
  return names.length?names.join(" · "):role==="assistant"?"Assistant response":`Event ${index+1}`;
}
function readableChildText(value) {
  if (typeof value === "string") return value;
  if (!value || typeof value !== "object") return null;
  const message = value.choices?.[0]?.message || value.message;
  if (message) return readableChildText(message.content);
  if (typeof value.content === "string") return value.content;
  if (Array.isArray(value)) return value.filter(part=>part?.type === "text").map(part=>part.text).join("\n") || null;
  return null;
}
function renderChildCalls(records,parent) {
  parent.append(el("p","muted","Recorded child calls. Parent links are shown only when captured; absent history remains unknown."));
  for (const [i,child] of records.entries()) {
    const section=el("section","child-summary");
    const choices=child.response?.choices || [];
    section.append(el("h3","",`Child session ${i+1} · ${child.mode || "one-shot"} · ${child.child_invocation_id || child.call_id || "invocation unknown"}`));
    section.append(el("p","muted",`Parent turn ${child.parent_turn_id || "unknown"} · action ${child.parent_action_id || child.parent_tool_call_id || "unknown"} · ${child.usage_accounting === "incomplete" ? `at least ${child.observed_completion_tokens ?? "unknown"} generated tokens · accounting incomplete` : child.completion_tokens == null ? "usage unknown" : `${child.completion_tokens} generated tokens`} / ${child.child_total_tokens ?? "unknown"} total cap · ${child.termination_reason || "termination unknown"}`));
    const summary=el("p","muted");
    summary.textContent=[child.status,choices[0]?.finish_reason,typeof child.elapsed_seconds==="number"?`${child.elapsed_seconds.toFixed(1)} seconds`:null].filter(Boolean).join(" · ");section.append(summary);
    const assignment=readableChildText(child.prompt||child.assignment||child.input) || (child.request?.messages||[]).filter(m=>m.role==="user").map(m=>readableChildText(m.content)).filter(Boolean).join("\n\n");
    const response=readableChildText(child.response||child.output);
    const assignmentPanel=panel("Assignment");
    assignmentPanel.append(el("div","child-assignment-text",assignment||"Assignment text not recorded"));
    section.append(assignmentPanel,panel("Response",response||"Response text not recorded"));
    if(child.error)section.append(panel("Error",readableChildText(child.error)||"See raw events for structured error details."));
    for (const [turnIndex,turn] of (child.turns||[]).entries()) {
      const turnSection=el("details","child-turn");
      turnSection.append(el("summary","",`Model turn ${turn.step || turnIndex+1} · ${turn.finish_reason || "unknown"} · ${turn.observed_completion_tokens ?? "usage unknown"} output tokens · ${turn.outcome || "outcome unknown"}`));
      if(turn.prompt_messages)turnSection.append(panel("Exact child request",JSON.stringify(turn.prompt_messages,null,2)));
      const raw=turn.raw_reply?.content ?? turn.visible_reply;
      if(raw != null)turnSection.append(panel("Raw child reply · generated markers are not executed calls",String(raw)));
      if(turn.format_error)turnSection.append(panel("Malformed action · not executed",turn.format_error));
      if(turn.parsed_action)turnSection.append(panel("Parsed child action · request, not execution",JSON.stringify(turn.parsed_action,null,2)));
      if(turn.observation)turnSection.append(panel(turn.outcome==="static_rejected"?"Static validation observation · compiler/checker not run":"Compiler/check observation",JSON.stringify(turn.observation,null,2)));
      section.append(turnSection);
    }
    if(child.checks?.length) section.append(panel("Compiler/check actions and observations",JSON.stringify(child.checks,null,2)));
    parent.append(section);
  }
}
function renderTurnTimeline(events,parent) {
  const groups=[];let current=null;
  for(const event of events) {
    if(event.source==="kernel_child_progress"||event.source==="kernel_subcalls"||event.source==="kernel_child_sessions"||event.source==="kernel_root_actions")continue;
    if(event.role==="assistant") {current={assistant:event,observations:[]};groups.push(current);}
    else if(current)current.observations.push(event);
    else {const context=el("details","timeline-context");context.append(el("summary","",event.role==="system"?"System instructions":"Initial request"));renderEvents([event],context,{expanded:true});parent.append(context);}
  }
  for(const [index,group] of groups.entries()) {
    const turn=el("section","turn-group");turn.append(el("h3","turn-heading",`Turn ${index+1}`));
    const body=el("div","turn-body");renderEvents([group.assistant],body,{expanded:true});turn.append(body);
    if(group.observations.length){const observations=el("details","turn-observations");observations.append(el("summary","",`Observations · ${group.observations.length}`));renderEvents(group.observations,observations,{expanded:true});turn.append(observations);}
    parent.append(turn);
  }
  const children=events.filter(event=>event.source==="kernel_subcalls"||event.source==="kernel_child_sessions"||event.source==="kernel_root_actions");
  if(children.length){const section=el("details","timeline-context");section.append(el("summary","","Recorded child requests and sessions"));for(const event of children){section.append(el("h3","",event.source==="kernel_child_sessions"?"Child invocation sessions":event.source==="kernel_root_actions"?"Root actions and observations":"Recorded child model requests"));if(event.source==="kernel_root_actions"){for(const [index,action] of (event.records||[]).entries()){const actionView=el("details","child-turn");actionView.append(el("summary","",`Root action ${action.action_id || index+1} · ${action.status} · ${action.tool_call_id || "call unknown"}`));actionView.append(panel("Executed source",action.code||""),panel("Observation",action.observation||""));section.append(actionView);}}else renderChildCalls(event.records||[],section);}parent.append(section);}
}

function renderEvents(events,parent,{expanded=false}={}) {
  for(const [position,event] of events.entries()) {
    const role=event.role||"child";
    const node=el("article","event trace-event role-"+role);
    const title=el("div","event-head");title.append(el("span","event-label",`${event.index == null ? "" : "#"+(event.index+1)+" · "}${eventTitle(event,position)}`));
    if(event.usage)title.append(el("span","muted",`in ${event.usage.prompt_tokens??"—"} · out ${event.usage.completion_tokens??"—"}`));node.append(title);
    if(event.source==="kernel_child_progress") {
      const latest=new Map();for(const record of event.records||[])latest.set(record.call_id,record);
      for(const record of latest.values())node.append(el("p","muted",`${statusLabel(record.stage)} · ${record.completion_tokens ?? "—"} generated tokens · ${typeof record.elapsed_seconds === "number" ? record.elapsed_seconds.toFixed(1)+" seconds" : "Duration unavailable"}`));
    } else if(Array.isArray(event.records))renderChildCalls(event.records,node);
    if(event.reasoning){const reason=el("details","reasoning");reason.append(el("summary","","Reasoning"),markdown(text(event.reasoning)));node.append(reason);}
    if(event.content) {
      const message=el("details","event-content");message.open=expanded||role==="assistant";
      message.append(el("summary","",role==="tool"?"Output":role==="system"?"System prompt":"Message"));
      message.append(role==="tool"?codeBlock(text(event.content),"REPL output"):markdown(text(event.content)));node.append(message);
    }
    for(const call of event.tool_calls||[]) {
      const fn=call.function||call;let args=fn.arguments;
      if(typeof args==="string"){try{args=JSON.parse(args);}catch{/* Display original malformed arguments. */}}
      const code=typeof args?.code==="string"?args.code:text(args);
      const tool=el("details","tool-call");tool.open=expanded;
      tool.append(el("summary","",`${fn.name||"Tool"} · ${code.split("\n")[0].slice(0,100)}`),codeBlock(code,typeof args?.code==="string"?"Python / REPL":"Arguments"));node.append(tool);
    }
    parent.append(node);
  }
}
function renderStepBrowser(events,parent) {
  const toolbar=el("div","trace-toolbar"),stepsButton=el("button","active","Steps"),timelineButton=el("button","","Timeline");
  toolbar.append(stepsButton,timelineButton);parent.append(toolbar);
  const body=el("div","step-browser");parent.append(body);let mode="steps",selected=0;
  const update=()=>{
    const matches=events.map((event,index)=>({event,index}));
    body.replaceChildren();body.className=mode==="steps"?"step-browser":"timeline-list";
    stepsButton.classList.toggle("active",mode==="steps");timelineButton.classList.toggle("active",mode==="timeline");
    if(!matches.length){body.append(el("p","muted","No matching steps."));return;}
    if(mode==="timeline"){renderTurnTimeline(events,body);return;}
    if(!matches.some(item=>item.index===selected))selected=matches[0].index;
    const list=el("div","step-index"),detail=el("div","step-detail");list.setAttribute("aria-label","Trajectory steps");
    for(const {event,index} of matches){const button=el("button","step-item"+(selected===index?" selected":""),`${index+1}  ${eventTitle(event,index)}`);button.setAttribute("aria-pressed",String(selected===index));button.addEventListener("click",()=>{const offset=list.scrollTop;selected=index;update();body.querySelector(".step-index").scrollTop=offset;});list.append(button);}
    renderEvents([events[selected]],detail,{expanded:true});body.append(list,detail);
  };
  stepsButton.addEventListener("click",()=>{mode="steps";update();});timelineButton.addEventListener("click",()=>{mode="timeline";update();});update();
}
function pair(parent, label, value) {
  const row = el("div", "metric-pair"); row.append(el("span", "", label), el("strong", "", text(value))); parent.append(row);
}
async function runPage(id, traceId, tab, signal, ticket) {
  const query = traceId == null ? "" : "?trajectory=" + encodeURIComponent(traceId);
  const data = await api("runs/" + encodeURIComponent(id) + query, signal);
  if (ticket !== renderId) return;
  const run = data.run, traces = data.trajectories || [];
  heading.textContent = run.title || id;
  content.append(link("← All trajectories", "#trajectories", "back"));
  (data.warnings || []).forEach(notice);
  if (!traces.length) { notice("No trajectory export is available for this record. The historical record has not been deleted."); return; }
  if (!traceId) {
    content.append(el("p", "muted", `${traces.length} saved trajectory attempts in this workload. Repeated task names are separate sampled attempts, not experiment names.`));
    const aggregate = aggregateRunUsage(traces);
    content.append(tokenPanel("Recorded run token usage · root calls", aggregate.root), tokenPanel("Recorded child-call token usage", aggregate.children), tokenPanel("Combined recorded calls", aggregate.all));
    if (run.elapsed_seconds != null) content.append(panel("Run elapsed time", `${run.elapsed_seconds} seconds${run.elapsed_scope ? ` · ${run.elapsed_scope}` : " · scope not recorded"}. This is run elapsed time, not a sum of per-call durations.`));
    trajectoryList(traces,content,id);
    return;
  }
  const t = traces.find(t => t.id === traceId);
  if (!t) { notice("Trajectory not found in this export."); return; }
  const select = el("select", "trajectory-picker"); select.setAttribute("aria-label", "Select trajectory");
  for (const [i, trace] of traces.entries()) { const option=el("option","",`${trace.title || trace.task || trace.id} · Attempt ${trace.attempt ?? i+1} · ${trace.completion || "Unknown"} · ${trace.id}`); option.value=trace.id; option.selected=trace.id===traceId; select.append(option); }
  select.addEventListener("change",()=>{location.hash=route(id,select.value);}); content.append(select);
  heading.textContent = t.title || t.task || t.id;
  const tags = el("div", "detail-actions"); experimentTags(tags,t.experiments); tags.append(badge(t.completion)); content.append(tags);
  const metadata=el("dl","trajectory-header-meta");
  const metadataRow=(label,value)=>{const row=el("div","metadata-row");row.append(el("dt","",label),el("dd","",text(value)));metadata.append(row);return row;};
  metadataRow("Benchmark",t.benchmark);
  const modelRow=metadataRow(t.model_scope === "base" ? "Base model" : "Model",t.model);
  if (t.model_source) {
    const source=el("details","muted");source.append(el("summary","",t.model_scope === "base" ? "Base model evidence" : "Model source"),el("p","",t.model_source));
    if (t.model_scope === "base" && t.model_evidence) source.append(el("p","",`${t.model_evidence.path}#${t.model_evidence.model_field}`),el("p","",`Base revision: ${t.model_revision}. Base only; not sampled policy checkpoint identity.`));
    modelRow.lastElementChild.append(source);
  }
  const runRow=el("div","metadata-row"),runStats=el("dd","trajectory-run-stats");
  runStats.append(el("span","",`${t.metrics?.steps ?? "—"} steps`),el("span","",`${t.metrics?.tool_calls ?? "—"} tool calls`),el("span","muted",`Started ${t.started_at || "not recorded"}`));
  runRow.append(el("dt","","Run"),runStats);metadata.append(runRow);content.append(metadata);
  const metrics=el("div","trajectory-metrics");
  for(const [label,value] of [["Peak input context",t.metrics?.peak_context == null ? "Unavailable" : `${t.metrics.peak_context.toLocaleString()} tokens`],["Prompt cache",typeof t.metrics?.prompt_cache === "number" ? t.metrics.prompt_cache.toFixed(1)+"%" : "Unavailable"],["Commands failed",t.metrics?.commands_failed]]) {
    const stat=el("div");stat.append(el("span","muted",label),el("strong","",text(value)));metrics.append(stat);
  }
  content.append(metrics);
  const tokenUsage = t.metrics?.token_usage;
  if (tokenUsage) {
    content.append(tokenPanel("Root-call usage", tokenUsage.root));
    content.append(tokenPanel("Child-call usage", tokenUsage.children));
    content.append(tokenPanel("Combined usage", tokenUsage.all));
  }
  if (t.metrics?.run_elapsed_seconds != null) content.append(panel("Run elapsed time", `${t.metrics.run_elapsed_seconds} seconds${t.metrics.run_elapsed_scope ? ` · ${t.metrics.run_elapsed_scope}` : " · scope not recorded"}. Per-call durations are not wall time.`));
  const extra=el("details","trajectory-extra");extra.append(el("summary","","Details"));
  pair(extra,"Task / attempt",`${t.task || "Unavailable"} / ${t.attempt ?? "—"}`);pair(extra,"Trajectory ID",t.id);pair(extra,"Workload",run.title || run.id);pair(extra,"Hardware",run.hardware);
  extra.append(el("p","muted","Steps are root assistant turns; tool calls are emitted calls. Peak input context is the largest recorded root prompt, not the configured context limit. Missing measurements are not zero."));content.append(extra);
  const io = el("div", "two");
  const asked = panel("Asked", `${t.title || t.task} · ${t.task || ""}`);
  if(t.task_source) raw(asked,t.task_source,"Reference source · selected task catalog / export");
  const firstUser=(t.events||[]).find(event=>event.role==="user");
  if(firstUser?.content)raw(asked,firstUser.content,"Full initial request");
  const returned=panel("Returned",t.outcome.submitted === true ? "Final candidate submitted. See independent verification below." : "No final candidate submission was recorded.");
  if(t.raw?.info?.kernel_submission)raw(returned,t.raw.info.kernel_submission,"Submitted source");
  io.append(asked,returned);content.append(io);
  const outcome=el("div","trajectory-metrics");
  for(const [label,value] of [["Submitted",yesNo(t.outcome.submitted)],["Correct",yesNo(t.outcome.correct)],["Reward",metric(t.outcome.reward)],["Speedup",typeof t.outcome.speedup === "number" ? metric(t.outcome.speedup)+"×" : "Unavailable"]]) {
    const stat=el("div");stat.append(el("span","muted",label),el("strong","",value));outcome.append(stat);
  }
  content.append(outcome);
  const viewer=el("section","trace-viewer"),tabs=el("nav","tabs");
  for(const [name,label] of [["trajectory","Trajectory"],["verifier","Verifier"],["events","Raw events"]]) {
    tabs.append(link(label,route(id,traceId,name),"tab"+((tab||"trajectory")===name?" active":"")));
  }
  viewer.append(tabs);content.append(viewer);
  if(tab==="verifier") {
    const v=t.verification||{}, summary=el("div","verifier-summary");
    pair(summary,"Protocol",v.verification_protocol);pair(summary,"Status",statusLabel(v.verification_status));pair(summary,"Compiled",yesNo(v.compiled));pair(summary,"Correct",yesNo(v.correct));pair(summary,"Reference (ms)",metric(v.ref_ms));pair(summary,"Candidate (ms)",metric(v.kernel_ms));viewer.append(summary);
    const timings = v.timings && typeof v.timings === "object" ? v.timings : null;
    if (timings) { const timingPanel = panel("Final verifier timing breakdown", Object.entries(timings).map(([name,value]) => `${name}: ${typeof value === "number" ? value.toFixed(3) + " s" : text(value)}`)); viewer.append(timingPanel); }
    else viewer.append(panel("Final verifier timing breakdown", "Unavailable; no final timing breakdown was recorded."));
    const output=v.stdout || v.output || v.error;
    if(output) {const block=codeBlock(text(output),"Verifier output");block.classList.add("verifier-output");viewer.append(block);}
    else viewer.append(el("p","muted","No text verifier output was recorded. Structured final results are available below."));
    raw(viewer,v,"Structured final verifier record");
  } else if(tab==="events") (t.events||[]).forEach(e=>raw(viewer,e));
  else renderStepBrowser(t.events||[],viewer);
  raw(content,t.raw,"Raw trajectory record");
}
let writingDraft = null;
function hasUnsavedWriting() { return writingDraft && writingDraft.editor.value !== writingDraft.original; }
window.addEventListener("beforeunload",event=>{if(hasUnsavedWriting()){event.preventDefault();event.returnValue="";}});
async function writingPage(id,signal,ticket,workspace="Writing") {
  const literature = workspace === "Literature";
  const index=literature ? null : await api("writing",signal);if(ticket!==renderId)return;
  document.body.classList.add("has-collection");collection.hidden=false;
  document.body.classList.toggle("literature-view", literature);
  collection.replaceChildren(el("div","collection-heading",literature ? "On this page" : workspace));
  const selected=id||index?.documents[0]?.id;if(!selected)return;
  const topics=el("nav","literature-topics");topics.setAttribute("aria-label","Literature topics");
  if (literature) collection.append(topics);
  else {
    const tree=el("div","file-tree");tree.append(el("div","muted","docs / research /"));
    for(const doc of index.documents){const a=link(doc.path?.split("/").pop()||doc.title,"#writing/"+encodeURIComponent(doc.id),"writing-link"+(selected===doc.id?" selected":""));tree.append(a);}
    collection.append(tree);
  }
  const doc=await api("writing/"+encodeURIComponent(selected),signal);if(ticket!==renderId)return;
  heading.textContent=doc.markdown.match(/^#\s+(.+)$/m)?.[1] || doc.title;
  content.append(el("p","writing-path",doc.path));
  const toolbar=el("div","writing-toolbar");
  const previewButton=el("button","active","Preview"),sourceButton=el("button","","Source / Edit"),save=el("button","","Save file"),revert=el("button","","Discard edits"),status=el("span","muted","Saved local file");
  const preview=el("article","writing-preview"+(literature ? " literature-preview" : ""));
  const previewMarkdown=value=>literature ? literatureMarkdown(value,topics,title=>{if(preview.hidden)previewButton.click();return preview.querySelector("#"+title.id);}) : markdown(value.replace(/^#\s+[^\n]+\n?/, ""));
  preview.append(previewMarkdown(doc.markdown));
  const editor=el("textarea","writing-source");editor.value=doc.markdown;editor.spellcheck=false;editor.setAttribute("aria-label","Markdown source");editor.hidden=true;
  writingDraft={editor,original:doc.markdown,hash:location.hash};
  const update=()=>{const dirty=editor.value!==writingDraft.original;save.disabled=!dirty;revert.disabled=!dirty;status.textContent=dirty?"Unsaved changes":"Saved local file";};
  previewButton.addEventListener("click",()=>{preview.replaceChildren(previewMarkdown(editor.value));preview.hidden=false;editor.hidden=true;previewButton.classList.add("active");sourceButton.classList.remove("active");});
  sourceButton.addEventListener("click",()=>{preview.hidden=true;editor.hidden=false;sourceButton.classList.add("active");previewButton.classList.remove("active");});
  editor.addEventListener("input",update);
  revert.addEventListener("click",()=>{if(confirm("Discard unsaved edits to this file?")){editor.value=writingDraft.original;preview.replaceChildren(previewMarkdown(editor.value));update();}});
  save.addEventListener("click",async()=>{
    save.disabled=true;status.textContent="Saving…";
    const submitted=editor.value;
    try {
      const response=await fetch("/api/writing/"+encodeURIComponent(doc.id),{method:"POST",headers:{"Content-Type":"application/json","X-Write-Token":doc.write_token},body:JSON.stringify({markdown:submitted,version:doc.version})});
      if(!response.ok)throw new Error(response.status===409?"File changed on disk. Copy your draft, reload and merge before saving.":`Save failed (${response.status}). Your draft is still here.`);
      const saved=await response.json();doc.version=saved.version;
      if(writingDraft?.editor===editor){writingDraft.original=submitted;update();status.textContent=editor.value===submitted?"Saved · previous version backed up":"Saved earlier draft · newer edits unsaved";}
    } catch(error){status.textContent=error.message;save.disabled=false;}
  });
  toolbar.setAttribute("aria-label","Document editing");status.setAttribute("role","status");
  toolbar.append(previewButton,sourceButton,save,revert,status);
  content.append(toolbar);
  if (!literature) content.append(el("p","muted","Edits save to this allowlisted Markdown file only. Each save preserves its previous version locally."));
  else save.title="Save this Markdown file; its previous version is backed up locally";
  content.append(preview,editor);update();
}

async function render() {
  if(hasUnsavedWriting() && !confirm("Discard unsaved Markdown edits and leave this file?")) {
    history.replaceState(null,"",writingDraft.hash);return;
  }
  writingDraft=null;
  const ticket=++renderId;
  pendingRequest?.abort();pendingRequest=new AbortController();const signal=pendingRequest.signal;
  let [view,...parts]=location.hash.slice(1).split("/");view=view||"home";
  content.dataset.ready="false";content.replaceChildren();collection.replaceChildren();collection.hidden=true;document.body.classList.remove("has-collection","literature-view");
  heading.textContent={home:"Home",experiments:"Experiments",trajectories:"Trajectories",lineage:"Lineage",writing:"Writing",literature:"Literature"}[view]||"Workspace";
  document.querySelectorAll("#nav a").forEach(a=>{a.classList.toggle("active",a.dataset.view===view);});
  try {
    parts=parts.map(decodeURIComponent);
    if(view==="experiments") await experimentsPage(parts[0],signal,ticket);
    else if(view==="trajectories"&&parts[0]) await runPage(parts[0],parts[1],parts[2],signal,ticket);
    else if(view==="trajectories") await trajectoriesPage(signal,ticket);
    else if(view==="lineage") await lineagePage(parts[0],signal,ticket);
    else if(view==="writing") await writingPage(parts[0],signal,ticket);
    else if(view==="literature") await writingPage("literature-review",signal,ticket,"Literature");
    else {
      const d=await api("home",signal);if(ticket!==renderId)return;
      const summary=panel("L1 simple kernels · Stage 1");pair(summary,"Status",statusLabel(d.campaign.status));pair(summary,"Trained",d.campaign.trained);pair(summary,"Deferred",d.campaign.deferred);content.append(summary);
      content.append(el("h2","section-title","Research workspace"),el("p","muted","Explore experiments, inspect the evidence, and develop your own research direction."));
      content.append(link("Literature review →", "#literature", "writing-link"));
      for(const ex of d.experiments.slice(0,3))content.append(link(ex.title,"#experiments/"+ex.id,"writing-link"));
    }
    document.title=heading.textContent+" · Kernel RLM";content.dataset.ready="true";
  } catch(error) {
    if(error.name!=="AbortError"&&ticket===renderId){notice(error.message);content.append(link("Back to runs","#trajectories","back"));}
  }
}
window.addEventListener("hashchange",render);
render();
