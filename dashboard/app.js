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
      const docs = {"README.md":"README", "first-campaign.md":"first-campaign", "protocol.md":"protocol", "artifact-publication.md":"artifact-publication", "experiment-tracking.md":"experiment-tracking"};
      const anchor = el("a", "", label);
      if (docs[target]) anchor.href = "#writing/" + docs[target];
      else if (target.endsWith("/dashboard/guide.html")) anchor.href = "/archive/guide";
      else if (/^https?:\/\//i.test(target)) { anchor.href = target; anchor.rel = "noreferrer"; }
      else anchor.append(document.createTextNode(" (" + target + ")"));
      parent.append(anchor);
    }
    end = match.index + token.length;
  }
  parent.append(document.createTextNode(source.slice(end)));
}
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
function notice(message) { content.append(el("div", "warning", message)); }
function rows(parent, records, target) {
  for (const record of records) {
    const row = link("", target(record), "run-list-row");
    row.append(el("strong", "run-name", record.title || record.id),
      el("span", "run-count", record.detail || (record.trajectory_count == null ? "No episode export" : `${record.trajectory_count} trajectories`)), badge(["running", "paused"].includes(record.status) ? "saved: " + record.status : record.status));
    parent.append(row);
  }
}
function experimentNav(experiments, selected) {
  document.body.classList.add("has-collection");
  collection.hidden = false;
  collection.replaceChildren(el("div", "collection-heading", "L1 post-training"));
  collection.append(el("p", "collection-context", "One campaign · training and supporting work"));
  const search = el("input", "collection-search");
  search.type = "search"; search.placeholder = "Search records…"; search.setAttribute("aria-label", "Search campaign records");
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
      const a = link("", "#experiments/" + encodeURIComponent(ex.id), "experiment-row" + (selected === ex.id ? " selected" : ""));
      if (selected === ex.id) a.setAttribute("aria-current", "page");
      a.append(categoryTag(ex.category), el("strong", "", ex.title), el("small", "", `${ex.runs.length} workloads`));
      list.append(a);
    }
    if (!matches.length) list.append(el("p", "muted", "No matching records."));
  };
  search.addEventListener("input", update); filter.addEventListener("change", update);
  collection.append(search, filter, list); update();
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
  actions.append(categoryTag(selected.category), el("span", "muted", `${selected.campaign || "Campaign"} · saved status: ${statusLabel(selected.status)}`));
  content.append(actions);
  const context = el("div", "campaign-context");
  context.append(el("p", "", "The project began with Modal trials and moved to CloudRift after the compute grant. These records cover training, setup checks, applied fixes and pilot branches—not independent proof of improvement."), link("Read the campaign history →", "#writing/first-campaign"));
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
function trajectoryCard(t, fallbackRun) {
  const runId = t.run?.id || fallbackRun;
  const card = el("article", "trajectory-card");
  const title = t.title || t.task || t.id;
  card.append(link(title, route(runId, t.id), "trajectory-title"));
  const meta = el("p", "trajectory-meta", `${t.task || "Task unavailable"} · Attempt ${t.attempt ?? "—"}`);
  card.append(meta);raw(card,t.id,"Episode ID");
  const tags = el("div", "detail-actions"); experimentTags(tags, t.experiments); card.append(tags);
  card.append(el("p", "muted", `${t.benchmark || "Benchmark unavailable"} · ${t.model || "Model unavailable"}`));
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
  heading.textContent="L1 campaign lineage";
  content.append(el("p","scope-note","Recorded research follow-ups—not checkpoint inheritance. The Modal and native TRL trainer lineages are separate; pilot branches are not cumulative training progress."));
  const byId=new Map(data.experiments.map(ex=>[ex.id,ex]));
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
  for(const ex of data.experiments) {
    const pos=positions.get(ex.id),card=el("article","lineage-node"+(focus===ex.id?" selected":""));card.style.left=pos.x+"px";card.style.top=pos.y+"px";
    card.append(categoryTag(ex.category),link(ex.title,"#experiments/"+encodeURIComponent(ex.id),"writing-link"),el("small","muted",`${ex.runs.length} workloads · saved: ${statusLabel(ex.status)}`));
    card.title=ex.scope_note||ex.title;graph.append(card);
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
  if(focus&&positions.has(focus)){viewport.scrollLeft=Math.max(0,positions.get(focus).x-100);viewport.scrollTop=positions.get(focus).y;}
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
function renderEvents(events,parent,{expanded=false}={}) {
  for(const [position,event] of events.entries()) {
    const role=event.role||"child";
    const node=el("article","event trace-event role-"+role);
    const title=el("div","event-head");title.append(el("span","event-label",`#${(event.index??position)+1} · ${eventTitle(event,position)}`));
    if(event.usage)title.append(el("span","muted",`in ${event.usage.prompt_tokens??"—"} · out ${event.usage.completion_tokens??"—"}`));node.append(title);
    if(event.source==="kernel_child_progress")raw(node,event.records,"Recorded child progress");
    else if(Array.isArray(event.records))for(const [i,child] of event.records.entries()) {
      const childBlock=el("details","tool-call");childBlock.open=expanded;childBlock.append(el("summary","",`Child ${i+1} · ${statusLabel(child.status)}`),panel("Assignment",child.prompt||child.assignment||child.input),panel("Response",child.response||child.output));raw(childBlock,child);node.append(childBlock);
    }
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
    raw(node,event,"Raw event");parent.append(node);
  }
}
function renderStepBrowser(events,parent) {
  const toolbar=el("div","trace-toolbar"),stepsButton=el("button","active","Steps"),timelineButton=el("button","","Timeline");
  const filter=el("input","search");filter.type="search";filter.placeholder="Filter steps…";filter.setAttribute("aria-label","Filter trajectory steps");
  toolbar.append(filter,stepsButton,timelineButton);parent.append(toolbar);
  const body=el("div","step-browser");parent.append(body);let mode="steps",selected=0;
  const update=()=>{
    const matches=events.map((event,index)=>({event,index})).filter(({event,index})=>[eventTitle(event,index),event.role,event.content,event.reasoning,JSON.stringify(event.tool_calls)].join(" ").toLowerCase().includes(filter.value.toLowerCase()));
    body.replaceChildren();body.className=mode==="steps"?"step-browser":"timeline-list";
    stepsButton.classList.toggle("active",mode==="steps");timelineButton.classList.toggle("active",mode==="timeline");
    if(!matches.length){body.append(el("p","muted","No matching steps."));return;}
    if(mode==="timeline"){renderEvents(matches.map(item=>item.event),body);return;}
    if(!matches.some(item=>item.index===selected))selected=matches[0].index;
    const list=el("div","step-index"),detail=el("div","step-detail");list.setAttribute("aria-label","Trajectory steps");
    for(const {event,index} of matches){const button=el("button","step-item"+(selected===index?" selected":""),`${index+1}  ${eventTitle(event,index)}`);button.setAttribute("aria-pressed",String(selected===index));button.addEventListener("click",()=>{const offset=list.scrollTop;selected=index;update();body.querySelector(".step-index").scrollTop=offset;});list.append(button);}
    renderEvents([events[selected]],detail,{expanded:true});body.append(list,detail);
  };
  stepsButton.addEventListener("click",()=>{mode="steps";update();});timelineButton.addEventListener("click",()=>{mode="timeline";update();});filter.addEventListener("input",update);update();
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
  const metadataRow=(label,value)=>{const row=el("div","metadata-row");row.append(el("dt","",label),el("dd","",text(value)));metadata.append(row);};
  metadataRow("Benchmark",t.benchmark);metadataRow("Model",t.model || run.model);
  const runRow=el("div","metadata-row"),runStats=el("dd","trajectory-run-stats");
  runStats.append(el("span","",`${t.metrics?.steps ?? "—"} steps`),el("span","",`${t.metrics?.tool_calls ?? "—"} tool calls`),el("span","muted",`Started ${t.started_at || "not recorded"}`));
  runRow.append(el("dt","","Run"),runStats);metadata.append(runRow);content.append(metadata);
  const metrics=el("div","trajectory-metrics");
  for(const [label,value] of [["Peak input context",t.metrics?.peak_context == null ? "Unavailable" : `${t.metrics.peak_context.toLocaleString()} tokens`],["Prompt cache",typeof t.metrics?.prompt_cache === "number" ? t.metrics.prompt_cache.toFixed(1)+"%" : "Unavailable"],["Commands failed",t.metrics?.commands_failed]]) {
    const stat=el("div");stat.append(el("span","muted",label),el("strong","",text(value)));metrics.append(stat);
  }
  content.append(metrics);
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
async function writingPage(id,signal,ticket) {
  const index=await api("writing",signal);if(ticket!==renderId)return;
  document.body.classList.add("has-collection");collection.hidden=false;
  collection.replaceChildren(el("div","collection-heading","Writing"));
  const tree=el("div","file-tree");tree.append(el("div","muted","docs / research /"));
  for(const doc of index.documents){const a=link(doc.path?.split("/").pop()||doc.title,"#writing/"+encodeURIComponent(doc.id),"writing-link"+(id===doc.id?" selected":""));tree.append(a);}
  collection.append(tree);
  const selected=id||index.documents[0]?.id;if(!selected)return;
  const doc=await api("writing/"+encodeURIComponent(selected),signal);if(ticket!==renderId)return;
  heading.textContent=doc.path||doc.title;
  const toolbar=el("div","writing-toolbar");
  const previewButton=el("button","active","Preview"),sourceButton=el("button","","Source / Edit"),save=el("button","","Save file"),revert=el("button","","Discard edits"),status=el("span","muted","Saved local file");
  const preview=el("article","writing-preview");preview.append(markdown(doc.markdown));
  const editor=el("textarea","writing-source");editor.value=doc.markdown;editor.spellcheck=false;editor.setAttribute("aria-label","Markdown source");editor.hidden=true;
  writingDraft={editor,original:doc.markdown,hash:location.hash};
  const update=()=>{const dirty=editor.value!==writingDraft.original;save.disabled=!dirty;revert.disabled=!dirty;status.textContent=dirty?"Unsaved changes":"Saved local file";};
  previewButton.addEventListener("click",()=>{preview.replaceChildren(markdown(editor.value));preview.hidden=false;editor.hidden=true;previewButton.classList.add("active");sourceButton.classList.remove("active");});
  sourceButton.addEventListener("click",()=>{preview.hidden=true;editor.hidden=false;sourceButton.classList.add("active");previewButton.classList.remove("active");});
  editor.addEventListener("input",update);
  revert.addEventListener("click",()=>{if(confirm("Discard unsaved edits to this file?")){editor.value=writingDraft.original;preview.replaceChildren(markdown(editor.value));update();}});
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
  toolbar.append(previewButton,sourceButton,save,revert,status);content.append(toolbar,el("p","muted","Edits save to this allowlisted Markdown file only. Each save preserves its previous version locally."),preview,editor);update();
}

async function render() {
  if(hasUnsavedWriting() && !confirm("Discard unsaved Markdown edits and leave this file?")) {
    history.replaceState(null,"",writingDraft.hash);return;
  }
  writingDraft=null;
  const ticket=++renderId;
  pendingRequest?.abort();pendingRequest=new AbortController();const signal=pendingRequest.signal;
  let [view,...parts]=location.hash.slice(1).split("/");view=view||"home";
  content.dataset.ready="false";content.replaceChildren();collection.replaceChildren();collection.hidden=true;document.body.classList.remove("has-collection");
  heading.textContent={home:"Home",experiments:"Experiments",trajectories:"Trajectories",lineage:"Lineage",writing:"Writing"}[view]||"Workspace";
  document.querySelectorAll("#nav a").forEach(a=>{a.classList.toggle("active",a.dataset.view===view);});
  try {
    parts=parts.map(decodeURIComponent);
    if(view==="experiments") await experimentsPage(parts[0],signal,ticket);
    else if(view==="trajectories"&&parts[0]) await runPage(parts[0],parts[1],parts[2],signal,ticket);
    else if(view==="trajectories") await trajectoriesPage(signal,ticket);
    else if(view==="lineage") await lineagePage(parts[0],signal,ticket);
    else if(view==="writing") await writingPage(parts[0],signal,ticket);
    else {
      const d=await api("home",signal);if(ticket!==renderId)return;
      const summary=panel("First L1 campaign");pair(summary,"Status",statusLabel(d.campaign.status));pair(summary,"Trained",d.campaign.trained);pair(summary,"Deferred",d.campaign.deferred);content.append(summary);
      content.append(el("h2","section-title","Research workspace"),el("p","muted","Explore experiments, inspect the evidence, and develop your own research direction."));
      for(const ex of d.experiments.slice(0,3))content.append(link(ex.title,"#experiments/"+ex.id,"writing-link"));
    }
    document.title=heading.textContent+" · Kernel RLM";content.dataset.ready="true";
  } catch(error) {
    if(error.name!=="AbortError"&&ticket===renderId){notice(error.message);content.append(link("Back to runs","#trajectories","back"));}
  }
}
window.addEventListener("hashchange",render);
render();
