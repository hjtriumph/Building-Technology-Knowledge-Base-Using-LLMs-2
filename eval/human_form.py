from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REQUIRED = {
    "sample_id",
    "question",
    "gold_answer",
    "answer",
    "citations",
    "completeness_human",
    "faithfulness_human",
    "notes",
}

TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>人工抽查 · Building Technology KB</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --line:#e3e6ea; --ink:#1f2328; --muted:#6b7280; --accent:#2563eb; --ok:#047857; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--ink); }
  .bar { position:sticky; top:0; z-index:10; background:rgba(246,247,249,.96); border-bottom:1px solid var(--line); padding:14px 20px; display:flex; gap:16px; align-items:center; justify-content:space-between; flex-wrap:wrap; }
  .bar h1 { font-size:19px; margin:0; }
  .muted { color:var(--muted); font-size:13px; }
  .actions { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  button { font:inherit; padding:8px 12px; border-radius:8px; border:1px solid var(--line); background:#fff; cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  button.danger { color:#b91c1c; }
  main { max-width:1040px; margin:0 auto; padding:18px 20px 80px; }
  .intro { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .intro ul { margin:8px 0 0 18px; padding:0; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:16px; }
  .card.done { border-color:#a7f3d0; }
  .head { display:flex; gap:10px; align-items:baseline; margin-bottom:10px; }
  .sid { color:var(--accent); font-weight:600; font-size:13px; }
  .q { font-weight:600; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  @media (max-width:820px) { .grid { grid-template-columns:1fr; } }
  h4 { font-size:13px; color:var(--muted); margin:0 0 6px; font-weight:600; }
  .box { white-space:pre-wrap; word-break:break-word; border:1px solid var(--line); border-radius:8px; padding:10px 12px; max-height:230px; overflow:auto; font-size:14px; }
  .gold { background:#f0f7ff; }
  details { margin-top:12px; }
  summary { cursor:pointer; color:var(--muted); font-size:13px; }
  pre.cites { white-space:pre-wrap; word-break:break-word; background:#fafafa; border:1px solid var(--line); border-radius:8px; padding:10px 12px; max-height:220px; overflow:auto; font-size:13px; margin:8px 0 0; }
  .rate { display:flex; gap:24px; flex-wrap:wrap; margin-top:14px; align-items:center; }
  .rate .item { display:flex; gap:8px; align-items:center; }
  .btns { display:flex; gap:6px; }
  .rate-btn { min-width:52px; }
  .rate-btn.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  .notes { width:100%; margin-top:12px; padding:9px 12px; border:1px solid var(--line); border-radius:8px; font:inherit; }
  .badge { font-size:12px; color:var(--ok); }
  label.check { font-size:13px; color:var(--muted); display:flex; gap:6px; align-items:center; }
</style>
</head>
<body>
<div class="bar">
  <div>
    <h1>人工抽查</h1>
    <div class="muted" id="progress"></div>
  </div>
  <div class="actions">
    <label class="check"><input type="checkbox" id="onlyTodo" /> 只看未完成</label>
    <button class="primary" id="export">导出 CSV</button>
    <button class="danger" id="reset">重置</button>
  </div>
</div>
<main>
  <div class="intro">
    <b>怎么打分</b>
    <ul>
      <li><b>完整度</b>：和参考答案比。1 = 要点全覆盖；0.5 = 覆盖一部分；0 = 主要缺失或答错。</li>
      <li><b>忠实度</b>：和引用原文比。1 = 回答基本都被引用支持；0.5 = 部分没有；0 = 大部分编造或矛盾。</li>
      <li>拒答（INSUFFICIENT_CONTEXT）的题：完整度填 0，忠实度留空。</li>
      <li>无答案题（参考答案以 “The corpus does not state” 开头）：两列留空。</li>
      <li>进度自动保存在浏览器本地，刷新不会丢。评完点“导出 CSV”。</li>
    </ul>
  </div>
  <div id="list"></div>
</main>
<script>
const SAMPLES = __DATA__;
const FIELDS = Object.keys(SAMPLES[0] || {});
const STORAGE_KEY = "human_review_form_v1";
let state = {};
try { state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") || {}; } catch (e) { state = {}; }

function entry(id) {
  if (!state[id]) state[id] = { c: "", f: "", notes: "" };
  return state[id];
}
function persist() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (e) { /* file:// may block storage */ }
}
function updateProgress() {
  let c = 0, f = 0;
  SAMPLES.forEach(s => {
    const e = state[s.sample_id] || {};
    if (e.c !== undefined && e.c !== "") c += 1;
    if (e.f !== undefined && e.f !== "") f += 1;
  });
  document.getElementById("progress").textContent =
    "完整度已评 " + c + "/" + SAMPLES.length + " · 忠实度已评 " + f + "/" + SAMPLES.length;
}
function ratingGroup(sample, field) {
  const wrap = document.createElement("div");
  wrap.className = "btns";
  [["", "留空"], ["0", "0"], ["0.5", "0.5"], ["1", "1"]].forEach(([value, label]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "rate-btn";
    b.textContent = label;
    if ((entry(sample.sample_id)[field] || "") === value) b.classList.add("active");
    b.onclick = () => {
      const e = entry(sample.sample_id);
      e[field] = value;
      persist();
      wrap.querySelectorAll(".rate-btn").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      updateProgress();
      syncDone(sample.sample_id);
    };
    wrap.appendChild(b);
  });
  return wrap;
}
function syncDone(id) {
  const card = document.querySelector('[data-card="' + id + '"]');
  if (!card) return;
  const e = entry(id);
  card.classList.toggle("done", e.c !== "" && e.f !== "");
  if (document.getElementById("onlyTodo").checked && e.c !== "" && e.f !== "") {
    card.style.display = "none";
  }
}
function buildCard(s) {
  const card = document.createElement("article");
  card.className = "card";
  card.dataset.card = s.sample_id;

  const head = document.createElement("div");
  head.className = "head";
  const sid = document.createElement("span");
  sid.className = "sid";
  sid.textContent = s.sample_id;
  const q = document.createElement("span");
  q.className = "q";
  q.textContent = s.question;
  head.append(sid, q);
  card.appendChild(head);

  const grid = document.createElement("div");
  grid.className = "grid";
  [["参考答案", "gold", s.gold_answer], ["模型回答", "box-answer", s.answer]].forEach(([title, cls, text]) => {
    const cell = document.createElement("div");
    const h = document.createElement("h4");
    h.textContent = title;
    const box = document.createElement("div");
    box.className = "box " + cls;
    box.textContent = text || "";
    cell.append(h, box);
    grid.appendChild(cell);
  });
  card.appendChild(grid);

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "引用原文（faithfulness 依据）";
  const pre = document.createElement("pre");
  pre.className = "cites";
  pre.textContent = s.citations || "（无引用）";
  details.append(summary, pre);
  card.appendChild(details);

  const rate = document.createElement("div");
  rate.className = "rate";
  [["完整度", "c"], ["忠实度", "f"]].forEach(([label, field]) => {
    const item = document.createElement("div");
    item.className = "item";
    const span = document.createElement("span");
    span.textContent = label;
    item.append(span, ratingGroup(s, field));
    rate.appendChild(item);
  });
  card.appendChild(rate);

  const notes = document.createElement("input");
  notes.className = "notes";
  notes.placeholder = "备注（可选）";
  notes.value = entry(s.sample_id).notes || "";
  notes.oninput = () => { entry(s.sample_id).notes = notes.value; persist(); };
  card.appendChild(notes);

  return card;
}
function render() {
  const list = document.getElementById("list");
  list.innerHTML = "";
  SAMPLES.forEach(s => {
    const card = buildCard(s);
    const e = entry(s.sample_id);
    if (e.c !== "" && e.f !== "") card.classList.add("done");
    if (document.getElementById("onlyTodo").checked && e.c !== "" && e.f !== "") card.style.display = "none";
    list.appendChild(card);
  });
  updateProgress();
}
function csvCell(value) {
  const text = value === undefined || value === null ? "" : String(value);
  if (/[",\r\n]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
  return text;
}
function exportCsv() {
  const lines = [FIELDS.map(csvCell).join(",")];
  SAMPLES.forEach(s => {
    const e = entry(s.sample_id);
    const row = Object.assign({}, s, {
      completeness_human: e.c,
      faithfulness_human: e.f,
      notes: e.notes,
    });
    lines.push(FIELDS.map(f => csvCell(row[f])).join(","));
  });
  const blob = new Blob(["﻿" + lines.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "human_review_filled.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
document.getElementById("export").addEventListener("click", exportCsv);
document.getElementById("onlyTodo").addEventListener("change", render);
document.getElementById("reset").addEventListener("click", () => {
  if (!confirm("确定清空所有打分？")) return;
  state = {};
  persist();
  render();
});
render();
</script>
</body>
</html>
"""


def build_html(csv_path: str, out_path: str) -> None:
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = [dict(row) for row in reader]

    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload)
    Path(out_path).write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a local HTML form for human spot-check")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or str(Path(args.csv).with_suffix(".html"))
    build_html(args.csv, out)
    print(f"[human] wrote {out}")


if __name__ == "__main__":
    main()
