#!/usr/bin/env python3
"""Build docs/research/index.json — the database docs/research/viewer.html reads.

Scans plan/ results/ analysis/, the README ledger and the todo/ + DONE queues, and emits
one record per doc with its thread, stage, question, verdict, jobs and links to sibling
docs. Hand edits belong in the "overrides" block of index.json; a rebuild preserves it.

    python3 docs/research/tools/build_index.py

Writes index.json (the database) and index.js (the same payload as a `window.RESEARCH_INDEX`
assignment, so the viewer also works when opened straight off disk, where fetch is blocked).
"""
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
RESEARCH = TOOLS.parent
DOCS = RESEARCH.parent
REPO = DOCS.parent
OUT_JSON = RESEARCH / "index.json"
OUT_JS = RESEARCH / "index.js"
STAGES = ("plan", "results", "analysis")
DOC_EXT = ("md", "html", "csv", "png")
SKIP = {"README.md", "TEMPLATE.md"}

# a doc reference anywhere in prose: the stage dir always sits right before the filename
RE_PATHREF = re.compile(r"\b(plan|results|analysis)/(\d{4}-\d{2}-\d{2}_[\w.\-]+?\.(?:md|html|csv|png))\b")
# a bare filename, as used by most `prior cycle:` fields
RE_BAREREF = re.compile(r"\b(\d{4}-\d{2}-\d{2}_[\w.\-]+?\.(?:md|html|csv|png))\b")
RE_JOBRUN = re.compile(r"\b(?:BSC|JZ|jean-zay)\s*(?:job\s*)?:?\s*(\d{7,9}(?:\s*[/–-]\s*\d{4,9})*)", re.I)
RE_MDLINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
RE_TITLE_H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)
RE_TITLE_HTML = re.compile(r"<title>(.*?)</title>", re.S | re.I)
# tokens too generic to imply two docs are the same cycle
STOP = {"slides", "plan", "vs", "and", "the", "at", "of", "on", "report", "ablation", "sweep", "experiments"}


def read(p):
    return p.read_text(encoding="utf-8", errors="replace")


def jobs_in(text):
    """Job ids, expanding `BSC:45296347 / 45296348 / 45296349` runs into one id each."""
    out = []
    for run in RE_JOBRUN.findall(text or ""):
        for n in re.split(r"\s*[/–-]\s*", run):
            n = n.strip()
            if n and n not in out:
                out.append(n)
    return out


def split_row(line):
    """Cells of a markdown table row, `| a | b |` -> ['a','b']."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def strip_md(s):
    """Prose with markdown links flattened to their label and code ticks removed."""
    s = RE_MDLINK.sub(r"\1", s or "")
    return s.replace("`", "").replace("**", "").strip()


# ---------------------------------------------------------------- README ledger
def parse_readme(threads_seen):
    """Thread blurbs, open questions, and the per-doc ledger row (question + verdict)."""
    text = read(RESEARCH / "README.md")
    threads, ledger, open_q = {}, {}, {}

    sections = re.split(r"^## ", text, flags=re.M)[1:]
    for sec in sections:
        head, _, body = sec.partition("\n")
        head = head.strip()

        if head.startswith("Open questions"):
            for line in body.splitlines():
                if not line.startswith("|") or set(line) <= set("|- "):
                    continue
                cells = split_row(line)
                if len(cells) < 2 or cells[0].lower() == "thread":
                    continue
                key = strip_md(cells[0]).split("]")[0]
                open_q[key] = strip_md(cells[1])
            continue

        if head.startswith("Cross-thread"):
            tid, ttitle = "cross", "cross-thread — spans every thread"
        else:
            tid, _, ttitle = head.partition(" — ")
            tid = tid.strip()
            if tid not in threads_seen:
                continue

        # the blurb is everything before the table; `**Open.**` trails it
        blurb = body.split("\n|", 1)[0].strip()
        openp = ""
        m = re.search(r"\*\*Open\.\*\*(.*?)(?=\n##|\Z)", body, re.S)
        if m:
            openp = strip_md(" ".join(m.group(1).split()))
        threads[tid] = {
            "id": tid,
            "title": ttitle.strip(),
            "blurb": strip_md(" ".join(blurb.split())),
            "open": openp,
        }

        for line in body.splitlines():
            if not line.startswith("|") or set(line) <= set("|- "):
                continue
            cells = split_row(line)
            if len(cells) < 3 or cells[0].lower() == "date":
                continue
            link = RE_MDLINK.search(cells[1])
            if not link:
                continue
            rel = link.group(2).split("#")[0]
            stage_cell = cells[2]
            status = ""
            sm = re.search(r"\*\*(open|closed)\*\*", stage_cell)
            if sm:
                status = sm.group(1)
            entry = {
                "thread": tid,
                "label": link.group(1),
                "stage": stage_cell.split(",")[0].strip(),
                "status": status,
                "question": strip_md(cells[3]) if len(cells) > 3 else "",
                "verdict": strip_md(cells[4]) if len(cells) > 4 else "",
            }
            # the cross-thread table has one "Holds" column instead of question+verdict
            if tid == "cross" and len(cells) == 4:
                entry["question"], entry["verdict"] = "", strip_md(cells[3])
            ledger[rel] = entry

    return threads, ledger, open_q


# ---------------------------------------------------------------- doc records
def thread_of(name, known):
    """Longest known thread tag that the filename carries after its date."""
    stem = name[11:]  # drop `YYYY-MM-DD_`
    best = ""
    for t in known:
        if stem.startswith(t + "_") and len(t) > len(best):
            best = t
    return best


def meta_fields(head):
    """The `Created … · thread … · jobs … · deck …` header line of a plan/analysis."""
    out = {}
    m = re.search(r"Created\s+(\d{4}-\d{2}-\d{2})", head) or re.search(r"\*\*Date:\*\*\s*(\d{4}-\d{2}-\d{2})", head)
    if m:
        out["created"] = m.group(1)
    m = re.search(r"thread\s+`([\w]+)`", head)
    if m:
        out["thread"] = m.group(1)
    for key, pat in (("arms", r"arms?:\s*(.+?)(?=\s·|\n|$)"),
                     ("control", r"controls?:\s*(.+?)(?=\s·|\n|$)"),
                     ("jobs_raw", r"jobs:\s*(.+?)(?=\s·|\n|$)"),
                     ("deck_raw", r"deck:\s*(.+?)(?=\s·|\n|$)")):
        m = re.search(pat, head)
        if m:
            v = strip_md(m.group(1))
            if v and v != "_pending_":
                out[key] = v
    out["todos"] = sorted({int(n) for n in re.findall(r"TODO\s+(\d+)", head)})
    return out


def scan_docs(known_threads):
    docs = {}
    for stage in STAGES:
        d = RESEARCH / stage
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.name in SKIP or p.suffix.lstrip(".") not in DOC_EXT:
                continue
            if not re.match(r"\d{4}-\d{2}-\d{2}_", p.name):
                print(f"  ! undated filename, skipped: {stage}/{p.name}", file=sys.stderr)
                continue
            rel = f"{stage}/{p.name}"
            ext = p.suffix.lstrip(".")
            thread = thread_of(p.name, known_threads)
            slug = p.stem[11 + len(thread) + 1:] if thread else p.stem[11:]
            rec = {
                "id": rel,
                "stage": stage,
                "ext": ext,
                "date": p.name[:10],
                "thread": thread or "unfiled",
                "slug": slug,
                "title": "",
                "bytes": p.stat().st_size,
                "question": "",
                "verdict": "",
                "status": "",
                "jobs": [],
                "jobs_seen": [],
                "todos": [],
                "meta": {},
                "in_ledger": False,
                "body": None,
            }
            if ext in ("md", "csv"):
                text = read(p)
                rec["body"] = text
                if ext == "md":
                    m = RE_TITLE_H1.search(text)
                    rec["title"] = strip_md(m.group(1)) if m else p.stem
                    head = "\n".join(text.splitlines()[:12])
                    rec["meta"] = meta_fields(head)
                    rec["todos"] = rec["meta"].pop("todos", [])
                    rec["jobs"] = jobs_in(rec["meta"].get("jobs_raw", ""))
                    rec["jobs_seen"] = [j for j in jobs_in(text) if j not in rec["jobs"]]
                else:
                    rec["title"] = p.stem
            else:
                if ext == "html":
                    text = read(p)
                    m = RE_TITLE_HTML.search(text)
                    rec["title"] = strip_md(" ".join(m.group(1).split())) if m else p.stem
                    rec["jobs_seen"] = jobs_in(text)
                    rec["_text"] = text
                else:
                    rec["title"] = p.stem
            docs[rel] = rec
    return docs


# ---------------------------------------------------------------- todo queues
def parse_queue(path, kind):
    """Rows of todo/<date>.md (`Status|#|Item|Thread|Paper|Note`) or DONE.md
    (`#|Item|Thread|Paper|Closed|Status`)."""
    text = read(path)
    rows, order = [], 0
    for line in text.splitlines():
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = split_row(line)
        if len(cells) < 5 or cells[1].lower() in ("#", "item"):
            continue
        if kind == "open":
            status, num, item, thread, paper, note = (cells + [""] * 6)[:6]
        else:
            num, item, thread, paper, closed, note = (cells + [""] * 6)[:6]
            sm = re.match(r"`?(done|dropped)", strip_md(note))
            status = sm.group(1) if sm else "closed"
        if not re.fullmatch(r"\d+", num.strip()):
            continue
        order += 1
        rows.append({
            "num": int(num),
            "status": status.strip(),
            "item": strip_md(item),
            "thread": strip_md(thread),
            "paper": strip_md(paper),
            "note": strip_md(note),
            "note_md": note.strip(),
            "closed": strip_md(closed) if kind == "done" else "",
            "state": kind,
            "source": str(path.relative_to(REPO)),
            "order": order,
            "jobs": jobs_in(note),
            "docs": sorted({f"{a}/{b}" for a, b in RE_PATHREF.findall(note)}),
            "updates": [],
        })

    # trailing `**TODO 7 status, 2026-09-04.** …` paragraphs
    for m in re.finditer(r"\*\*TODO (\d+) status,\s*([\d-]+)\.\*\*(.*?)(?=\n\*\*TODO \d+ status|\Z)", text, re.S):
        num, when, body = int(m.group(1)), m.group(2), m.group(3)
        for r in rows:
            if r["num"] == num:
                r["updates"].append({"date": when, "text": body.strip()})
    return rows


# ---------------------------------------------------------------- links
def add_link(links, seen, a, b, kind):
    if a == b or not a or not b:
        return
    key = (a, b, kind)
    if key in seen:
        return
    seen.add(key)
    links.append({"from": a, "to": b, "kind": kind})


def resolve(ref, by_base):
    """A raw reference -> a doc id, via its `stage/file` form or a unique basename."""
    m = RE_PATHREF.search(ref)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = RE_BAREREF.search(ref)
    if m:
        hits = by_base.get(m.group(1), [])
        return hits[0] if len(hits) == 1 else None
    return None


def build_links(docs, todos, ledger):
    by_base = {}
    for rel in docs:
        by_base.setdefault(rel.split("/", 1)[1], []).append(rel)

    links, seen = [], set()
    for rel, entry in ledger.items():
        for a, b in RE_PATHREF.findall(entry["question"] + " " + entry["verdict"]):
            if rel in docs and f"{a}/{b}" in docs:
                add_link(links, seen, rel, f"{a}/{b}", "ref")

    for rel, rec in docs.items():
        meta = rec.get("meta", {})
        text = rec.get("body") or rec.get("_text") or ""

        # `prior cycle:` — the previous cycle of the same question
        head = "\n".join(text.splitlines()[:12])
        m = re.search(r"prior cycle:\s*(.+?)(?=\s·|\n|$)", head)
        if m:
            tgt = resolve(m.group(1), by_base)
            if tgt:
                add_link(links, seen, rel, tgt, "prior")
        # `deck:` — the results file this plan produced
        if meta.get("deck_raw"):
            tgt = resolve(meta["deck_raw"], by_base)
            if tgt:
                add_link(links, seen, rel, tgt, "deck")
        # every other doc this one mentions
        for a, b in RE_PATHREF.findall(text):
            add_link(links, seen, rel, f"{a}/{b}", "ref")

    # todo rows own the docs they point at
    for t in todos:
        tid = f"todo/{t['num']}"
        for d in t["docs"]:
            if d in docs:
                add_link(links, seen, tid, d, "todo")
        for u in t["updates"]:
            for a, b in RE_PATHREF.findall(u["text"]):
                if f"{a}/{b}" in docs:
                    add_link(links, seen, tid, f"{a}/{b}", "todo")
        for rel, rec in docs.items():
            if t["num"] in rec.get("todos", []):
                add_link(links, seen, tid, rel, "todo")

    # a .png/.csv companion belongs to the doc of the same name
    by_stem = {}
    for rel, rec in docs.items():
        by_stem.setdefault(Path(rel).stem, []).append(rel)
    for stem, group in by_stem.items():
        if len(group) > 1:
            host = min(group, key=lambda r: 0 if r.endswith((".md", ".html")) else 1)
            for other in group:
                add_link(links, seen, host, other, "asset")

    # weak fallback: same thread, different stage, close in time, shared slug words
    def toks(rec):
        return {w for w in re.split(r"[_\-.]", rec["slug"].lower()) if w and w not in STOP and len(w) > 2}

    items = [r for r in docs.values() if r["thread"] != "unfiled"]
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a["thread"] != b["thread"] or a["stage"] == b["stage"]:
                continue
            gap = abs((date.fromisoformat(a["date"]) - date.fromisoformat(b["date"])).days)
            if gap > 60:
                continue
            if len(toks(a) & toks(b)) >= 2 or gap == 0:
                add_link(links, seen, a["id"], b["id"], "slug")
    return links


# ---------------------------------------------------------------- main
def main():
    known_threads = []
    for m in re.finditer(r"^## ([\w]+) — ", read(RESEARCH / "README.md"), re.M):
        known_threads.append(m.group(1))
    known_threads.append("cross")

    threads, ledger, open_q = parse_readme(known_threads)
    docs = scan_docs(known_threads)

    for rel, entry in ledger.items():
        rec = docs.get(rel)
        if rec is None:
            print(f"  ! README points at a missing file: {rel}", file=sys.stderr)
            continue
        rec["in_ledger"] = True
        rec["question"] = entry["question"]
        rec["verdict"] = entry["verdict"]
        rec["status"] = entry["status"]
        if not rec["title"]:
            rec["title"] = entry["label"]
        if rec["thread"] == "unfiled":
            rec["thread"] = entry["thread"]

    todos = parse_queue(DOCS / "todo" / "02-09-2026.md", "open")
    todos += parse_queue(DOCS / "DONE.md", "done")

    links = build_links(docs, todos, ledger)

    for rec in docs.values():
        rec.pop("_text", None)
        if not rec["in_ledger"]:
            print(f"  · not in the README ledger: {rec['id']}", file=sys.stderr)

    for tid, t in threads.items():
        t["open"] = t["open"] or open_q.get(tid, "")

    prev = {}
    if OUT_JSON.exists():
        try:
            prev = json.loads(read(OUT_JSON))
        except json.JSONDecodeError:
            print("  ! index.json is not valid JSON; overrides not carried over", file=sys.stderr)
    overrides = prev.get("overrides") or {"hide": [], "unlink": [], "links": [], "docs": {}}

    # apply overrides: drop hidden docs, drop unlinked pairs, add manual links, patch fields
    hidden = set(overrides.get("hide") or [])
    docs = {k: v for k, v in docs.items() if k not in hidden}
    cut = {tuple(sorted(p)) for p in (overrides.get("unlink") or [])}
    links = [l for l in links
             if l["from"] in docs or l["from"].startswith("todo/")
             if l["to"] in docs
             if tuple(sorted((l["from"], l["to"]))) not in cut]
    seen = {(l["from"], l["to"], l["kind"]) for l in links}
    for l in overrides.get("links") or []:
        add_link(links, seen, l.get("from"), l.get("to"), l.get("kind", "manual"))
    for rel, patch in (overrides.get("docs") or {}).items():
        if rel in docs:
            docs[rel].update(patch)

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": "docs/research",
        "threads": [threads[t] for t in known_threads if t in threads],
        "docs": sorted(docs.values(), key=lambda r: (r["date"], r["stage"], r["slug"]), reverse=True),
        "todos": todos,
        "links": links,
        "overrides": overrides,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_JS.write_text("window.RESEARCH_INDEX=" + json.dumps(payload, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"{len(payload['docs'])} docs · {len(payload['todos'])} todos · {len(links)} links "
          f"-> {OUT_JSON.relative_to(REPO)} ({OUT_JSON.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
