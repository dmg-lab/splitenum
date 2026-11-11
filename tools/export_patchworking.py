#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Export one Harnack-split witness per M-scheme into a CombinatorialPatchworking degree folder.

usage: export_patchworking.py RUN_DIR WITNESS_PCOM_DIR TARGET_DEG_DIR [--orevkov TABLE.md]
                              [--pcomcheck BIN] [--flips K]

RUN_DIR holds run.json and types.csv of a finished splitenum run, WITNESS_PCOM_DIR is the
`-pcom` directory `bin/witnesses` wrote for it, TARGET_DEG_DIR is e.g.
CombinatorialPatchworking/deg8. The Orevkov table lists the degree-8 schemes not realized.
With --pcomcheck, a nonregular witness triangulation is replaced by a regular one reached by
at most K flips of edges that lie on no split, provided bin/pcomcheck confirms its scheme.
"""

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, milp

REPO = "https://github.com/dmg-lab/CombinatorialPatchworking/blob/main"
RAW = "https://raw.githubusercontent.com/dmg-lab/CombinatorialPatchworking/refs/heads/main"
VIZ = "https://patchworking.polymake.org#"


def flat_points(delta):
    return [(x, y) for y in range(delta + 1) for x in range(delta + 1 - y)]


def det(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def barycentric(pts, tri, q):
    a, b, c = (pts[i] for i in tri)
    d = det(a, b, c)
    la, lb, lc = det(q, b, c), det(a, q, c), det(a, b, q)
    assert la % d == 0 and lb % d == 0 and lc % d == 0
    return la // d, lb // d, lc // d


def check_triangulation(pts, cells, delta):
    assert len(cells) == delta * delta, "cell count"
    for t in cells:
        assert abs(det(*(pts[i] for i in t))) == 1, f"cell {t} not unimodular"
    edges = defaultdict(list)
    for t in cells:
        for k in range(3):
            u, v = sorted((t[k], t[(k + 1) % 3]))
            edges[(u, v)].append(t[(k + 2) % 3])
    assert all(len(o) in (1, 2) for o in edges.values()), "edge in more than two cells"
    return edges


def min_weights(pts, cells, edges):
    n = len(pts)
    rows = []
    for (u, v), opp in edges.items():
        if len(opp) != 2:
            continue
        a, b = opp
        lu, lv, la = barycentric(pts, (u, v, a), pts[b])
        r = np.zeros(n)
        r[b] += 1
        r[u] -= lu
        r[v] -= lv
        r[a] -= la
        rows.append(r)
    res = milp(np.ones(n), constraints=LinearConstraint(np.array(rows), lb=1, ub=np.inf),
               integrality=np.ones(n), bounds=(0, np.inf))
    if res.status != 0:
        return None
    w = [int(round(v)) for v in res.x]
    lifted_ok(pts, cells, w)
    return w


def split_edges(pts, splits):
    idx = {p: i for i, p in enumerate(pts)}
    out = set()
    for s in splits.split():
        path = [tuple(map(int, q.strip("()").split(","))) for q in s.split("-")]
        for a, b in zip(path, path[1:]):
            out.add(tuple(sorted((idx[a], idx[b]))))
    return out


def flips(pts, cells, delta, fixed):
    cs = {tuple(sorted(t)) for t in cells}
    for (u, v), opp in check_triangulation(pts, cells, delta).items():
        if len(opp) != 2 or (u, v) in fixed:
            continue
        a, b = opp
        du, dv = det(pts[a], pts[b], pts[u]), det(pts[a], pts[b], pts[v])
        if du * dv >= 0 or abs(du) != 1 or abs(dv) != 1:
            continue
        nc = cs - {tuple(sorted((u, v, a))), tuple(sorted((u, v, b)))}
        nc |= {tuple(sorted((a, b, u))), tuple(sorted((a, b, v)))}
        yield (u, v), sorted(nc)


def scheme(pcomcheck, delta, signs, cells):
    inp = "".join("1" if s else "0" for s in signs) + "\n" + "\n".join(" ".join(map(str, c)) for c in cells) + "\n"
    out = subprocess.run([pcomcheck, str(delta)], input=inp, capture_output=True, text=True, check=True).stdout.split()
    return out[1], out[7] == "1"


def regular_by_flips(pts, cells, delta, signs, viro, fixed, pcomcheck, depth):
    seen = {tuple(map(tuple, sorted(tuple(sorted(t)) for t in cells)))}
    queue = deque([(cells, [])])
    while queue:
        cur, path = queue.popleft()
        if len(path) == depth:
            continue
        for e, nc in flips(pts, cur, delta, fixed):
            key = tuple(nc)
            if key in seen:
                continue
            seen.add(key)
            w = min_weights(pts, nc, check_triangulation(pts, nc, delta))
            if w is not None and scheme(pcomcheck, delta, signs, nc) == (viro, True):
                return [list(t) for t in nc], w, [(pts[e[0]], pts[e[1]]) for e in path + [e]]
            queue.append((nc, path + [e]))
    return None


def lifted_ok(pts, cells, w):
    for t in cells:
        for q in range(len(pts)):
            if q in t:
                continue
            l = barycentric(pts, t, pts[q])
            assert w[q] > sum(li * w[i] for li, i in zip(l, t)), f"weights do not induce cell {t}"


def viro_to_file(viro):
    return viro.replace("<", "(").replace(">", ")") + ".pcom"


def viro_to_unicode(viro):
    return viro.replace("<", "⟨").replace(">", "⟩").replace("v", "∪")


def url_path(path):
    return path.replace("(", "%28").replace(")", "%29")


def dumps(obj):
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s*\n\s*", " ", m.group(1)) + "]", text)


def grouped(v):
    return f"{v:,}"


def md_to_viro(cell):
    s = cell.strip().replace("**", "").rstrip("\\*").strip()
    return s.replace("⟨", "<").replace("⟩", ">").replace("⊔", "v").strip()


def orevkov(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.startswith("| ") or "⟨" not in line:
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        for col, c in zip(["(19,3)", "(15,7)", "(11,11)", "(7,15)", "(3,19)"], cols):
            if "⟨" in c:
                out[md_to_viro(c)] = (col, c.rstrip().endswith("*"))
    assert len(out) == 89, len(out)
    return out


def main():
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("run")
    ap.add_argument("wdir")
    ap.add_argument("target")
    ap.add_argument("--orevkov")
    ap.add_argument("--pcomcheck")
    ap.add_argument("--flips", type=int, default=2)
    args = ap.parse_args()
    run, wdir, target = Path(args.run), Path(args.wdir), Path(args.target)
    meta = json.loads((run / "run.json").read_text())
    delta = meta["delta"]
    pts = flat_points(delta)
    rows = list(csv.DictReader(open(run / "types.csv")))
    total = defaultdict(int)
    for r in rows:
        total[r["viro"]] += int(r["count"])
    assert sum(total.values()) == meta["total_nodes"]

    best = {}
    for r in rows:
        ids = [i for i in r["first_witness"].split(";") if i]
        key = (int(r["support_degree"]), len(ids), r["slack"])
        if r["viro"] not in best or key < best[r["viro"]][0]:
            best[r["viro"]] = (key, r, ids)

    spl = target / "spl"
    spl.mkdir(parents=True, exist_ok=True)
    out = []
    for viro, (_, r, ids) in best.items():
        src = wdir / f"{r['p']}_{r['n']}_{viro.replace('<', '_').replace('>', '_')}_s{r['slack'].replace(';', '')}.pcom"
        w = json.loads(src.read_text())
        cells = [list(t) for t in w["DUAL_SUBDIVISION"]["MAXIMAL_CELLS"]]
        signs = w["PATCHWORK"][0]["SIGNS"]
        assert len(signs) == len(pts)
        edges = check_triangulation(pts, cells, delta)
        weights = min_weights(pts, cells, edges)
        splits = w["ORIGIN"].split(": ", 1)[1]
        flipped = []
        if weights is None and args.pcomcheck:
            assert scheme(args.pcomcheck, delta, signs, cells) == (viro, True)
            found = regular_by_flips(pts, cells, delta, signs, viro, split_edges(pts, splits), args.pcomcheck,
                                     args.flips)
            if found:
                cells, weights, flipped = found
        name = viro_to_file(viro)
        sub = {"POINTS": [["1", str(y), str(x), str(delta - x - y)] for x, y in pts]}
        if weights is not None:
            sub["MIN_WEIGHTS"] = weights
        sub["MAXIMAL_CELLS"] = cells
        doc = {
            "_ns": {"polymake": ["https://polymake.org", "4.13"]},
            "_type": "tropical::Hypersurface<Min>",
            "_id": f"deg{delta}_{name[:-5]}",
            "_attrs": {"ORIGIN": {"attachment": True}},
            "ORIGIN": f"splitenum, exhaustive enumeration of Harnack splits of degree {delta}; "
                      f"splits: {splits}"
                      + ("" if not flipped else
                         "; triangulation: enumeration refinement with the edges "
                         + " ".join(f"{a}-{b}".replace(" ", "") for a, b in flipped) + " flipped"),
            "PURE": True,
            "DUAL_SUBDIVISION": sub,
            "PATCHWORK": [{"SIGNS": signs, "TYPE": viro}],
        }
        (spl / name).write_text(dumps(doc) + "\n")
        out.append({
            "viro": viro, "p": int(r["p"]), "n": int(r["n"]), "nodes": total[viro],
            "share": total[viro] / meta["total_nodes"], "support_degree": int(r["support_degree"]),
            "nsplits": len(ids), "regular": weights is not None, "file": f"spl/{name}", "flipped": bool(flipped),
        })

    out.sort(key=lambda o: (-o["p"], [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", o["viro"])]))
    missing = None
    if args.orevkov:
        table = orevkov(args.orevkov)
        assert all(o["viro"] in table for o in out)
        missing = [(v, c, s) for v, (c, s) in table.items() if v not in best]
    (target / "README.md").write_text(readme(delta, meta, out, missing))
    print(f"degree {delta}: {len(out)} schemes, {sum(not o['regular'] for o in out)} nonregular, "
          f"{sum(bool(o['flipped']) for o in out)} made regular by flips")


def row(delta, o):
    path = f"deg{delta}/{o['file']}"
    link = f"[Link]({url_path(o['file'])}) / [Visual]({VIZ}{RAW}/{url_path(path)})"
    support = "–" if o["nsplits"] == 0 else str(o["support_degree"])
    return (f"| {viro_to_unicode(o['viro'])} | ({o['p']},{o['n']}) | {o['nsplits']} | {support} | "
            f"{grouped(o['nodes'])} | {100 * o['share']:.3g}% | {link} |")


HEAD = ("| type | (p,n) | splits | support | nodes | share | pcom |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | --- |")


def readme(delta, meta, out, missing):
    assert all(o["regular"] for o in out), "a nonregular witness needs a regularity column"
    lines = [
        f"# Degree {delta}: maximal T-curves from Harnack splits",
        "",
        "| | |",
        "| --- | --- |",
        f"| Harnack splits | {grouped(meta['nsplits'])} |",
        f"| compatible pairs | {grouped(meta['compat_pairs'])} |",
        f"| compatible collections | {grouped(meta['total_nodes'])} |",
        f"| classified up to the S₃ symmetry | {grouped(meta['classified_nodes'])} |",
        f"| largest collection | {meta['max_depth_reached']} splits |",
        f"| real schemes | {len(out)} |",
        "",
        "## Columns",
        "",
        "- **(p,n)**: the numbers of positive and negative regions.",
        "- **splits**: number of splits in the witness collection; 0 is the Harnack curve itself.",
        "- **support**: the smallest degree whose simplex already contains the witness collection after "
        "translation along the boundary.",
        "- **nodes** and **share**: how many compatible collections produce the scheme.",
        "",
        "Every witness triangulation is regular and carries `MIN_WEIGHTS`.",
        "The split coordinates are stored in the `ORIGIN` field of each `.pcom`,",
        "with A–B for a simple split and A–apex–B for a double split.",
        "",
        "## Real schemes",
        "",
    ]
    if delta % 2 == 0:
        lines += [HEAD] + [row(delta, o) for o in out] + [""]
    else:
        groups = defaultdict(list)
        for o in out:
            groups[(o["p"], o["n"])].append(o)
        keys = sorted(groups, key=lambda k: -k[0])
        lines += [f"Every scheme consists of the pseudoline J and {(delta - 1) * (delta - 2) // 2} ovals. "
                  "The schemes are grouped by (p,n).", ""]
        for p, n in keys:
            g = groups[(p, n)]
            nodes = sum(o["nodes"] for o in g)
            lines += ["<details>",
                      f"<summary><b>(p,n) = ({p},{n})</b>: {len(g)} scheme{'' if len(g) == 1 else 's'}, "
                      f"{100 * nodes / meta['total_nodes']:.3g}% of all collections</summary>",
                      "", HEAD] + [row(delta, o) for o in g] + ["", "</details>", ""]
    if missing is not None:
        lines += [
            "## Schemes not realized by Harnack splits",
            "",
            "| (p,n) | real schemes |",
            "| --- | --- |",
        ]
        bycol = defaultdict(list)
        for v, c, s in missing:
            bycol[c].append(viro_to_unicode(v) + ("\\*" if s else ""))
        for c in ["(19,3)", "(15,7)", "(11,11)", "(7,15)", "(3,19)"]:
            if bycol[c]:
                lines.append(f"| {c} | {', '.join(bycol[c])} |")
        lines += [
            "",
            f"The certificates in [deg8.pcoms.txz](../deg8.pcoms.txz), found by a different search, "
            f"realize the same {len(out)} maximal schemes.",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
