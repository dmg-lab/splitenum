#!/usr/bin/env python3
"""Diff found degree-8 types against Orevkov's 89-scheme M-curve table.

Usage: orevkov_diff.py TABLE.md < types.csv

TABLE.md is a Markdown table of Orevkov's 89 real schemes of degree-8
M-curves, one scheme per cell in angle-bracket notation such as
⟨17⊔1⟨2⊔1⟨1⟩⟩⟩, bold if already realized as a T-curve, with a trailing *
if not known to be realized by any real algebraic curve.  Stdin is a types
CSV with a `viro` first column, types.csv of a finished run.

Buckets every found scheme as: BOLD (already realized as a T-curve per the
paper), PLAIN (known classically, new as a T-curve), STAR (not known to be
realized by any real algebraic curve), or OUTSIDE (not in the table at
all, which indicates a bug).
"""

import csv
import re
import sys


def md_to_viro(cell):
    """⟨17⊔1⟨2⊔1⟨1⟩⟩⟩ -> <17v1<2v1<1>>>"""
    s = cell.strip().replace("**", "").rstrip("\\*").strip()
    s = s.replace("⟨", "<").replace("⟩", ">")
    s = s.replace("⊔", "v")
    return s.strip()


def parse_table(path):
    schemes = {}
    in_table = False
    for line in open(path, encoding="utf-8"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r"-*|:?-+:?", c) for c in cells):
                in_table = True
                continue
            if not in_table:
                continue
            for c in cells:
                if "⟨" not in c:
                    continue
                kind = ("bold" if c.startswith("**") else
                        "star" if c.rstrip().endswith("*") else "plain")
                schemes[md_to_viro(c)] = kind
        elif in_table and line.strip():
            in_table = False
    return schemes


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip())
    table = parse_table(sys.argv[1])
    if len(table) != 89:
        print(f"warning: parsed {len(table)} schemes from the table, "
              "expected 89", file=sys.stderr)

    found = {}
    for row in csv.DictReader(sys.stdin):
        count = int(row.get("count_so_far") or row.get("count"))
        found[row["viro"]] = found.get(row["viro"], 0) + count

    buckets = {"bold": [], "plain": [], "star": [], "outside": []}
    for viro, count in found.items():
        buckets[table.get(viro, "outside")].append((viro, count))

    def show(name, label):
        rows = buckets[name]
        print(f"== {label}: {len(rows)}")
        for viro, count in sorted(rows):
            print(f"   {viro}  (count {count})")

    show("bold", "found, already known as T-curves (bold)")
    show("plain", "found, NEW as T-curves (known classically, plain)")
    show("star", "found, not known classically at all (*) -- verify!")
    show("outside", "found, NOT IN THE 89-TABLE -- investigate")
    nbold = sum(1 for k in table.values() if k == "bold")
    print(f"== missing: {len(table) - len(found) + len(buckets['outside'])} "
          f"of 89 not (yet) found; the paper's own T-curve count is {nbold}")
    todo = sorted(v for v, k in table.items()
                  if k != "bold" and v not in found)
    print(f"== still-open non-bold schemes not yet found: {len(todo)}")
    for v in todo:
        print(f"   {v}  ({table[v]})")


if __name__ == "__main__":
    main()
