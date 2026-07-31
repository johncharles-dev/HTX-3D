"""Analyse the blinded manual evaluation.

Reads blind_key.csv + every scores_*.csv (excluding the template), un-blinds the
slots, and writes manual_eval_report.md containing:

  1. coverage / data-validity check
  2. per-pipeline mean geometry & texture scores with paired bootstrap 95% CIs
  3. paired Wilcoxon signed-rank tests between every pipeline pair
  4. defect frequency table (the most citable output)
  5. inter-rater agreement, if more than one rater scored

Usage:
    python analyze_manual.py
    python analyze_manual.py --boot 50000
"""

import argparse
import csv
import itertools
import random
import statistics as st
import warnings
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KEY_PATH = SCRIPT_DIR / "blind_key.csv"
REPORT_PATH = SCRIPT_DIR / "manual_eval_report.md"

PIPELINES = [
    "trellis2_rembg", "trellis2_sam3", "trellis_rembg", "trellis_sam3",
    "hunyuan_rembg", "hunyuan_sam3", "sam3d_sam3",
]
DEFECTS = ["floaters", "holes", "flat_collapse", "janus_duplicate", "front_only_texture"]
TRUEY = {"1", "x", "y", "yes", "true", "t"}


def parse_score(v: str):
    v = (v or "").strip()
    if not v:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if 1.0 <= f <= 5.0 else None


def parse_flag(v: str) -> bool:
    return (v or "").strip().lower() in TRUEY


def load_key() -> dict[tuple[str, str], str]:
    if not KEY_PATH.exists():
        raise SystemExit(f"missing {KEY_PATH} — run make_blind_montages.py first")
    with open(KEY_PATH) as f:
        return {(r["object_id"], r["slot"]): r["pipeline"] for r in csv.DictReader(f)}


def load_scores(key, pattern: str = "scores_*.csv") -> tuple[list[dict], list[str]]:
    files = sorted(p for p in SCRIPT_DIR.glob(pattern)
                   if p.stem != "scores_TEMPLATE")
    if not files:
        raise SystemExit(f"no {pattern} found (copy scores_TEMPLATE.csv first)")
    recs, warnings = [], []
    for p in files:
        rater = p.stem.replace("scores_", "")
        with open(p) as f:
            for i, row in enumerate(csv.DictReader(f), start=2):
                oid, slot = row.get("object_id", ""), row.get("slot", "")
                pipe = key.get((oid, slot))
                if pipe is None:
                    warnings.append(f"{p.name}:{i} unknown (object={oid}, slot={slot})")
                    continue
                g = parse_score(row.get("geometry_1to5", ""))
                t = parse_score(row.get("texture_1to5", ""))
                if g is None and t is None:
                    continue  # unscored row — silently skipped, counted in coverage
                if g is None or t is None:
                    warnings.append(f"{p.name}:{i} {oid}/{slot} partial score "
                                    f"(geom={row.get('geometry_1to5')!r} "
                                    f"tex={row.get('texture_1to5')!r})")
                recs.append({
                    "rater": rater, "object_id": oid, "slot": slot, "pipeline": pipe,
                    "geometry": g, "texture": t,
                    "combined": None if (g is None or t is None) else (g + t) / 2.0,
                    **{d: parse_flag(row.get(d, "")) for d in DEFECTS},
                })
    return recs, warnings


def per_object_mean(recs, field) -> dict[tuple[str, str], float]:
    """(object, pipeline) -> mean over raters, ignoring blanks."""
    acc = defaultdict(list)
    for r in recs:
        if r[field] is not None:
            acc[(r["object_id"], r["pipeline"])].append(r[field])
    return {k: st.mean(v) for k, v in acc.items()}


def bootstrap_ci(pipe, table, objs, rng, n):
    pool = [o for o in objs if (o, pipe) in table]
    if len(pool) < 2:
        return None, None
    means = []
    for _ in range(n):
        s = [rng.choice(pool) for _ in pool]
        means.append(st.mean(table[(o, pipe)] for o in s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 3:
        return None
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--pattern", default="scores_*.csv",
                    help="which score sheets to analyse; use 'vlm_scores_*.csv' "
                         "to analyse the model-based pass SEPARATELY. Human and "
                         "VLM scores must never be pooled.")
    ap.add_argument("--out", default=None, help="output .md path")
    ap.add_argument("--key", default=None,
                    help="blinding key CSV (default: blind_key.csv; use "
                         "blind_key_b2.csv for the batch-2 set)")
    ap.add_argument("--slots", default=None,
                    help="comma-separated slot letters (default: derived from the key)")
    args = ap.parse_args()

    rng = random.Random(0)
    global KEY_PATH, SLOTS
    if args.key:
        KEY_PATH = Path(args.key)
    key = load_key()
    # slots and pipelines come from whichever key was loaded, so a 9-pipeline set
    # analyses identically to the 7-pipeline one
    SLOTS = (args.slots.split(",") if args.slots
             else sorted({s for (_, s) in key}))
    global PIPELINES
    PIPELINES = sorted(set(key.values()))
    recs, warnings = load_scores(key, args.pattern)
    is_vlm = "vlm" in args.pattern
    out_path = Path(args.out) if args.out else (
        SCRIPT_DIR / ("vlm_judge_report.md" if is_vlm else "manual_eval_report.md"))
    raters = sorted({r["rater"] for r in recs})
    objs = sorted({r["object_id"] for r in recs})
    pipes = [p for p in PIPELINES if any(r["pipeline"] == p for r in recs)]

    if is_vlm:
        L = ["# VLM-as-judge evaluation — blinded montage scoring\n"]
        L.append("> ⚠ **These scores were produced by a vision-language model, "
                 "not by human raters.** They are NOT the manual evaluation and "
                 "must not be pooled with, or presented as, human judgements. "
                 "They are unvalidated against a human subset, so treat them as "
                 "an automated screen, not as perceptual ground truth.\n")
    else:
        L = ["# Manual (human) evaluation — blinded montage scoring\n"]
    L.append(f"{'Judges' if is_vlm else 'Raters'}: **{len(raters)}** "
             f"({', '.join(raters)}) · objects scored: "
             f"**{len(objs)}/{len({o for (o, _) in key})}** · "
             f"pipelines: **{len(pipes)}**\n")

    # ---- 1. coverage -------------------------------------------------------
    expected = len(objs) * len(pipes) * len(raters)
    got = sum(1 for r in recs if r["combined"] is not None)
    L.append("## 1 · Coverage & validity\n")
    L.append(f"- Complete (geometry+texture) judgements: **{got} / {expected}** "
             f"({100*got/expected:.0f}%)\n" if expected else "")
    if warnings:
        L.append(f"- ⚠ {len(warnings)} data warnings:\n")
        for w in warnings[:20]:
            L.append(f"  - {w}")
        if len(warnings) > 20:
            L.append(f"  - …and {len(warnings)-20} more")
        L.append("")
    else:
        L.append("- No malformed rows.\n")

    # ---- 2. scores ---------------------------------------------------------
    tables = {f: per_object_mean(recs, f) for f in ("geometry", "texture", "combined")}
    L.append("## 2 · Mean scores (1–5, higher is better)\n")
    L.append("Paired bootstrap 95% CI, resampling objects.\n")
    L.append("| Pipeline | Geometry | Texture | Combined | Combined 95% CI |")
    L.append("|---|---:|---:|---:|:---:|")
    rows = []
    for p in pipes:
        vals = {}
        for f in ("geometry", "texture", "combined"):
            v = [tables[f][(o, p)] for o in objs if (o, p) in tables[f]]
            vals[f] = st.mean(v) if v else float("nan")
        lo, hi = bootstrap_ci(p, tables["combined"], objs, rng, args.boot)
        rows.append((vals["combined"], p, vals, lo, hi))
    for _, p, vals, lo, hi in sorted(rows, reverse=True):
        ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "—"
        L.append(f"| {p} | {vals['geometry']:.2f} | {vals['texture']:.2f} | "
                 f"**{vals['combined']:.2f}** | {ci} |")
    L.append("")

    # ---- 3. significance ---------------------------------------------------
    L.append("## 3 · Paired Wilcoxon signed-rank (combined score, paired by object)\n")
    try:
        from scipy.stats import wilcoxon
        comb = tables["combined"]
        sig, near = [], []
        for a, b in itertools.combinations(pipes, 2):
            pair = [(comb[(o, a)], comb[(o, b)]) for o in objs
                    if (o, a) in comb and (o, b) in comb]
            if len(pair) < 6:
                continue
            d = [x - y for x, y in pair]
            if all(abs(v) < 1e-12 for v in d):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, pv = wilcoxon(d)
            except Exception:
                continue
            entry = (pv, a, b, st.mean(d), len(pair))
            (sig if pv < 0.05 else near).append(entry)
        if sig:
            L.append("| Pair | mean diff | n | p |")
            L.append("|---|---:|---:|---:|")
            for pv, a, b, md, n in sorted(sig):
                L.append(f"| {a} vs {b} | {md:+.2f} | {n} | **{pv:.3f}** |")
            L.append("")
        else:
            L.append("No pipeline pair separates at p<0.05 — differences are "
                     "within noise at this sample size.\n")
        if near:
            L.append("<details><summary>Non-significant pairs (p ≥ 0.05)</summary>\n")
            L.append("| Pair | mean diff | n | p |")
            L.append("|---|---:|---:|---:|")
            for pv, a, b, md, n in sorted(near):
                L.append(f"| {a} vs {b} | {md:+.2f} | {n} | {pv:.3f} |")
            L.append("\n</details>\n")
    except ImportError:
        L.append("_scipy unavailable — install scipy for significance tests._\n")

    # ---- 4. defects --------------------------------------------------------
    L.append("## 4 · Defect frequency (count of objects flagged, per pipeline)\n")
    L.append("Counted once per (object, pipeline): flagged if any rater flagged it.\n")
    flagged = defaultdict(set)
    for r in recs:
        for d in DEFECTS:
            if r[d]:
                flagged[(r["pipeline"], d)].add(r["object_id"])
    L.append("| Pipeline | " + " | ".join(d.replace("_", " ") for d in DEFECTS) + " | any |")
    L.append("|---|" + "---:|" * (len(DEFECTS) + 1))
    for p in pipes:
        cells, anyset = [], set()
        for d in DEFECTS:
            s = flagged.get((p, d), set())
            anyset |= s
            cells.append(f"{len(s)}/{len(objs)}")
        L.append(f"| {p} | " + " | ".join(cells) + f" | **{len(anyset)}/{len(objs)}** |")
    L.append("")

    # ---- 5. agreement ------------------------------------------------------
    L.append("## 5 · Inter-rater agreement\n")
    if len(raters) < 2:
        L.append("_Single rater — this is a single-observer assessment. "
                 "Describe it as such; no agreement statistic is possible._\n")
    else:
        cells = sorted({(r["object_id"], r["pipeline"]) for r in recs})
        byr = {}
        for rt in raters:
            m = {(r["object_id"], r["pipeline"]): r["combined"]
                 for r in recs if r["rater"] == rt and r["combined"] is not None}
            byr[rt] = m
        cors = []
        for a, b in itertools.combinations(raters, 2):
            common = [c for c in cells if c in byr[a] and c in byr[b]]
            if len(common) < 3:
                continue
            rho = spearman([byr[a][c] for c in common], [byr[b][c] for c in common])
            if rho is not None:
                cors.append((a, b, rho, len(common)))
        if cors:
            L.append("| Rater pair | Spearman ρ | n |")
            L.append("|---|---:|---:|")
            for a, b, rho, n in cors:
                L.append(f"| {a} ↔ {b} | {rho:.2f} | {n} |")
            L.append(f"\nMean pairwise ρ = **{st.mean(c[2] for c in cors):.2f}** "
                     f"(>0.6 is respectable for subjective 3D quality).\n")
        else:
            L.append("_Not enough overlapping judgements to compute agreement._\n")

    out_path.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n✓ written to {out_path}")


if __name__ == "__main__":
    main()
