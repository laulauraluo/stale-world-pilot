"""Score the agent runs; write the summary table (markdown + csv) and the figure (vector PDF + SVG).

Metrics, per model x world x procedure (n = episodes in the cell, 95% Wilson intervals):
  success  : the chosen bay holds the target crate now AND the chosen aisle is passable now
             (passability comes from Habitat's navmesh planner and is stored in episodes.jsonl)
  flagged  : the agent's final world_changed is true. Changed world: detection rate.
             Static world: false-alarm rate.
  stale    : changed world only. The action fits the 09:00 world but not the 09:20 one:
             moved -> the agent went to the crate's old bay; blocked -> it took the blocked aisle.
  looked   : memory procedure only. The agent asked for a new camera image.
  invalid  : the final reply had no valid bay or aisle (scored as a failure, not dropped).
Memory audit: how many 09:00 descriptions call an aisle blocked (no aisle is blocked at 09:00).

Usage (CPU, needs matplotlib):  python analyze.py --data data --results results [--compare results_v1]
--compare also writes <results>/compare.txt: for each model and procedure, how many final actions
(bay, aisle, world_changed) are identical between the two runs.
"""
import argparse, csv, json, math, re
from collections import defaultdict
from pathlib import Path

PROCS, WORLDS = ("memory", "recheck", "facts", "replay"), ("static", "changed")
NAMES = {"qwen3vl8b": "Qwen3-VL-8B", "mistral24b": "Mistral-Small-3.2-24B",
         "glm41v9b": "GLM-4.1V-9B-Thinking"}  # also the row order


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


BLOCKED = re.compile(r"(left|right) aisle\W{0,5}(is\s+)?blocked", re.I)  # "Right Aisle**: Blocked", "left aisle is blocked"


def memory_says_blocked(records, model):
    """(k, n): episodes whose 09:00 description calls an aisle blocked, out of all episodes."""
    first = {r["id"]: r["replies"][0] for r in records if r["model"] == model}
    return sum(bool(BLOCKED.search(d)) for d in first.values()), len(first)


def score(ep, rec):
    st, fa = ep["states"][rec["world"]], rec["final"]
    valid = fa["bay"] is not None and fa["aisle"] is not None
    out = {"success": valid and st["bays"][fa["bay"] - 1] == ep["target"] and st["clear"][fa["aisle"]],
           "flagged": fa["world_changed"] is True, "invalid": not valid, "stale": None,
           "looked": rec["looked"] if rec["proc"] == "memory" else None}
    if rec["world"] == "changed":
        old_bay = ep["states"]["static"]["bays"].index(ep["target"]) + 1
        out["stale"] = fa["bay"] == old_bay if ep["change"] == "moved" else fa["aisle"] == st["barrier"]
    return out


def selftest():
    assert [round(x, 4) for x in wilson(0, 10)] == [0.0, 0.2775]
    assert [round(x, 4) for x in wilson(5, 10)] == [0.2366, 0.7634]
    ep = {"target": "red", "change": "moved", "states": {
        "static": {"bays": ["red", None, "blue", "green"], "barrier": None, "clear": {"left": True, "right": True}},
        "changed": {"bays": [None, "red", "blue", "green"], "barrier": None, "clear": {"left": True, "right": True}}}}
    rec = lambda world, bay, aisle="left", wc=False: {"world": world, "proc": "recheck", "looked": False,
                                                      "final": {"bay": bay, "aisle": aisle, "world_changed": wc}}
    assert score(ep, rec("static", 1))["success"] and not score(ep, rec("changed", 1))["success"]
    assert score(ep, rec("changed", 1))["stale"] and not score(ep, rec("changed", 2))["stale"]
    assert score(ep, rec("changed", 2, wc=True))["flagged"] and score(ep, rec("changed", None))["invalid"]
    ep2 = {"target": "red", "change": "blocked", "states": {
        "static": {"bays": ["red", None, "blue", "green"], "barrier": None, "clear": {"left": True, "right": True}},
        "changed": {"bays": ["red", None, "blue", "green"], "barrier": "left", "clear": {"left": False, "right": True}}}}
    assert score(ep2, rec("changed", 1, "left"))["stale"] and not score(ep2, rec("changed", 1, "left"))["success"]
    assert score(ep2, rec("changed", 1, "right"))["success"] and not score(ep2, rec("changed", 1, "right"))["stale"]
    assert BLOCKED.search("- **Right Aisle**: Blocked (as indicated") and BLOCKED.search("the left aisle is blocked")
    assert not BLOCKED.search("Right aisle: clear, not blocked") and not BLOCKED.search("Left aisle: not blocked")


def summarize(eps, records):
    """{(model, world, proc, change): {metric: (k, n)}}, with change='all' plus 'moved'/'blocked'."""
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in records:
        ep = eps[r["id"]]
        for change in ("all", ep["change"]):
            for m, v in score(ep, r).items():
                if v is not None:
                    cell = cells[(r["model"], r["world"], r["proc"], change)][m]
                    cell[0] += int(v)
                    cell[1] += 1
    return cells


def fmt(kn):
    k, n = kn
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"  # n=200: rates are multiples of 0.5%


def compare(old_dir, records):
    """Lines 'model proc: k/n identical' over the cases present in both runs."""
    old = {(r["model"], r["id"], r["world"], r["proc"]): r["final"]
           for f in sorted(old_dir.glob("*.jsonl")) for r in map(json.loads, open(f))}
    same = defaultdict(lambda: [0, 0])
    for r in records:
        key = (r["model"], r["id"], r["world"], r["proc"])
        if key in old:
            same[key[0], key[3]][0] += old[key] == r["final"]
            same[key[0], key[3]][1] += 1
    return [f"{NAMES[m]} {p}: {k}/{n} identical" for (m, p), (k, n) in sorted(same.items())]


def main():
    selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--compare", type=Path, default=None, help="an earlier results dir to compare with")
    a = ap.parse_args()
    eps = {e["id"]: e for e in map(json.loads, open(a.data / "episodes.jsonl"))}
    records = [json.loads(l) for f in sorted(a.results.glob("*.jsonl")) for l in open(f)]
    cells = summarize(eps, records)
    models = [m for m in NAMES if any(r["model"] == m for r in records)]
    procs = [p for p in PROCS if any(r["proc"] == p for r in records)]  # older runs have no "facts"

    with open(a.results / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "world", "procedure", "change", "metric", "k", "n", "rate", "ci_lo", "ci_hi"])
        for (model, world, proc, change), ms in sorted(cells.items()):
            for m, (k, n) in sorted(ms.items()):
                w.writerow([model, world, proc, change, m, k, n, f"{k / n:.4f}", *(f"{x:.4f}" for x in wilson(k, n))])

    cols = [("success, static", "static", "success"), ("success, changed", "changed", "success"),
            ("stale action, changed", "changed", "stale"), ("detected, changed", "changed", "flagged"),
            ("false alarm, static", "static", "flagged"), ("asked to look", "changed", "looked")]
    lines = ["| model | procedure | " + " | ".join(c[0] for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for model in models:
        for proc in procs:
            row = []
            for _, world, m in cols:
                kn = cells[(model, world, proc, "all")].get(m)
                row.append(fmt(kn) if kn else "n/a")
            lines.append(f"| {NAMES[model]} | {proc} | " + " | ".join(row) + " |")
    n_cell = cells[(models[0], "changed", "memory", "all")]["success"][1]
    lines.append(f"\nEach cell: rate over n = {n_cell} episodes, with a 95% Wilson interval in brackets.")
    audit = ", ".join(f"{NAMES[m]} {k}/{n}" for m in models for k, n in [memory_says_blocked(records, m)])
    lines.append(f"09:00 descriptions that call an aisle blocked (none is at 09:00): {audit}.")
    no_answer = ", ".join(f"{NAMES[m]} {sum(r['final']['bay'] is None or r['final']['aisle'] is None for r in records if r['model'] == m)}"
                          f"/{sum(r['model'] == m for r in records)}" for m in models)
    lines.append(f"Final replies with no valid bay or aisle (scored as failures): {no_answer}.")
    (a.results / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(cells, models, procs, a.results / "fig_results")
    if a.compare:
        lines = compare(a.compare, records)
        (a.results / "compare.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


def plot(cells, models, procs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    green, red, blue, red_light = "#8BCF8B", "#B64342", "#0F4D92", "#E9A6A1"   # figures4papers palette
    panels = [("task success", [("static", "success", "no change", green), ("changed", "success", "world changed", red)]),
              ("changed world", [("changed", "stale", "acted on old state", red_light),
                                 ("changed", "flagged", "reported a change", blue)])]
    fig, axes = plt.subplots(len(models), 2, figsize=(7.0, 2.4 * len(models)), squeeze=False, sharey=True)
    for i, model in enumerate(models):
        for j, (title, series) in enumerate(panels):
            ax = axes[i][j]
            width = 0.8 / len(series)
            for s, (world, m, label, color) in enumerate(series):
                ks = [cells[(model, world, p, "all")][m] for p in procs]
                rates = [k / n for k, n in ks]
                cis = [wilson(k, n) for k, n in ks]
                xs = [x + (s - (len(series) - 1) / 2) * width for x in range(len(procs))]
                err = [[r - lo for r, (lo, _) in zip(rates, cis)], [hi - r for r, (_, hi) in zip(rates, cis)]]
                ax.bar(xs, rates, width, yerr=err, capsize=2, label=label, color=color,
                       edgecolor="black", linewidth=0.6, error_kw={"linewidth": 0.8})
            ax.set_xticks(range(len(procs)))
            ax.set_xticklabels(procs)  # two calls: also works on matplotlib < 3.5
            ax.set_ylim(0, 1.25)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_title(f"{NAMES[model]}: {title}", fontsize=9)
            if j == 0:
                ax.set_ylabel("rate (95% CI)")
            if i == 0:
                ax.legend(fontsize=7, loc="upper center", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "svg"):  # both vector; svg shows inline in markdown
        fig.savefig(path.with_suffix("." + ext))
    print(f"wrote {path}.pdf/.svg")


if __name__ == "__main__":
    main()
