"""Reproducible figures for the project brief. Run with PYTHONPATH=../../src."""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from expplatform.demo import demo_config, demo_frame
from expplatform.report import build_readout
from expplatform.sequential import msprt_path

ACC, GREY, ORG, RED = "#1f4e9e", "#8a8f98", "#e69f00", "#c0392b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
R = json.load(open(HERE.parent.parent / "examples" / "results.json"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#eef3fb", ec=ACC, bold=False, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color="#444", lw=1.1))


# 1 architecture
fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")
box(ax, 0.1, 3.7, 1.6, 0.9, "design.py\npower / sample\nsize / MDE")
box(ax, 0.1, 2.3, 1.6, 0.9, "assignment.py\nSHA-256 buckets\nlayers")
box(ax, 0.1, 0.9, 1.6, 0.9, "simulator.py\nground-truth\ndata", fc="#fff4de", ec=ORG)
box(ax, 2.5, 2.3, 1.7, 0.9, "Events\n(CSV / parquet,\nunit-level rows)", fc="white", ec=GREY)
box(ax, 4.85, 3.7, 2.1, 0.9, "assignment.py audit\nSRM chi-square,\nhash re-check")
box(ax, 4.85, 2.3, 2.1, 0.9, "analysis.py\nWelch, delta method,\nCUPED/CUPAC, Holm/BH", fs=7)
box(ax, 4.85, 0.9, 2.1, 0.9, "sequential.py\nmSPRT CS, OBF bounds\ncate.py: segment effects", fs=7)
box(ax, 7.4, 2.3, 1.3, 0.9, "report.py\nbuild_readout\nJSON", bold=True, fs=7.5)
box(ax, 9.0, 3.5, 0.95, 0.8, "FastAPI\nservice", fc="#e6f2e6", ec="#2e7d32", fs=7.5)
box(ax, 9.0, 1.2, 0.95, 0.8, "Streamlit\ndashboard", fc="#e6f2e6", ec="#2e7d32", fs=7.5)
arrow(ax, (1.7, 2.75), (2.5, 2.75))
arrow(ax, (1.7, 1.35), (2.5, 2.4))
arrow(ax, (4.2, 2.9), (4.85, 4.0))
arrow(ax, (4.2, 2.75), (4.85, 2.75))
arrow(ax, (4.2, 2.6), (4.85, 1.4))
arrow(ax, (6.95, 4.1), (7.4, 3.1))
arrow(ax, (6.95, 2.75), (7.4, 2.75))
arrow(ax, (6.95, 1.4), (7.4, 2.4))
arrow(ax, (8.7, 3.0), (9.0, 3.8))
arrow(ax, (8.7, 2.5), (9.0, 1.7))
ax.text(5.0, 0.25, "Statistical claims are checked against the simulator's known ground truth in 55 tests.",
        ha="center", fontsize=8, style="italic", color="#444")
save(fig, "architecture.png")

# 2 peeking
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw={"width_ratios": [1, 1.25]})
nk = R["peeking"]["null_known_variance"]
labs = ["Fixed\n(1 look)", "Naive peeking\n(50 looks)", "mSPRT", "O'Brien-\nFleming (5)"]
vals = [nk["fixed"], nk["naive_peeking"], nk["msprt"], nk["obf"]]
bars = a.bar(labs, vals, color=[GREY, RED, ACC, ACC])
a.axhline(0.05, ls="--", c="k", lw=0.8)
a.text(2.2, 0.20, "dashed line = nominal 5%", ha="center", fontsize=7)
for r_, v in zip(bars, vals):
    a.text(r_.get_x() + r_.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
a.set_ylabel("False-positive rate (no true effect)")
a.set_ylim(0, 0.37)
a.tick_params(axis="x", labelsize=7)
a.set_title("20,000 simulated A/A tests", fontsize=9)
K, m = 50, 100
ns = m * np.arange(1, K + 1)
v = 2.0 / ns
for seed in range(2000):
    rng = np.random.default_rng(seed)
    d = np.cumsum(rng.normal(0, math.sqrt(2.0 / m), K)) / np.arange(1, K + 1)
    z = d / np.sqrt(v)
    if (np.abs(z) > 1.96).any() and not (np.abs(z) > 3.0).any() and np.argmax(np.abs(z) > 1.96) > 5:
        break
path = msprt_path(d, v, 0.2 ** 2)
b.fill_between(ns, path.cs_low, path.cs_high, color=ACC, alpha=0.15, label="mSPRT confidence sequence")
b.plot(ns, 1.96 * np.sqrt(v), c=RED, ls="--", lw=1, label="naive 95% band")
b.plot(ns, -1.96 * np.sqrt(v), c=RED, ls="--", lw=1)
b.plot(ns, d, c="k", lw=1.2, label="estimated effect (true = 0)")
b.axhline(0, c=GREY, lw=0.6)
hit = int(np.argmax(np.abs(d) > 1.96 * np.sqrt(v)))
b.scatter([ns[hit]], [d[hit]], c=RED, zorder=5, s=25)
b.annotate("naive test 'wins' here", (ns[hit], d[hit]), (ns[hit] - 1500, d[hit] + 0.2), fontsize=7,
           arrowprops=dict(arrowstyle="-", lw=0.6))
b.set_xlabel("units per arm")
b.set_ylabel("effect estimate")
b.legend(fontsize=6.5, frameon=False)
b.set_title(f"One illustrative null experiment (seed {seed})", fontsize=9)
fig.tight_layout()
save(fig, "peeking.png")

# 3 readout: sequential + segments
out = build_readout(demo_frame(), demo_config())
s = out["sequential"]
n = np.array(s["n"])
fig, (a, b, c) = plt.subplots(1, 3, figsize=(8.4, 2.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
a.fill_between(n, s["cs_low"], s["cs_high"], color=ACC, alpha=0.18, label="95% confidence sequence")
a.plot(n, s["estimate"], c=ACC, marker="o", ms=3, label="CUPED estimate")
a.axhline(0, c="k", lw=0.7)
a.axvline(s["rejected_at_n"], c=ORG, ls="--", lw=1)
a.text(s["rejected_at_n"] + 1500, 0.2, "mSPRT rejects\nat n=4,000", fontsize=7, color="#a86400")
a.set_xlabel("units observed")
a.set_ylabel("revenue / user lift")
a.legend(fontsize=6.5, frameon=False, loc="upper right")
a.set_title("Anytime-valid monitoring", fontsize=9)
frs = [o["info_fraction"] for o in s["obf"]]
zs = [o["z"] for o in s["obf"]]
bs = [o["bound"] for o in s["obf"]]
b.plot(frs, bs, c=RED, marker="s", ms=3, lw=1.2, label="O'Brien-Fleming bound")
b.plot(frs, zs, c=ACC, marker="o", ms=3, label="observed z")
b.set_xlabel("information fraction")
b.set_ylabel("z")
b.legend(fontsize=6.5, frameon=False)
b.set_title("Group-sequential checkpoints", fontsize=9)
seg = out["segments"]["segments"]
y = np.arange(len(seg))[::-1]
est = [x["estimate"] for x in seg]
c.errorbar(est, y, xerr=[[e - x["ci_low"] for e, x in zip(est, seg)],
                         [x["ci_high"] - e for e, x in zip(est, seg)]],
           fmt="o", c=ACC, capsize=3)
c.axvline(0, c="k", lw=0.7)
c.set_yticks(y)
c.set_yticklabels([f"segment {x['segment']}" for x in seg])
c.set_xlabel("effect (95% CI)")
c.set_title("Segment effects", fontsize=9)
fig.tight_layout()
save(fig, "readout_seq.png")

# 4 CUPED + SRM
cu = R["cuped"]
sr = R["srm"]["flag_rate"]
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.7))
labs = ["Single pre-period\ncovariate (CUPED)", "Cross-fitted GBM\n(CUPAC)"]
vals = [cu["cupac"]["variance_reduction_pre_period_cuped"], cu["cupac"]["variance_reduction_cupac_gbm"]]
bb = a.bar(labs, [x * 100 for x in vals], color=[GREY, ACC], width=0.55)
for r_, x in zip(bb, vals):
    a.text(r_.get_x() + r_.get_width() / 2, x * 100 + 1, f"{x:.1%}", ha="center", fontsize=8)
a.set_ylabel("Variance reduction (%)")
a.set_ylim(0, 55)
a.tick_params(axis="x", labelsize=7.5)
a.set_title("Non-linear outcomes, n=8,000, 8 datasets", fontsize=9)
ks = ["drop_0.0", "drop_0.01", "drop_0.02", "drop_0.05"]
xs = [0, 1, 2, 5]
b.plot(xs, [sr[k] * 100 for k in ks], marker="o", c=ACC)
for x, k in zip(xs, ks):
    b.text(x + 0.1, sr[k] * 100 + 5, f"{sr[k]:.0%}", ha="center", fontsize=8)
b.set_xlabel("treated users lost (%)")
b.set_ylabel("SRM flagged (% of 200 runs)")
b.set_ylim(0, 110)
b.set_title("SRM detection, n=40,000, p<0.001", fontsize=9)
fig.tight_layout()
save(fig, "cuped_srm.png")

# 5 readout rendering
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.add_patch(FancyBboxPatch((0.05, 0.05), 9.9, 4.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc="white", ec=GREY))
ax.text(0.3, 4.55, f"Experiment: {out['experiment']}", fontweight="bold", fontsize=10)
ax.add_patch(FancyBboxPatch((6.9, 4.35), 2.8, 0.45, boxstyle="round,pad=0.02", fc="#e6f2e6", ec="#2e7d32"))
ax.text(8.3, 4.575, f"status: {out['status']}", ha="center", va="center", fontsize=8.5,
        color="#2e7d32", fontweight="bold")
ax.text(0.3, 4.1, f"{out['n_units']:,} units   |   SRM p = {out['srm']['p_value']:.3f} (no mismatch)   |   "
        f"hash re-check: {out['assignment_check']['mismatched']} mismatched", fontsize=8, color="#333")
rows = [("Metric", "Role", "Effect [95% CI]", "p (Holm)", "Var. red.")]
for mm in out["metrics"]:
    e = mm["effect"]
    pa = mm["p_adjusted"]
    rows.append((mm["name"], mm["role"], f"{e['estimate']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]",
                 f"{pa:.1e}" if pa < 1e-3 else f"{pa:.3f}", f"{mm['variance_reduction']:.1%}"))
g = out["guardrails"][0]
rows.append((g["name"], "guardrail", f"{g['effect']['estimate']:+.2f} (margin {g['margin']:.0f})",
             "NI p<1e-3" if g["p_value"] < 1e-3 else f"{g['p_value']:.3f}",
             "non-inferior" if g["non_inferior"] else "FAIL"))
xs_ = [0.3, 2.9, 4.1, 6.9, 8.4]
y0 = 3.5
for i, r in enumerate(rows):
    for x, t in zip(xs_, r):
        ax.text(x, y0 - i * 0.5, t, fontsize=8, fontweight="bold" if i == 0 else "normal")
    if i == 0:
        ax.plot([0.3, 9.7], [y0 - 0.2] * 2, c=GREY, lw=0.8)
q = out["sequential"]
ax.text(0.3, 1.2, f"Sequential: mSPRT rejected at n = {q['rejected_at_n']:,} of {out['n_units']:,}; "
        f"O'Brien-Fleming checkpoints crossed: {sum(o['crossed'] for o in q['obf'])}/{len(q['obf'])} evaluated",
        fontsize=8)
sg = out["segments"]
ax.text(0.3, 0.6, "Segments: " + ";  ".join(
    f"{x['segment']} {x['estimate']:+.2f}{'*' if x['significant'] else ''}" for x in sg["segments"])
    + "   (* significant after Holm correction)", fontsize=8)
save(fig, "readout.png")
print({k: v for k, v in sg.items() if k != "segments"}, out["status"], len(q["obf"]))
print("null path seed", seed, "hit at n=", ns[hit])
