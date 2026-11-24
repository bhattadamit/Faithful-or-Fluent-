"""Replication script: Faithful or Fluent? (evidence-calibrated screening model).
All inputs are published figures; sources are given in the manuscript (Table 3).
Run: python3 analysis.py   -> results.json and fig1..fig6 PNG in ./out
"""
import json, math, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
R = {}

# ---------------------------------------------------------------- inputs
# RAGTruth response-level detection, QA task (Niu et al. 2024, Table 5): precision, recall (%)
RT = {
    "Prompt, GPT-3.5-turbo": (18.8, 84.4),
    "Prompt, GPT-4-turbo": (33.2, 90.6),
    "SelfCheckGPT": (35.0, 58.0),
    "LMvLM": (18.7, 76.9),
    "Fine-tuned Llama-2-13B": (61.6, 76.3),
}
N_QA, POS_QA = 900, 160  # QA test responses and hallucinated responses (Ravi et al. 2024)

# ------------------------------------------- A1 reconstruct confusion matrices
def reconstruct(prec, rec, n=N_QA, pos=POS_QA):
    best = None
    for tp in range(0, pos + 1):
        for fp in range(0, n - pos + 1):
            if tp + fp == 0:
                continue
            p = 100 * tp / (tp + fp); r = 100 * tp / pos
            err = abs(p - prec) + abs(r - rec)
            if best is None or err < best[0]:
                best = (err, tp, fp)
    return best

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi

det = {}
for name, (p_, r_) in RT.items():
    err, tp, fp = reconstruct(p_, r_)
    fn, tn = POS_QA - tp, N_QA - POS_QA - fp
    se, sp = tp / POS_QA, tn / (N_QA - POS_QA)
    det[name] = dict(tp=tp, fp=fp, fn=fn, tn=tn, se=se, sp=sp, J=se + sp - 1,
                     lr_pos=se / (1 - sp), lr_neg=(1 - se) / sp,
                     se_ci=cp(tp, POS_QA), sp_ci=cp(tn, N_QA - POS_QA),
                     prec_check=100 * tp / (tp + fp), rec_check=100 * tp / POS_QA,
                     fit_err=err)
R["detectors"] = det

# ------------------------------------------- A2 Youden J from accuracy on balanced sets
HB_FIN = {"GPT-4o": 85.3, "GPT-4-Turbo": 82.2, "Lynx (70B)": 81.4, "Llama-3-Instruct-70B": 72.7,
          "Lynx (8B)": 72.5, "Claude-3-Sonnet": 69.7, "GPT-3.5-Turbo": 60.9,
          "RAGAS Faithfulness": 59.5, "Claude-3-Haiku": 58.4}
FB = {"GPT-4-Turbo, zero-shot": 57.65, "GPT-4o, zero-shot": 56.29, "HHEM-2.1": 55.68,
      "MiniCheck-Roberta-L": 55.03, "True-Teacher": 54.21, "HHEM-2.1-Open": 51.37}
R["J_financebench"] = {k: dict(acc=v, J=2 * v / 100 - 1, infl=1 / (2 * v / 100 - 1) ** 2) for k, v in HB_FIN.items()}
R["J_faithbench"] = {k: dict(ba=v, J=2 * v / 100 - 1, infl=1 / (2 * v / 100 - 1) ** 2) for k, v in FB.items()}

# ------------------------------------------- A3 predictive values
def screen(p, se, sp):
    w = p * se + (1 - p) * (1 - sp)                    # share flagged (escalated)
    resid = p * (1 - se) / (p * (1 - se) + (1 - p) * sp)   # P(unfaithful | passed)
    ppv = p * se / w
    return dict(w=w, resid=resid, ppv=ppv, removed=se)

BASE = [0.02, 0.05, 0.10, 0.17, 0.29, 0.33]
R["predictive"] = {n: {str(p): screen(p, d["se"], d["sp"]) for p in BASE} for n, d in det.items()}

# required sensitivity for target residual tau when Sp given, and for perfect Sp
def se_needed(p, tau, sp=1.0):
    # tau = p(1-se)/(p(1-se)+(1-p)sp)  ->  1-se = tau(1-p)sp/(p(1-tau))
    return 1 - tau * (1 - p) * sp / (p * (1 - tau))
R["se_needed"] = {f"p={p},tau={t}": se_needed(p, t, 0.9) for p in (0.05, 0.10, 0.17, 0.29) for t in (0.01, 0.02)}

# ------------------------------------------- A4 audit sample sizes
def n_zero(tau, a=0.05):
    return math.ceil(math.log(a) / math.log(1 - tau))
R["n_zero_failure"] = {str(t): n_zero(t) for t in (0.05, 0.02, 0.01, 0.005)}

def n_gold(p, d, z=1.959964):
    return math.ceil(z * z * p * (1 - p) / d ** 2)
def n_judge(p, d, se, sp, z=1.959964):
    q = p * se + (1 - p) * (1 - sp)
    return math.ceil(z * z * q * (1 - q) / (d ** 2 * (se + sp - 1) ** 2)), q
def phi(p, se, sp):
    q = p * se + (1 - p) * (1 - sp)
    return (se + sp - 1) * math.sqrt(p * (1 - p) / (q * (1 - q)))

aud = {}
for p, d in ((0.05, 0.02), (0.10, 0.03)):
    row = {"gold": n_gold(p, d)}
    for n_, dd in det.items():
        nj, q = n_judge(p, d, dd["se"], dd["sp"])
        ph = phi(p, dd["se"], dd["sp"])
        row[n_] = dict(n_judge=nj, infl=nj / n_gold(p, d), q=q, phi=ph,
                       hybrid_gain=1 / (1 - ph ** 2), label_saving=ph ** 2)
    aud[f"p={p},d={d}"] = row
R["audit"] = aud

# ------------------------------------------- A5 policy thresholds (in units of one human review, c_h = 1)
def thresholds(p, se, sp, h, cj=0.0):
    w = p * se + (1 - p) * (1 - sp)
    L1 = (cj + w) / (p * se * h)                 # release-all -> screen-and-escalate
    L2 = ((1 - w) - cj) / (p * (1 - se) * h)     # screen-and-escalate -> review-all
    L0 = 1 / (p * h)                             # release-all -> review-all (no detector)
    return dict(w=w, L1=L1, L2=L2, L0=L0)
pol = {}
for n_, dd in det.items():
    pol[n_] = {str(p): thresholds(p, dd["se"], dd["sp"], 0.8) for p in (0.05, 0.10, 0.17, 0.29)}
R["policy"] = pol

# ------------------------------------------- A6 fluency vs. citation support (Liu et al. 2023, system level)
LIU = {"Bing Chat": (4.40, 4.34, 58.7, 89.5), "NeevaAI": (4.43, 4.48, 67.6, 72.0),
       "perplexity.ai": (4.51, 4.56, 68.7, 72.7), "YouChat": (4.59, 4.62, 11.1, 63.6)}
A = np.array(list(LIU.values()))
R["liu_r"] = {"utility_precision": float(np.corrcoef(A[:, 1], A[:, 3])[0, 1]),
              "fluency_precision": float(np.corrcoef(A[:, 0], A[:, 3])[0, 1]),
              "fluency_recall": float(np.corrcoef(A[:, 0], A[:, 2])[0, 1]),
              "utility_recall": float(np.corrcoef(A[:, 1], A[:, 2])[0, 1])}

# ------------------------------------------- A7 Monte Carlo
rng = np.random.default_rng(20261116)
M = 200_000
SCEN = {"A: grounded-summarisation range (1.8%-24.2%)": (0.018, 0.242),
        "B: domain RAG tool range (17%-33%)": (0.17, 0.33)}
mc = {}
for sname, (lo, hi) in SCEN.items():
    p = rng.uniform(lo, hi, M)
    h = rng.uniform(0.6, 1.0, M)
    mc[sname] = {"release-all": dict(resid_med=float(np.median(p)), resid_p5=float(np.percentile(p, 5)),
                                     resid_p95=float(np.percentile(p, 95)), w_med=0.0,
                                     pr_le_1=float((p <= 0.01).mean()), pr_le_2=float((p <= 0.02).mean()))}
    for n_, dd in det.items():
        se = rng.beta(dd["tp"] + 1, dd["fn"] + 1, M)
        sp = rng.beta(dd["tn"] + 1, dd["fp"] + 1, M)
        w = p * se + (1 - p) * (1 - sp)
        resid = p * (1 - se) / (p * (1 - se) + (1 - p) * sp)
        resid_h = (p * (1 - se) + p * se * (1 - h)) / 1.0   # per released+reviewed answer, after human review of flags
        L1 = w / (p * se * h); L2 = (1 - w) / (p * (1 - se) * h)
        mc[sname][n_] = dict(resid_med=float(np.median(resid)), resid_p5=float(np.percentile(resid, 5)),
                             resid_p95=float(np.percentile(resid, 95)), w_med=float(np.median(w)),
                             w_p5=float(np.percentile(w, 5)), w_p95=float(np.percentile(w, 95)),
                             pr_le_1=float((resid <= 0.01).mean()), pr_le_2=float((resid <= 0.02).mean()),
                             total_med=float(np.median(resid_h)),
                             L1_med=float(np.median(L1)), L1_p5=float(np.percentile(L1, 5)), L1_p95=float(np.percentile(L1, 95)),
                             L2_med=float(np.median(L2)), L2_p5=float(np.percentile(L2, 5)), L2_p95=float(np.percentile(L2, 95)))
    # sensitivity (Spearman) for the fine-tuned detector's residual
    dd = det["Fine-tuned Llama-2-13B"]
    se = rng.beta(dd["tp"] + 1, dd["fn"] + 1, M); sp = rng.beta(dd["tn"] + 1, dd["fp"] + 1, M)
    resid = p * (1 - se) / (p * (1 - se) + (1 - p) * sp)
    mc[sname]["spearman_resid"] = {k: float(stats.spearmanr(v, resid)[0]) for k, v in (("p", p), ("Se", se), ("Sp", sp))}
R["mc"] = mc
R["mc_draws"] = M

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)

# ================================================================ figures
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa8"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 600})
SHORT = {"Prompt, GPT-3.5-turbo": "Prompt GPT-3.5", "Prompt, GPT-4-turbo": "Prompt GPT-4",
         "SelfCheckGPT": "SelfCheckGPT", "LMvLM": "LMvLM", "Fine-tuned Llama-2-13B": "Fine-tuned detector"}
COL = dict(zip(RT.keys(), C))

# Fig 1: evaluation as screening (schematic)
fig, ax = plt.subplots(figsize=(9.6, 3.7)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
def box(x, y, w, h, t, fc, ec=MUTED, fs=8.2, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")
def arr(x1, y1, x2, y2, t=None, dy=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=MUTED))
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, t, ha="center", va="bottom", fontsize=8, color=MUTED)
box(1, 15, 15, 10, "Staff or customer\nprocedure question", "#eef4fc")
box(21, 15, 15, 10, "Retrieval from\nprocedure corpus", "#eef4fc")
box(41, 15, 15, 10, "Generated answer\n(unfaithful with\nprobability p)", "#fdeee7")
box(61, 15, 16, 10, "Faithfulness\nevaluator\n(Se, Sp)", "#e8f7f1", bold=True)
box(84, 27, 15, 9, "Released\n(residual rate ρ)", "#f3f2ee")
box(84, 4, 15, 9, "Escalated to\nhuman review\n(share w)", "#fdf5e0")
arr(16.6, 20, 20.6, 20); arr(36.6, 20, 40.6, 20); arr(56.6, 20, 60.6, 20)
arr(77.6, 22.5, 83.6, 30); arr(77.6, 17.5, 83.6, 10)
ax.text(79.2, 28.2, "passes", ha="right", fontsize=8, color=MUTED); ax.text(79.2, 10.6, "flagged", ha="right", fontsize=8, color=MUTED)
ax.text(48.5, 34, "Fluency is visible to the reader at this point; faithfulness is not", ha="center", fontsize=8, color=MUTED, style="italic")
ax.plot([48.5, 48.5], [25.8, 32.8], color=MUTED, lw=0.7, ls=":")
box(21, 1, 56, 7, "Periodic audit: n sampled answers labelled by compliance staff\nestimate p, Se and Sp", "#ffffff", fs=7.8)
arr(49, 8.6, 49, 14.4)
fig.savefig(os.path.join(OUT, "fig1.png"), bbox_inches="tight"); plt.close(fig)

# Fig 2: published unfaithfulness rates (ranges / points)
items = [("Grounded summarisation, 100+ models\n(public leaderboard, 2026)", 1.8, 24.2, C[0]),
         ("RAG question answering, GPT-3.5/GPT-4,\nall tasks (RAGTruth)", 13.5, 13.7, C[0]),
         ("Commercial legal research tools\n(retrieval-grounded)", 17, 33, C[1]),
         ("RAG question answering, six models,\nQA task (RAGTruth)", 29.1, 29.1, C[0]),
         ("Industry RAG systems, answers not\nhallucination-free (CRAG)", 37, 37, C[1]),
         ("Generative search engines, sentences\nnot fully supported", 48.5, 48.5, C[1]),
         ("RAG question answering, open-weight\nmodels, all tasks (RAGTruth)", 47.1, 65.9, C[0])]
fig, ax = plt.subplots(figsize=(7.2, 3.5))
for i, (lab, lo, hi, c) in enumerate(items):
    y = len(items) - 1 - i
    if hi - lo > 0.5:
        ax.plot([lo, hi], [y, y], color=c, lw=5, solid_capstyle="round")
        ax.text(hi + 1.5, y, f"{lo:g}% to {hi:g}%", va="center", fontsize=8, color=INK)
    else:
        ax.plot([(lo + hi) / 2], [y], "o", color=c, ms=8, mec=SURF, mew=1.5)
        ax.text(hi + 2.2, y, f"{lo:g}% to {hi:g}%" if lo != hi else f"{lo:g}%", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(items))); ax.set_yticklabels([it[0] for it in items][::-1], fontsize=8, color=INK)
ax.set_xlim(0, 80); ax.set_xlabel("Share of outputs with unsupported or incorrect content (%)")
ax.grid(axis="y", visible=False)
ax.plot([], [], color=C[0], lw=5, label="Benchmark with human or model-based labels of faithfulness to supplied context")
ax.plot([], [], color=C[1], lw=5, label="Evaluation of deployed retrieval-grounded products")
ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.17), frameon=False, fontsize=8, ncol=1)
fig.savefig(os.path.join(OUT, "fig2.png"), bbox_inches="tight"); plt.close(fig)

# Fig 3: operating points in ROC space
fig, ax = plt.subplots(figsize=(4.9, 4.3))
for J in (0.2, 0.4, 0.6, 0.8):
    x = np.linspace(0, 1 - J, 50); ax.plot(x, x + J, color=GRID, lw=0.8)
    ax.text(1 - J, 1.012, f"J = {J}", fontsize=8, color=MUTED, ha="center", va="bottom")
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
ax.text(0.43, 0.395, "no discrimination", fontsize=8, color=MUTED, rotation=38)
off = {"Prompt, GPT-3.5-turbo": (-0.02, 0.05, "right"), "Prompt, GPT-4-turbo": (-0.065, 0.03, "right"), "SelfCheckGPT": (0.035, -0.045, "left"),
       "LMvLM": (0.045, -0.075, "left"), "Fine-tuned Llama-2-13B": (0.04, -0.03, "left")}
for n_, dd in det.items():
    x, y = 1 - dd["sp"], dd["se"]
    ax.errorbar(x, y, xerr=[[x - (1 - dd["sp_ci"][1])], [(1 - dd["sp_ci"][0]) - x]],
                yerr=[[y - dd["se_ci"][0]], [dd["se_ci"][1] - y]], fmt="o", color=COL[n_], ms=7, mec=SURF, mew=1.2, lw=1.1, capsize=2)
    dx, dy, ha = off[n_]
    ax.text(x + dx, y + dy, f"{SHORT[n_]}\nJ = {dd['J']:.2f}", fontsize=8, color=INK, ha=ha, va="center")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("False-alarm rate, 1 − Sp (faithful answers flagged)"); ax.set_ylabel("Sensitivity, Se (unfaithful answers flagged)")
fig.savefig(os.path.join(OUT, "fig3.png"), bbox_inches="tight"); plt.close(fig)

# Fig 4: residual rate and workload vs base rate (small multiples, one axis each)
pg = np.linspace(0.005, 0.35, 200)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
show = ["Prompt, GPT-4-turbo", "SelfCheckGPT", "Fine-tuned Llama-2-13B"]
axs[0].plot(pg * 100, pg * 100, color=MUTED, lw=1.2, ls="--"); axs[0].text(0.7, 4.6, "no screening", fontsize=8, color=MUTED, rotation=62)
lab_y = {}
for n_ in show:
    dd = det[n_]; s = [screen(p, dd["se"], dd["sp"]) for p in pg]
    axs[0].plot(pg * 100, [x["resid"] * 100 for x in s], color=COL[n_], lw=2)
    axs[1].plot(pg * 100, [x["w"] * 100 for x in s], color=COL[n_], lw=2)
    la = {"Prompt, GPT-4-turbo": (35.6, 7.7, "left"), "SelfCheckGPT": (14.6, 8.3, "left"), "Fine-tuned Llama-2-13B": (35.6, 11.6, "left")}[n_]
    lb = {"Prompt, GPT-4-turbo": 57.3, "SelfCheckGPT": 37.3, "Fine-tuned Llama-2-13B": 31.2}[n_]
    if n_ == "SelfCheckGPT":
        yy = screen(0.165, dd["se"], dd["sp"])["resid"] * 100
        axs[0].annotate(SHORT[n_], xy=(16.5, yy), xytext=(19.0, yy), fontsize=8, color=INK, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=0))
    else:
        axs[0].text(la[0], la[1], SHORT[n_], fontsize=8, color=INK, va="center", ha=la[2])
    axs[1].text(35.6, lb, SHORT[n_], fontsize=8, color=INK, va="center")
for t in (1, 2):
    axs[0].axhline(t, color=MUTED, lw=0.7, ls=":"); axs[0].text(34.6, t + 0.22, f"{t}% tolerance", fontsize=8, color=MUTED, ha="right")
axs[0].set_ylim(0, 12); axs[0].set_xlim(0, 35); axs[1].set_xlim(0, 35); axs[1].set_ylim(0, 70)
axs[0].set_xlabel("Base rate of unfaithful answers, p (%)"); axs[1].set_xlabel("Base rate of unfaithful answers, p (%)")
axs[0].set_ylabel("Unfaithful share of released, ρ (%)"); axs[1].set_ylabel("Escalated to human review, w (%)")
axs[0].set_title("(a) Residual risk after screening", fontsize=9, loc="left", color=INK)
axs[1].set_title("(b) Review workload created", fontsize=9, loc="left", color=INK)
fig.subplots_adjust(wspace=0.75, right=0.86)
fig.savefig(os.path.join(OUT, "fig4.png"), bbox_inches="tight"); plt.close(fig)

# Fig 5: label inflation factor 1/J^2
rows = [("Fine-tuned detector, RAGTruth QA", det["Fine-tuned Llama-2-13B"]["J"], C[0]),
        ("Prompt GPT-4, RAGTruth QA", det["Prompt, GPT-4-turbo"]["J"], C[0]),
        ("SelfCheckGPT, RAGTruth QA", det["SelfCheckGPT"]["J"], C[0]),
        ("GPT-4o judge, FinanceBench subset", R["J_financebench"]["GPT-4o"]["J"], C[1]),
        ("Lynx (70B), FinanceBench subset", R["J_financebench"]["Lynx (70B)"]["J"], C[1]),
        ("Claude-3-Sonnet judge, FinanceBench subset", R["J_financebench"]["Claude-3-Sonnet"]["J"], C[1]),
        ("RAGAS Faithfulness, FinanceBench subset", R["J_financebench"]["RAGAS Faithfulness"]["J"], C[1]),
        ("GPT-4-Turbo judge, FaithBench", R["J_faithbench"]["GPT-4-Turbo, zero-shot"]["J"], C[2]),
        ("HHEM-2.1, FaithBench", R["J_faithbench"]["HHEM-2.1"]["J"], C[2])]
rows.sort(key=lambda r: 1 / r[1] ** 2)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for i, (lab, J, c) in enumerate(rows):
    v = 1 / J ** 2; y = len(rows) - 1 - i
    ax.barh(y, v, color=c, height=0.62, left=1)
    ax.text((v + 1) * 1.12, y, f"{v:,.1f}×  (J = {J:.2f})", va="center", fontsize=8, color=INK)
ax.set_xscale("log"); ax.set_xlim(1, 400); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8, color=INK)
ax.set_xlabel("Lower bound on judge-labelled answers needed per human-labelled answer, 1/J² (log scale)")
ax.grid(axis="y", visible=False)
for c, l in ((C[0], "RAGTruth QA (reconstructed)"), (C[1], "FinanceBench subset of HaluBench"), (C[2], "FaithBench (hard cases)")):
    ax.barh([-5], [0], color=c, label=l)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3)
fig.savefig(os.path.join(OUT, "fig5.png"), bbox_inches="tight"); plt.close(fig)

# Fig 6: regime map (fine-tuned detector, h = 0.8)
dd = det["Fine-tuned Llama-2-13B"]
pg = np.linspace(0.01, 0.35, 300)
L1 = np.array([thresholds(p, dd["se"], dd["sp"], 0.8)["L1"] for p in pg])
L2 = np.array([thresholds(p, dd["se"], dd["sp"], 0.8)["L2"] for p in pg])
fig, ax = plt.subplots(figsize=(6.2, 3.7))
ax.fill_between(pg * 100, 0.5, L1, color="#cde2fb", lw=0)
ax.fill_between(pg * 100, L1, L2, color="#c9eedd", lw=0)
ax.fill_between(pg * 100, L2, 5000, color="#fbd9c9", lw=0)
ax.plot(pg * 100, L1, color=C[0], lw=2); ax.plot(pg * 100, L2, color=C[1], lw=2)
ax.set_yscale("log"); ax.set_ylim(0.5, 5000); ax.set_xlim(1, 35)
ax.text(24, 1.05, "Release without screening", fontsize=8.4, color=INK, ha="center")
ax.text(22, 9, "Screen and escalate flagged answers", fontsize=8.4, color=INK, ha="center")
ax.text(22, 600, "Human review of every answer", fontsize=8.4, color=INK, ha="center")
ax.text(4.2, L1[25] * 0.42, "λ₁", fontsize=9, color=INK); ax.text(4.2, L2[25] * 1.5, "λ₂", fontsize=9, color=INK)
ax.set_xlabel("Base rate of unfaithful answers, p (%)")
ax.set_ylabel("Cost of one unfaithful answer acted upon,\nin multiples of the cost of one human review (log scale)")
ax.grid(visible=False)
fig.savefig(os.path.join(OUT, "fig6.png"), bbox_inches="tight"); plt.close(fig)
print("done")

# =====================================================================================
# A9  Re-analysis of released benchmark data (RAGTruth, HaluBench, FaithBench)
#     Inputs: reanalysis/results_reanalysis.json, union_results.json, numeric_groups.json,
#     produced by the re-analysis kit (HHEM-2.1-Open and the deterministic number check).
# =====================================================================================
RA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reanalysis")
if os.path.exists(os.path.join(RA_DIR, "results_reanalysis.json")):
    RA = json.load(open(os.path.join(RA_DIR, "results_reanalysis.json")))
    UN = json.load(open(os.path.join(RA_DIR, "union_results.json")))
    NG = json.load(open(os.path.join(RA_DIR, "numeric_groups.json")))
    def cm(tp, fp, fn, tn):
        se, sp = tp / (tp + fn), tn / (tn + fp)
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=tp + fp + fn + tn, pos=tp + fn, se=se, sp=sp, J=se + sp - 1,
                    se_ci=cp(tp, tp + fn), sp_ci=cp(tn, tn + fp))
    acc = {}
    for row in RA["accuracy"]:
        if row["subset"] in ("QA", "ALL", "FinanceBench", "DROP", "RAGTruth", "Data2txt", "Summary"):
            acc[f'{row["dataset"]}|{row["subset"]}|{row["checker"]}'] = cm(row["tp"], row["fp"], row["fn"], row["tn"])
    for nm, key in (("RAGTruth|QA|UNION", "RAGTruth QA test"), ("RAGTruth|ALL|UNION", "RAGTruth all test"),
                    ("HaluBench|FinanceBench|UNION", "HaluBench FinanceBench"), ("FaithBench|ALL|UNION", "FaithBench strict")):
        u = UN[key]["union"]; acc[nm] = cm(u["tp"], u["fp"], u["fn"], u["tn"])
    num = {}
    for scope, d in NG.items():
        num[scope] = {}
        for c in ("HHEM", "PARAM", "UNION"):
            g = d[c]; r = {}
            for k, (n, k_) in g.items():
                r[k] = dict(n=n, k=k_, rate=k_ / n, ci=cp(k_, n))
            r["p_se"] = float(stats.fisher_exact([[g["hall_numeric"][1], g["hall_numeric"][0] - g["hall_numeric"][1]], [g["hall_other"][1], g["hall_other"][0] - g["hall_other"][1]]])[1])
            r["p_fa"] = float(stats.fisher_exact([[g["faithful_numeric"][1], g["faithful_numeric"][0] - g["faithful_numeric"][1]], [g["faithful_other"][1], g["faithful_other"][0] - g["faithful_other"][1]]])[1])
            num[scope][c] = r
    types = {c: {t: dict(n=v["n"], k=v["flagged"], rate=v["rate"], ci=v["ci"]) for t, v in RA["sensitivity_by_type"][c].items()} for c in ("HHEM", "PARAM")}
    types["UNION"] = {t: dict(n=n, k=k, rate=k / n, ci=cp(k, n)) for t, (n, k) in NG["RAGTruth all test"]["UNION_by_type"].items()}
    scr = {}
    for c in ("HHEM", "PARAM", "UNION"):
        m = acc[f"RAGTruth|QA|{c}"]; scr[c] = {}
        for p in (0.02, 0.05, 0.10, 0.17, 0.29):
            s = screen(p, m["se"], m["sp"]); q = s["w"]
            scr[c][str(p)] = dict(resid=s["resid"], w=s["w"], infl=q * (1 - q) / (p * (1 - p) * m["J"] ** 2),
                                  saving=m["J"] ** 2 * p * (1 - p) / (q * (1 - q)), L1=s["w"] / (p * m["se"] * 0.8), L2=(1 - s["w"]) / (p * (1 - m["se"]) * 0.8))
    R["reanalysis"] = dict(composition=RA["ragtruth_composition"], qa_check=RA["ragtruth_qa_test_check"], by_model=RA["ragtruth_by_model"],
                           span_types=RA["ragtruth_span_types"], numeric_spans=RA["ragtruth_numeric_spans"], numeric_responses=RA["ragtruth_numeric_responses"],
                           accuracy=acc, numeric=num, types=types, screening=scr, n_records=RA["n_records"])
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)

    # Figure 7: numeric versus other content, RAGTruth test split (all tasks)
    g = num["RAGTruth all test"]; labs = ["HHEM-2.1-Open", "Number\ncheck", "Either\nflags"]; keys = ["HHEM", "PARAM", "UNION"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2)); x = np.arange(3)
    for ax, (a, b, ttl) in zip(axs, (("hall_numeric", "hall_other", "(a) Sensitivity on unfaithful answers"), ("faithful_numeric", "faithful_other", "(b) False alarms on faithful answers"))):
        for off, kk, col, lab in ((-0.19, a, C[0], "Answer or hallucinated span states a number, amount or date"), (0.19, b, C[1], "No number, amount or date")):
            v = [g[k][kk]["rate"] * 100 for k in keys]; lo = [v[i] - g[k][kk]["ci"][0] * 100 for i, k in enumerate(keys)]; hi = [g[k][kk]["ci"][1] * 100 - v[i] for i, k in enumerate(keys)]
            ax.bar(x + off, v, 0.36, color=col, yerr=[lo, hi], capsize=2.5, error_kw=dict(lw=0.8, ecolor=MUTED), label=lab)
            for xi, vi, hh in zip(x + off, v, hi): ax.text(xi, vi + hh + 1.5, f"{vi:.0f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([l.replace("HHEM-2.1-Open", "HHEM-2.1\nOpen") for l in labs], fontsize=8); ax.set_ylim(0, 100); ax.set_title(ttl, fontsize=9, loc="left", color=INK); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Share flagged (%)")
    axs[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(1.1, -0.13), ncol=2)
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(os.path.join(OUT, "fig7.png"), bbox_inches="tight"); plt.close(fig)
    print("re-analysis section done")
