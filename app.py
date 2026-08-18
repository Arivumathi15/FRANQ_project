"""FRANQ-EXT — Interactive Demo (offline, CPU-only, no GPU / no internet).

Run:
    pip install streamlit
    streamlit run app.py

This drives the REAL franq_ext pipeline with the deterministic MockLLM backend, so it needs
no model download and no GPU. It traces one answer through all three pillars live —
extraction -> entity graph -> faithfulness/UQ -> calibrated router -> targeted correction ->
re-verification + correction-regret — exactly the flow used in the paper.

A second tab shows the REAL PopQA results (loaded from results_popqa/), so the same app
presents both the working system and the measured evidence.
"""
from __future__ import annotations

import os
import csv

# Keep the demo self-contained and side-effect free (no on-disk signal cache writes).
os.environ.setdefault("FRANQ_SIGNAL_CACHE", "0")

import streamlit as st

from franq_ext.config import Config
from franq_ext.data import load_sample
from franq_ext.pipeline import Pipeline
from franq_ext.train_router import train_router
from franq_ext.correction import Corrector
from franq_ext.verification import ReVerifier

# --------------------------------------------------------------------------- page + theme
st.set_page_config(page_title="FRANQ-EXT — Interactive Demo", page_icon="🔎", layout="wide")

NAVY, TEAL, AMBER, GREEN, RED, MUTED = "#0f2544", "#0ea5a4", "#f59e0b", "#10b981", "#e05a47", "#6b7280"

st.markdown(f"""
<style>
  .block-container {{padding-top:2.2rem;max-width:1150px}}
  h1,h2,h3 {{color:{NAVY}}}
  .hero {{background:linear-gradient(135deg,#0f2544,#16345c);color:#fff;border-radius:16px;
    padding:1.4rem 1.8rem;margin-bottom:1.2rem}}
  .hero h1 {{color:#fff;margin:0;font-size:1.9rem}}
  .hero p {{color:#c7d6ea;margin:.4rem 0 0;font-size:1.02rem}}
  .badge {{display:inline-block;padding:.18em .7em;border-radius:999px;font-size:.8rem;
    font-weight:700;color:#fff;margin-left:.3em}}
  .b-flag {{background:{AMBER}}} .b-ok {{background:{GREEN}}} .b-fix {{background:{TEAL}}}
  .b-bad {{background:{RED}}} .b-none {{background:{MUTED}}}
  .step {{border-left:4px solid {TEAL};background:#f7f9fc;border-radius:10px;
    padding:.9rem 1.1rem;margin:.5rem 0}}
  .step h4 {{margin:0 0 .5rem;color:{NAVY};font-size:1.05rem}}
  .factline {{font-family:ui-monospace,Consolas,monospace;font-size:.95rem;color:#243}}
  .bar {{height:10px;border-radius:6px;background:#e5e7eb;overflow:hidden;display:inline-block;
    width:120px;vertical-align:middle}}
  .bar > i {{display:block;height:100%}}
  .arrow {{color:{TEAL};font-weight:800}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- pipeline (cached)
EXAMPLES = {
    "einstein": ("Albert Einstein", "birth city hallucinated (Munich → should be Ulm)"),
    "newton":   ("Isaac Newton", "birth year hallucinated (1642 → should be 1643)"),
    "curie":    ("Marie Curie", "all facts correct — system should leave it untouched"),
}


@st.cache_resource(show_spinner="Training the calibrated router (Pillar 2)…")
def build_pipeline():
    cfg = Config()  # offline defaults: MockLLM, lexical retriever + scorer, learned router
    examples = load_sample()
    router = train_router(cfg, examples, augment=40)  # tiny set -> jitter augmentation
    pipe = Pipeline(cfg, router=router)
    return cfg, pipe, {e.qid: e for e in examples}


def faith_bar(v: float) -> str:
    color = GREEN if v >= 0.6 else (AMBER if v >= 0.35 else RED)
    return (f'<span class="bar"><i style="width:{int(v*100)}%;background:{color}"></i></span> '
            f'<b>{v:.2f}</b>')


def run_pipeline(cfg, pipe, example):
    """Drive the stages in order and capture each step's true state for rendering."""
    facts, graph, retriever = pipe.compute_signals(example)
    out = {"question": example.question, "generated": example.generated_answer}

    out["facts"] = [(f.entity, f.attribute, f.value) for f in facts]
    out["graph"] = {ent: [(g.attribute, g.value, g.dependency_strength) for g in grp]
                    for ent, grp in graph.groups().items()}
    out["scores"] = [(f.entity, f.attribute, f.value, f.faithfulness,
                      f.semantic_entropy, f.parametric_uq) for f in facts]

    pipe.router.route_facts(facts, graph, flag_threshold=cfg.router.flag_threshold)
    out["routing"] = [(f.entity, f.attribute, f.value, f.error_prob, f.route, f.flagged)
                      for f in facts]

    before_values = {f.key(): f.value for f in facts}
    corrector = Corrector(pipe.llm, retriever, pipe.scorer, cfg.correction)
    records = corrector.correct_flagged(facts)
    out["records"] = [(r.fact_key, r.old_value, r.new_value, r.old_faithfulness,
                       r.new_faithfulness, r.attempts, r.accepted) for r in records]

    reverifier = ReVerifier(pipe.scorer)
    reverifier.recompute_faithfulness(facts, example.contexts)
    ev = reverifier.evaluate(facts, example.gold_facts, before_values)
    out["reverify"] = ev.per_fact
    out["n_before"], out["n_after"] = ev.n_correct_before, ev.n_correct_after
    out["regret"] = ev.correction_regret
    out["n_flagged"] = sum(1 for f in facts if f.flagged)
    out["n_corrected"] = sum(1 for f in facts if f.corrected)
    out["n_facts"] = len(facts)
    return out


# --------------------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>🔎 FRANQ-EXT — Interactive Demo</h1>
  <p>Watch a RAG answer flow through all three pillars: detect the wrong fact, correct only
  that fact, and prove the fix caused no new damage. Runs fully offline — no GPU, no internet.</p>
</div>
""", unsafe_allow_html=True)

tab_live, tab_results = st.tabs(["  ▶  Live Pipeline  ", "  📊  Real Results (PopQA)  "])

# =========================================================================== LIVE TAB
with tab_live:
    cfg, pipe, ex_map = build_pipeline()

    c1, c2 = st.columns([2, 3])
    with c1:
        qid = st.selectbox("Choose an example answer to verify:",
                           list(EXAMPLES.keys()),
                           format_func=lambda k: EXAMPLES[k][0])
        st.caption("Scenario: " + EXAMPLES[qid][1])
        run = st.button("▶  Run the pipeline", type="primary", use_container_width=True)

    if run or qid:
        example = ex_map[qid]
        out = run_pipeline(cfg, pipe, example)

        with c2:
            st.markdown("**Question**")
            st.info(out["question"])
            st.markdown("**Generated answer (to be verified)**")
            st.warning(out["generated"])

        st.divider()

        # STEP 1-2 extraction
        st.markdown('<div class="step"><h4>① &nbsp;Fact extraction — answer → (entity, attribute, value)</h4>',
                    unsafe_allow_html=True)
        for e, a, v in out["facts"]:
            st.markdown(f'<div class="factline">• ({e}, <b>{a}</b>, {v})</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # STEP 3 graph
        st.markdown('<div class="step"><h4>② &nbsp;Entity graph (Pillar 1) — facts linked by entity</h4>',
                    unsafe_allow_html=True)
        for ent, grp in out["graph"].items():
            attrs = ", ".join(f"{a}={v}" for a, v, _ in grp)
            deps = [round(d, 2) for _, _, d in grp]
            st.markdown(f'<div class="factline">• <b>{ent}</b> → [{attrs}]<br>'
                        f'&nbsp;&nbsp;&nbsp;dependency&nbsp;strength = {deps}</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # STEP 4 scoring
        st.markdown('<div class="step"><h4>③ &nbsp;Faithfulness + uncertainty signals</h4>',
                    unsafe_allow_html=True)
        for e, a, v, faith, se, puq in out["scores"]:
            st.markdown(f'<div class="factline">• ({e}, <b>{a}</b>, {v}) &nbsp; '
                        f'faithfulness {faith_bar(faith)} &nbsp; '
                        f'sem-entropy <b>{se:.2f}</b> &nbsp; parametric-UQ <b>{puq:.2f}</b></div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # STEP 5 routing
        st.markdown('<div class="step"><h4>④ &nbsp;Router decision (Pillar 2) — calibrated P(wrong)</h4>',
                    unsafe_allow_html=True)
        for e, a, v, ep, route, flagged in out["routing"]:
            badge = ('<span class="badge b-flag">FLAGGED → check</span>' if flagged
                     else '<span class="badge b-ok">looks fine</span>')
            st.markdown(f'<div class="factline">• ({e}, <b>{a}</b>, {v}) &nbsp; '
                        f'P(wrong)=<b>{ep:.2f}</b> &nbsp; route={route} {badge}</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # STEP 6 correction
        st.markdown('<div class="step"><h4>⑤ &nbsp;Targeted correction (Pillar 3) — fix only the flagged fact</h4>',
                    unsafe_allow_html=True)
        if not out["records"]:
            st.markdown('<div class="factline">Nothing was flagged — no correction attempted.</div>',
                        unsafe_allow_html=True)
        for (ent, attr), old, new, of, nf, attempts, accepted in out["records"]:
            if accepted:
                tag = '<span class="badge b-fix">REPLACED</span>'
                body = (f'<b>{old}</b> (faith {of:.2f}) <span class="arrow">→</span> '
                        f'<b style="color:{GREEN}">{new}</b> (faith {nf:.2f})')
            else:
                tag = '<span class="badge b-none">kept</span>'
                body = f'<b>{old}</b> — no better-supported value found'
            st.markdown(f'<div class="factline">• focused search “<b>{ent} {attr}</b>” '
                        f'(budget {cfg.correction.budget}, attempts {attempts}) {tag}<br>'
                        f'&nbsp;&nbsp;&nbsp;{body}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # STEP 7 reverify + regret
        st.markdown('<div class="step"><h4>⑥ &nbsp;Re-verification + correction-regret (the safety check)</h4>',
                    unsafe_allow_html=True)
        for pf in out["reverify"]:
            ent, attr = pf["fact"]
            changed = pf["before"] != pf["after"]
            arrow = (f'<b>{pf["before"]}</b> <span class="arrow">→</span> <b>{pf["after"]}</b>'
                     if changed else f'{pf["after"]} (unchanged)')
            ok_b = "✓" if pf["correct_before"] else "✗"
            ok_a = "✓" if pf["correct_after"] else "✗"
            st.markdown(f'<div class="factline">• ({ent}, <b>{attr}</b>): {arrow} &nbsp; '
                        f'gold=<b>{pf["gold"]}</b> &nbsp; '
                        f'before {ok_b} → after {ok_a}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # summary metrics
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Facts checked", out["n_facts"])
        m2.metric("Flagged", out["n_flagged"])
        m3.metric("Corrected", out["n_corrected"])
        acc_b = out["n_before"] / max(out["n_facts"], 1)
        acc_a = out["n_after"] / max(out["n_facts"], 1)
        m4.metric("Accuracy", f"{acc_a*100:.0f}%", f"{(acc_a-acc_b)*100:+.0f}%")
        m5.metric("Correction-regret", f"{out['regret']:.2f}",
                  "no damage" if out["regret"] == 0 else "damage!",
                  delta_color="normal" if out["regret"] == 0 else "inverse")
        if out["regret"] == 0 and out["n_corrected"] > 0:
            st.success("✅ Hallucination fixed **and** correction-regret = 0 — no previously-correct "
                       "fact was broken. That is exactly the guarantee FRANQ-EXT adds.")
        elif out["n_corrected"] == 0:
            st.info("ℹ️ This answer was already faithful, so the system correctly left it untouched "
                    "(no needless edits — regret stays 0).")

# =========================================================================== RESULTS TAB
with tab_results:
    st.subheader("Real experiment results — PopQA (Qwen2.5-3B on GPU)")
    st.caption("These are the measured numbers from the full Kaggle run, loaded from results_popqa/.")

    # Find the results folder wherever it was dropped (names vary by how it was downloaded).
    _candidates = [os.environ.get("FRANQ_RESULTS", ""), "results_popqa",
                   "result_results_popqa", "results"]
    results_dir = next((d for d in _candidates
                        if d and os.path.exists(os.path.join(d, "tables", "ablation.csv"))),
                       "results_popqa")
    abl = os.path.join(results_dir, "tables", "ablation.csv")
    reg = os.path.join(results_dir, "tables", "regret_analysis.csv")

    def read_csv(path):
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    if os.path.exists(abl):
        rows = read_csv(abl)
        st.markdown("**Ablation ladder — each pillar isolated**")
        head = ["Condition", "AUROC ↑", "PRR ↑", "ECE ↓", "Acc before", "Acc after",
                "Corrected", "Regret ↓"]
        html = ("<table style='width:100%;border-collapse:collapse;font-size:.95rem'>"
                "<thead><tr>" + "".join(
                    f"<th style='background:{NAVY};color:#fff;padding:.5em .6em;text-align:center'>{h}</th>"
                    for h in head) + "</tr></thead><tbody>")
        for r in rows:
            hl = r["short_name"] == "A3_full"
            bg = "#fff7e8" if hl else "#fff"
            cells = [r["short_name"], f'{float(r["auroc"]):.3f}', f'{float(r["prr"]):.3f}',
                     f'{float(r["ece"]):.3f}', f'{float(r["factual_acc_before"])*100:.1f}%',
                     f'{float(r["factual_acc_after"])*100:.1f}%', r["n_corrected"],
                     f'{float(r["correction_regret"]):.3f}']
            row = f"<tr style='background:{bg};border-bottom:1px solid #e5e7eb'>"
            for j, c in enumerate(cells):
                align = "left" if j == 0 else "center"
                weight = "700" if j == 0 else "400"
                row += f"<td style='padding:.5em .6em;text-align:{align};font-weight:{weight};color:{NAVY}'>{c}</td>"
            html += row + "</tr>"
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)
        st.caption("Pillar 2 → ECE collapses 0.60 → 0.03 · Pillar 1 → PRR 0.48 → 0.54 · "
                   "Pillar 3 → accuracy +2.6 pts at near-zero regret.")
    else:
        st.warning(f"Ablation results not found at `{abl}`. "
                   "Place the downloaded `results_popqa/` folder next to app.py.")

    if os.path.exists(reg):
        st.markdown("**The key finding — safe vs reckless correction**")
        rows = {r["scenario"]: r for r in read_csv(reg)}
        cA, cB = st.columns(2)
        if "safety_checked" in rows:
            s = rows["safety_checked"]
            with cA:
                st.markdown(f"<h4 style='color:{GREEN}'>✓ Safety-checked (ours)</h4>", unsafe_allow_html=True)
                st.metric("Corrected", s["n_corrected"])
                st.metric("Correction-regret", f'{float(s["correction_regret"]):.3f}')
                st.metric("Final accuracy", f'{float(s["factual_acc_after"])*100:.1f}%')
        if "reckless" in rows:
            r = rows["reckless"]
            with cB:
                st.markdown(f"<h4 style='color:{RED}'>✗ Reckless</h4>", unsafe_allow_html=True)
                st.metric("Corrected", r["n_corrected"])
                st.metric("Correction-regret", f'{float(r["correction_regret"]):.3f}')
                st.metric("Final accuracy", f'{float(r["factual_acc_after"])*100:.1f}%')
        st.info("Reckless correction 'fixes' far more facts, but its regret is ~10× higher and it "
                "drags accuracy **below** the baseline. Correction-regret is what exposes this — "
                "and no prior method measures it.")
    else:
        st.caption("(Add results_popqa/ to also see the safe-vs-reckless comparison here.)")

st.markdown(f"<p style='color:{MUTED};font-size:.85rem;margin-top:2rem'>"
            "FRANQ-EXT · offline demo uses the deterministic MockLLM (no GPU). "
            "The GPU is only used to produce the real PopQA numbers shown in the second tab.</p>",
            unsafe_allow_html=True)
