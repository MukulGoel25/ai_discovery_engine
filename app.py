"""SIP Discovery Engine - Streamlit app with FinBot.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud with a free GEMINI_API_KEY in Secrets (see README).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SIP Discovery Engine", page_icon="🧭", layout="wide", initial_sidebar_state="auto")

from app_ui import charts  # noqa: E402
from app_ui import style as S  # noqa: E402
from engine import analyze as A  # noqa: E402
from engine import llm, rag, schema, settings, survey  # noqa: E402

S.inject()
LABELS = settings.theme_labels()
LIST_COLS = ["themes", "continuation_drivers", "checks", "confusions", "outside_app_actions"]
MAX_CALLS = int(settings.env("MAX_CALLS_PER_SESSION", "20"))
CFG = {"displayModeBar": False}
st.session_state.setdefault("calls", 0)
st.session_state.setdefault("fb_history", [])
SUGGESTIONS = ["Why do first-time investors stop their SIP?", "What happens after an instalment fails?",
               "What confuses people about XIRR and returns?", "Which problem should INDmoney fix first?"]


# =================================================================== data
def _mtime(p):
    return p.stat().st_mtime if p.exists() else 0


@st.cache_data(show_spinner=False)
def load_items(_m: float) -> pd.DataFrame:
    df = pd.read_parquet(settings.ITEMS_FILE)
    for c in LIST_COLS:
        df[c] = df[c].map(A.as_list)
    df["date"] = df["date"].fillna("").astype(str)
    plat = df["platform"].fillna("generic")
    xp = df["x_platform"] if "x_platform" in df else pd.Series("unknown", index=df.index)
    df["app"] = np.where(plat.isin(["generic", ""]), xp, plat)
    df["app"] = df["app"].replace({"Zerodha Kite": "Zerodha"}).fillna("unknown")
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(1)
    if "lang" not in df:
        df["lang"] = "en"
    return df


@st.cache_data(show_spinner=False)
def load_json(path: str, _m: float):
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource(show_spinner=False)
def build_index(_m: float):
    return rag.Index(A.relevant(load_items(_m)))


if not settings.ITEMS_FILE.exists():
    st.markdown("## No data yet")
    st.code("python pipeline.py all     # real data (needs a free GEMINI_API_KEY)\npython pipeline.py demo    # synthetic preview")
    st.stop()

M = _mtime(settings.ITEMS_FILE)
ALL = load_items(M)
REL_ALL = A.relevant(ALL)
FUNNEL = load_json(str(settings.FUNNEL_FILE), _mtime(settings.FUNNEL_FILE)) or {}
INSIGHTS = load_json(str(settings.INSIGHTS_FILE), _mtime(settings.INSIGHTS_FILE)) or {}
CLUSTERS = load_json(str(settings.CLUSTERS_FILE), _mtime(settings.CLUSTERS_FILE)) or {}
REPORT = load_json(str(settings.EVALS / "report.json"), _mtime(settings.EVALS / "report.json"))
APPS = ["INDmoney", "Groww", "Zerodha", "Other or unknown"]
import datetime as _dt  # noqa: E402

AS_OF = _dt.datetime.fromtimestamp(M).strftime("%d %b %Y")
N_SOURCES = REL_ALL["source"].nunique()
GROUND = float(REL_ALL["quote_grounded"].mean()) if len(REL_ALL) else 0.0
AGREE = (REPORT or {}).get("fields", {}).get("is_relevant", {}).get("agreement") if REPORT and not REPORT.get("demo") else None
GOLD_N = (REPORT or {}).get("golden_n")


def trust_strip(in_view: int | None = None):
    items = [("", f"Data as of <b>{AS_OF}</b>", "When the evidence was last coded"),
             ("", f"<b>{len(REL_ALL):,}</b> real SIP experiences from <b>{N_SOURCES}</b> sources",
              "Public posts kept after cleaning and AI relevance checks"),
             ("" if GROUND >= .9 else "warn", f"<b>{GROUND:.0%}</b> quotes verified word for word",
              "Every evidence quote is matched against the original post")]
    if AGREE is not None:
        items.append(("" if AGREE >= .75 else "warn", f"AI agrees with my hand labels <b>{AGREE:.0%}</b>",
                      f"Relevance agreement on {GOLD_N} hand-labelled posts. See Quality checks."))
    else:
        items.append(("warn", "Hand-label check <b>pending</b>", "Run the golden-set evaluation to show agreement"))
    if in_view is not None and in_view != len(REL_ALL):
        items.insert(0, ("", f"<b>{in_view:,}</b> posts match your filters", "Change them with the Filters button"))
    st.markdown('<div class="trust">' + "".join(f'<span title="{S.e(t)}"><i class="{k}"></i>{h}</span>' for k, h, t in items)
                + "</div>", unsafe_allow_html=True)


def footer():
    st.markdown(f'<div class="foot"><span><b>SIP Discovery Engine</b>, INDmoney growth research</span>'
                f'<span>Evidence: public posts, {AS_OF}</span><span>AI: Gemini free tier, checked against hand labels</span>'
                '<span><b>Not investment advice.</b> This tool describes investor experiences and never recommends funds.</span></div>',
                unsafe_allow_html=True)


# =================================================================== filters + header
def filters_ui():
    srcs = sorted(REL_ALL["source"].unique())
    langs = sorted(REL_ALL["lang"].dropna().unique())
    months = sorted({d[:7] for d in REL_ALL["date"] if len(d) >= 7})
    st.multiselect("Sources", srcs, default=srcs, key="f_src")
    st.multiselect("App mentioned", APPS, default=APPS, key="f_app")
    st.multiselect("Language", langs, default=langs, key="f_lang")
    c1, c2 = st.columns(2)
    c1.toggle("Mutual-fund SIPs only", value=True, key="f_mf")
    c2.toggle("First-time investors only", value=False, key="f_first")
    if len(months) > 1:
        st.select_slider("Period", options=months, value=(months[0], months[-1]), key="f_months")


def filtered() -> pd.DataFrame:
    ss = st.session_state
    D = REL_ALL
    D = D[D["source"].isin(ss.get("f_src", D["source"].unique()))]
    D = D[D["app"].where(D["app"].isin(APPS[:3]), "Other or unknown").isin(ss.get("f_app", APPS))]
    if ss.get("f_lang"):
        D = D[D["lang"].isin(ss["f_lang"])]
    if ss.get("f_mf", True):
        D = A.mf_only(D)
    if ss.get("f_first"):
        D = D[D["first_time_investor"] == "yes"]
    if ss.get("f_months"):
        a, b = ss["f_months"]
        dm = D["date"].str[:7]
        D = D[(dm == "") | ((dm >= a) & (dm <= b))]
    return D


def header(title: str, sub: str, show_filters: bool = True, eyebrow: str = "SIP continuation research") -> pd.DataFrame:
    with st.container(key="band"):
        if settings.is_demo():
            st.markdown('<div class="demo"><b>Demo data.</b> Synthetic posts for previewing the layout. '
                        'Run <code>python pipeline.py all</code> for real evidence.</div>', unsafe_allow_html=True)
        left, right = st.columns([5, 1.2], vertical_alignment="bottom")
        left.markdown(f'<div class="phead"><div class="eyebrow">{S.e(eyebrow)}</div><h1>{S.e(title)}</h1>'
                      f'<p>{S.e(sub)}</p></div>', unsafe_allow_html=True)
        if show_filters:
            with right.popover("Filters", icon=":material/tune:", width="stretch"):
                filters_ui()
        D = filtered()
        trust_strip(len(D) if show_filters else None)
    return D


def quotes(df: pd.DataFrame, n: int = 4, theme: str | None = None):
    q = A.quotes(df, n, theme=theme) if theme else A.quotes(df, n)
    if theme and len(q) < n:
        q = pd.concat([q, A.quotes(df[~df["id"].isin(q["id"])], n - len(q))])
    q = q.drop_duplicates("evidence_quote")
    if q.empty:
        st.caption("No verified quotes in this view.")
    for _, r in q.iterrows():
        src = f"{r['source']}, {r['app']}, {LABELS.get(r['primary_theme'], r['primary_theme'])}"
        st.markdown(S.quote_html(r["evidence_quote"], src), unsafe_allow_html=True)


def plot(fig):
    st.plotly_chart(fig, width="stretch", config=CFG)


# =================================================================== FinBot
def stats_summary(D: pd.DataFrame) -> str:
    opp = A.opportunity_table(D)
    jl = A.journey_leaks(D)
    lines = [f"Relevant posts in view: {len(D)} (sources: {D['source'].value_counts().to_dict()})",
             "Opportunity scores (theme: posts, share, % tied to drop-off, score): " +
             "; ".join(f"{r.label}: {r.items}, {r.share:.0%}, {r.drop_link:.0%}, {r.score:.0f}" for r in opp.head(10).itertuples()),
             "Drop-off moments by stage: " + "; ".join(f"{r.label}: {r.drops}" for r in jl.itertuples()),
             "Events: " + "; ".join(f"{k}: {v}" for k, v in D["event"].value_counts().head(8).items())]
    return "\n".join(lines)


def finbot_answer(q: str) -> dict:
    if not llm.available():
        return {"status": "error", "answer": "FinBot needs a free GEMINI_API_KEY in the app's secrets.", "cited": []}
    if st.session_state["calls"] >= MAX_CALLS:
        return {"status": "error", "answer": f"FinBot has answered {MAX_CALLS} questions this visit, the limit "
                "that keeps the free quota available for everyone. Reload later to ask more.", "cited": []}
    st.session_state["calls"] += 1
    try:
        res = rag.ask(build_index(M), q, stats=stats_summary(filtered()))
    except llm.QuotaExhausted:
        return {"status": "error", "answer": "Today's free AI quota is used up. Try again tomorrow.", "cited": []}
    except Exception as ex:  # noqa: BLE001
        return {"status": "error", "answer": f"FinBot couldn't answer: {ex}", "cited": []}
    hits = res.get("hits")
    cited = []
    if hits is not None and len(hits) and res.get("cited"):
        for _, r in hits[hits["id"].isin(res["cited"])].iterrows():
            cited.append({"id": r["id"], "source": r["source"], "text": str(r["full_text"])[:320],
                          "theme": LABELS.get(r["primary_theme"], r["primary_theme"])})
    return {"status": res["status"], "answer": res["answer"], "confidence": res.get("confidence", ""),
            "cited": cited, "dropped": res.get("dropped", 0), "n_hits": int(len(hits)) if hits is not None else 0}


def save_feedback(q: str, a: dict, val):
    try:
        import csv

        path = settings.DATA / "finbot_feedback.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["when", "question", "status", "helpful"])
            w.writerow([_dt.datetime.now().isoformat(timespec="seconds"), q, a.get("status"), val])
    except Exception:  # noqa: BLE001
        pass


STATUS = {"ok": ("Answered from the evidence", ""), "declined": ("Declined: investment advice", "warn"),
          "no_evidence": ("Not enough evidence", "warn"), "fallback": ("Evidence unclear, no guess made", "warn"),
          "error": ("Not answered", "bad")}


def render_answer(a: dict):
    label, kind = STATUS.get(a["status"], ("", ""))
    badges = S.chip(label, kind)
    if a.get("confidence") and a["status"] == "ok":
        badges += " " + S.chip(f"{a['confidence']} confidence", "gray")
    if a.get("dropped"):
        badges += " " + S.chip(f"{a['dropped']} unverified citation(s) removed", "warn")
    st.markdown(a["answer"])
    if a.get("n_hits"):
        badges += " " + S.chip(f"based on {a['n_hits']} matching posts", "gray")
    st.markdown(badges, unsafe_allow_html=True)
    if a.get("cited"):
        with st.expander(f"Sources ({len(a['cited'])})"):
            for c in a["cited"]:
                st.markdown(S.quote_html(c["text"], f"{c['source']}, {c['theme']}, {c['id']}"), unsafe_allow_html=True)


def ask(q: str):
    st.session_state["fb_history"].append({"role": "user", "text": q})
    with st.spinner("FinBot is reading the evidence"):
        a = finbot_answer(q)
    st.session_state["fb_history"].append({"role": "assistant", **a})


def history(limit: int | None = None, fb: bool = False):
    hist = st.session_state["fb_history"]
    start = max(0, len(hist) - limit) if limit else 0
    for i in range(start, len(hist)):
        m = hist[i]
        if m["role"] == "user":
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(m["text"])
        else:
            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                render_answer(m)
                if fb and m["status"] in ("ok", "fallback", "no_evidence"):
                    val = st.feedback("thumbs", key=f"fbk_{i}")
                    if val is not None and not m.get("rated"):
                        m["rated"] = True
                        save_feedback(hist[i - 1]["text"] if i else "", m, "yes" if val == 1 else "no")
                        st.caption("Thanks. Feedback helps check FinBot's answers.")


@st.dialog("FinBot", width="large")
def finbot_dialog():
    st.markdown('<p class="note">Ask anything about the investor conversations. Answers cite real posts; '
                'it never gives investment advice.</p>', unsafe_allow_html=True)
    if not st.session_state["fb_history"]:
        cols = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            if cols[i % 2].button(s, key=f"dlg_s{i}", width="stretch"):
                ask(s)
    history(limit=6)
    with st.form("dlg_form", clear_on_submit=True, border=False):
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        q = c1.text_input("Your question", placeholder="e.g. Why do people pause and not restart?", label_visibility="collapsed")
        if c2.form_submit_button("Ask", type="primary", width="stretch") and q.strip():
            ask(q.strip())
            st.rerun(scope="fragment")


# =================================================================== pages
def page_home():
    D = header("Why first-time investors keep or drop their SIPs",
               "Thousands of public reviews and comments, coded by AI and checked against human labels.",
               eyebrow="SIP continuation intelligence")
    raw_n = FUNNEL.get("funnel", {}).get("raw", len(ALL))
    ground = GROUND
    if not st.session_state.get("welcomed"):
        with S.card("New here? What this is, in 30 seconds", key="card_welcome"):
            st.markdown('<div class="welcome">'
                        '<div><b>What it is</b><span>A research tool that reads public reviews and comments about SIPs '
                        'and shows why first-time investors keep or stop investing.</span></div>'
                        '<div><b>How to trust it</b><span>Every number links back to real posts. Quotes are checked word for '
                        'word, and the AI is scored against human labels.</span></div>'
                        '<div><b>What it is not</b><span>Not investment advice. It never recommends funds or tells anyone '
                        'whether to start or stop a SIP.</span></div></div>', unsafe_allow_html=True)
            if st.button("Got it", key="welcome_ok", icon=":material/check:"):
                st.session_state["welcomed"] = True
                st.rerun()
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        with st.container(key="hero"):
            st.markdown(f'<div class="hero-k">FinBot</div><div class="hero-h">Ask the evidence anything</div>'
                        f'<div class="hero-p">Answers come from {len(REL_ALL):,} coded investor posts and the numbers below, '
                        'with every source cited. No investment advice, ever.</div>', unsafe_allow_html=True)
            with st.form("home_ask", clear_on_submit=True, border=False):
                c1, c2 = st.columns([3.2, 1.3], vertical_alignment="bottom")
                q = c1.text_input("Question", placeholder="Why do first-time investors stop their SIP?",
                                  label_visibility="collapsed")
                go_ = c2.form_submit_button("Ask FinBot", type="primary", width="stretch")
            pick = st.pills("Try", SUGGESTIONS, label_visibility="collapsed", key="home_pills")
            if (go_ and q.strip()) or pick:
                st.session_state["pending_q"] = q.strip() if go_ and q.strip() else pick
                st.switch_page(PAGES["finbot"])
    with right:
        rel_rate = len(REL_ALL) / max(len(ALL), 1)
        st.markdown(f"""<div class="kpis" style="height:100%">
          <div class="kpi"><div class="l">Posts collected</div><div class="v">{raw_n:,}</div><div class="d" style="color:#626977">4 public sources</div></div>
          <div class="kpi"><div class="l">Coded by AI</div><div class="v">{len(ALL):,}</div><div class="d" style="color:#626977">after cleaning</div></div>
          <div class="kpi"><div class="l">Real SIP experiences</div><div class="v">{len(REL_ALL):,}</div><div class="d">{rel_rate:.0%} of coded posts</div></div>
          <div class="kpi"><div class="l">Quotes verified</div><div class="v">{ground:.0%}</div><div class="d">word for word</div></div>
        </div><div class="basis">Play Store, App Store, TradingQ&amp;A and YouTube, coded on {AS_OF}.</div>""",
                    unsafe_allow_html=True)

    st.write("")
    jl = A.journey_leaks(D)
    peak = jl.loc[jl["drops"].idxmax()] if jl["drops"].sum() else None
    mx = max(jl["drops"].max(), 1)
    drops = int(A.is_drop(D).sum())
    unclear = int((A.is_drop(D) & (D["journey_stage"] == "unclear")).sum())
    head = "No drop-off moments in this view"
    if peak is not None:
        head = f"Most drop-off moments land in {peak['label'].lower()}"
        if peak["top_theme"]:
            head += f", led by {LABELS.get(peak['top_theme'], '').lower()}"
    with S.card(head, f"{drops:,} posts describe a pause, stop, skip, reduction, failed instalment or a considered stop. "
                      f"Placed by how far into the SIP it happened; {unclear:,} don't say when and are left out. "
                      "Under each stage: its most common root cause."):
        cols = "".join(f'<div class="col"><div class="n">{r.drops}</div><div class="bar{" peak" if peak is not None and r.stage == peak["stage"] else ""}" '
                       f'style="height:{max(r.drops / mx * 140, 3):.0f}px"></div></div>' for r in jl.itertuples())
        labs = "".join(f"<div><b>{S.e(r.label)}</b>{S.e(LABELS.get(r.top_theme, ''))}</div>" for r in jl.itertuples())
        st.markdown(f'<div class="strip">{cols}</div><div class="labels">{labs}</div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        with S.card("Biggest opportunities", "Frequency × severity × link to drop-off × addressability without advice or fear"):
            opp = A.opportunity_table(D)
            top = max(opp["score"].max(), 1e-9)
            st.markdown("".join(
                f'<div class="opp"><div class="rank">{i + 1}</div><div><div class="name">{S.e(r.label)}</div>'
                f'<div class="meta">{r.items} posts, {r.drop_link:.0%} tied to a drop-off, {S.e(A.BRANCH_LABEL.get(r.branch, r.branch))}</div></div>'
                f'<div><div class="meter"><span style="width:{r.score / top * 100:.0f}%"></span></div>'
                f'<div class="score">score {r.score:.0f}</div></div></div>' for i, r in enumerate(opp.head(5).itertuples())),
                unsafe_allow_html=True)
    with c2:
        with S.card("What FinBot found", "Pre-computed findings on why people stop, with checked citations"):
            ins = INSIGHTS.get("why_stop") if not settings.is_demo() else None
            if ins and ins.get("findings"):
                st.markdown("".join(f'<div class="finding"><h4>{S.e(f["headline"])}</h4><p>{S.e(f["detail"])}</p></div>'
                                    for f in ins["findings"][:3]), unsafe_allow_html=True)
            else:
                quotes(D[A.is_drop(D)], 3)


def page_finbot():
    header("FinBot", "Ask questions in plain language. FinBot reads the coded posts and the dataset numbers, cites its "
                     "sources, says when the evidence is thin, and declines investment-advice questions.", show_filters=True)
    pending = st.session_state.pop("pending_q", None)
    if pending:
        ask(pending)
    if not st.session_state["fb_history"]:
        with S.card(key="card_fbhello"):
            st.markdown('<div class="fb-hello"><div class="fb-ico"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
                        'stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
                        '<rect x="4" y="7" width="16" height="12" rx="3"></rect><path d="M12 3v4"></path><circle cx="9" cy="13" r="1"></circle>'
                        '<circle cx="15" cy="13" r="1"></circle></svg></div><div><div class="ctitle">Hi, I\'m FinBot</div>'
                        f'<div class="csub" style="margin:0">I\'ve read {len(REL_ALL):,} investor posts about SIPs. Ask me what they say, '
                        'or start with one of these.</div></div></div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, s in enumerate(SUGGESTIONS):
                if cols[i % 2].button(s, key=f"fb_s{i}", width="stretch", icon=":material/arrow_outward:"):
                    ask(s)
                    st.rerun()
    history(fb=True)
    c1, c2 = st.columns([6, 1])
    c1.caption(f"{st.session_state['calls']} of {MAX_CALLS} questions used this visit. Filters apply to the numbers FinBot sees. "
               "FinBot describes what investors said; it is not investment advice.")
    if st.session_state["fb_history"] and c2.button("Clear chat", icon=":material/delete:", width="stretch"):
        st.session_state["fb_history"] = []
        st.rerun()
    q = st.chat_input("Ask FinBot about SIP drop-offs, failures, expectations...")
    if q:
        ask(q)
        st.rerun()


def page_opportunities():
    D = header("Opportunities", "Which problems, if solved, would move 12-month SIP continuation most.")
    tax = settings.taxonomy_cfg()["themes"]
    with st.expander("How the score works, and adjust addressability", icon=":material/tune:"):
        st.markdown('<p class="note"><b>Frequency</b> (0–5): share of posts mentioning the theme, relative to the most common. '
                    '<b>Severity</b> (1–5): average impact the AI coded. <b>Metric link</b> (0–3): share tied to a pause, stop, '
                    'failure or considered stop. <b>Addressability</b> (0–3): can INDmoney act without recommending funds or '
                    'using fear, FOMO or urgency. Score = product of the four, scaled to 100.</p>', unsafe_allow_html=True)
        cols = st.columns(3)
        override = {t["id"]: cols[i % 3].slider(t["label"], 0, 3, int(t["addressability"]), key=f"ad_{t['id']}", help=t["rationale"])
                    for i, t in enumerate([t for t in tax if t["branch"] != "none"])}
    opp = A.opportunity_table(D, override)
    opp["branch_label"] = opp["branch"].map(A.BRANCH_LABEL)
    opp["share_pct"], opp["drop_pct"] = opp["share"] * 100, opp["drop_link"] * 100
    with S.card("Opportunity ranking"):
        st.dataframe(opp[["label", "branch_label", "items", "share_pct", "severity", "drop_pct", "addressability", "score"]],
                     hide_index=True, width="stretch", column_config={
                         "label": st.column_config.TextColumn("Theme", width="large"), "branch_label": "Metric-tree branch",
                         "items": "Posts", "share_pct": st.column_config.NumberColumn("Share", format="%.0f%%"),
                         "severity": st.column_config.NumberColumn("Severity", format="%.1f"),
                         "drop_pct": st.column_config.NumberColumn("Tied to drop-off", format="%.0f%%"),
                         "addressability": "Addressability",
                         "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=float(max(opp["score"].max(), 1)), format="%.0f")})
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with S.card("What drives each score", "Each bar is that component as a share of its maximum"):
            plot(charts.opp_components(opp))
    with c2:
        with S.card("Opportunity by metric-tree branch", "Sum of theme scores per branch of the continuation metric"):
            plot(charts.branch_bar(opp))
    with S.card("Look inside a theme"):
        pick = st.selectbox("Theme", opp["label"].tolist(), label_visibility="collapsed")
        tid = opp.loc[opp["label"] == pick, "theme"].iloc[0]
        sub = D[D["themes"].map(lambda v: tid in v)]
        t = next(t for t in tax if t["id"] == tid)
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown(f'<p class="note"><b>Definition.</b> {S.e(t["definition"])}<br><b>Addressability {t["addressability"]}.</b> '
                        f'{S.e(t["rationale"])}</p>', unsafe_allow_html=True)
            ev = A.dist(sub, "event")
            if len(ev):
                ev["event"] = ev["event"].str.replace("_", " ")
                plot(charts.hbar(ev.head(6), "event", "share", S.SAFFRON, height=240))
            needs = sub["unmet_need"].replace("", np.nan).dropna().value_counts().head(5)
            if len(needs):
                st.markdown("**Unmet needs, in users' words**\n\n" + "\n".join(f"- {S.e(k)} ({v})" for k, v in needs.items()))
        with b:
            quotes(sub, 4, theme=tid)
            works = sub["workaround"].replace("", np.nan).dropna().value_counts().head(4)
            if len(works):
                st.markdown("**What people do instead**\n\n" + "\n".join(f"- {S.e(k)} ({v})" for k, v in works.items()))


QS = {"why_start": "Why do people start a SIP, and what do they expect?", "what_changes": "What changes in the months after starting?",
      "why_stop": "What leads investors to pause, reduce or stop?", "what_check": "What do investors check, and how often?",
      "confusion": "What confuses or worries them in the app?", "outside_app": "What do they do outside the app when unsure?",
      "continuers": "What separates people who continue from those who drop off?", "segments": "How do patterns differ across segments?",
      "unmet_needs": "Which unmet needs come up again and again?"}


def page_questions():
    D = header("Research questions", "The nine questions from the brief, each answered with numbers, findings and real quotes.")
    key = st.selectbox("Question", list(QS), format_func=QS.get, label_visibility="collapsed")
    ins = INSIGHTS.get(key) if not settings.is_demo() else None
    drop = D[A.is_drop(D)]

    def chart_of(col, multi=False, color=S.NAVY, title="", data=None, n=8):
        base = D if data is None else data
        d = A.dist(base, col, multi).head(n)
        with S.card(title):
            if len(d):
                said = base[col].map(lambda v: len(A.as_list(v)) > 0 and not set(A.as_list(v)) <= {"unclear", "none"}).mean()
                st.caption(f"{said:.0%} of {len(base):,} posts say something about this")
                d[col] = d[col].astype(str).str.replace("_", " ")
                plot(charts.hbar(d, col, "share", color))
            else:
                st.caption("Not enough signal yet.")

    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        if key == "why_start":
            a, b = st.columns(2)
            with a:
                chart_of("why_started", title="Reason for starting")
            with b:
                chart_of("expectation", color=S.BLUE, title="Expectation at the start")
            chart_of("influencer", color=S.BLUE, title="Who influenced them")
        elif key == "what_changes":
            ev = D[D["journey_stage"] != "unclear"]
            with S.card("What happens to the SIP at each stage", "Share of posts in each stage"):
                if len(ev):
                    ct = pd.crosstab(ev["event"], ev["journey_stage"], normalize="columns").reset_index().melt(id_vars="event", value_name="share")
                    ct = ct[ct["journey_stage"].isin(A.STAGE_ORDER)]
                    ct["journey_stage"] = pd.Categorical(ct["journey_stage"].map(A.STAGE_LABEL),
                                                         [A.STAGE_LABEL[s] for s in A.STAGE_ORDER if s in set(ev["journey_stage"])])
                    ct["event"] = ct["event"].str.replace("_", " ")
                    plot(charts.heatmap(ct.sort_values("journey_stage"), "event", "journey_stage"))
        elif key == "why_stop":
            d = A.dist(drop, "primary_theme")
            d["primary_theme"] = d["primary_theme"].map(LABELS)
            with S.card(f"Root cause behind {len(drop):,} drop-off moments"):
                plot(charts.hbar(d, "primary_theme", "share", S.SAFFRON))
                mkt = drop["themes"].map(lambda v: "market_fall_trigger" in v)
                if mkt.any():
                    st.caption(f"Market falls appear in {mkt.mean():.0%} of drop-off posts; in "
                               f"{(drop[mkt]['primary_theme'] != 'market_fall_trigger').mean():.0%} of those another root cause was coded.")
        elif key == "what_check":
            a, b = st.columns(2)
            with a:
                chart_of("checks", True, title="What they look at")
            with b:
                chart_of("check_frequency", color=S.BLUE, title="How often they check")
        elif key == "confusion":
            a, b = st.columns(2)
            with a:
                chart_of("confusions", True, title="What confuses them")
            with b:
                chart_of("emotion", color=S.BLUE, title="How they feel")
        elif key == "outside_app":
            chart_of("outside_app_actions", True, color=S.BLUE, title="Where they go when unsure")
        elif key == "continuers":
            cvd, nc, nd = A.continuers_vs_drop(D)
            cvd = cvd[(cvd["continuers"] + cvd["drop_offs"]) > 0]
            with S.card(f"Themes: {nc:,} who kept going vs {nd:,} who dropped off"):
                plot(charts.grouped(cvd, "theme", ["continuers", "drop_offs"], ["Kept going", "Dropped off"]))
            with S.card("What helps people keep going"):
                plot(charts.grouped(A.driver_compare(D), "driver", ["continuers", "drop_offs"], ["Kept going", "Dropped off"]))
        elif key == "segments":
            a, b = st.columns(2)
            with a:
                chart_of("income", title="Income type mentioned")
            with b:
                chart_of("age_band", color=S.BLUE, title="Age band mentioned")
            st.caption("Open Segments for the full comparison.")
        elif key == "unmet_needs":
            cl = CLUSTERS.get("clusters", [])
            with S.card(f"{len(cl)} needs found bottom-up", f"Clustered from {CLUSTERS.get('n', 0):,} unmet-need statements"):
                if cl:
                    plot(charts.hbar(pd.DataFrame(cl).head(12), "name", "share", S.NAVY))
                    for c in cl[:6]:
                        with st.expander(f"{c['name']} ({c['items']})"):
                            st.write(c.get("description", ""))
                            st.markdown("\n".join(f"- {S.e(x)}" for x in c.get("examples", [])))
                else:
                    st.caption("Run `python pipeline.py themes`.")
    with c2:
        with S.card("Findings", "Written from the computed numbers; every cited post checked"):
            if ins and ins.get("findings"):
                st.markdown("".join(f'<div class="finding"><h4>{S.e(f["headline"])}</h4><p>{S.e(f["detail"])}</p></div>'
                                    for f in ins["findings"]), unsafe_allow_html=True)
            else:
                st.caption("Run `python pipeline.py insights` to add findings.")
        with S.card("In their words"):
            pool = {"why_stop": drop, "what_changes": drop, "what_check": D[D["checks"].map(len) > 0],
                    "confusion": D[D["confusions"].map(len) > 0], "outside_app": D[D["outside_app_actions"].map(len) > 0],
                    "why_start": D[D["why_started"] != "unclear"]}.get(key, D)
            quotes(pool, 4)


def page_segments():
    D = header("Segments", "How the problems differ by who is talking, where and in which language.")
    dims = {"income": "Income", "age_band": "Age", "influencer": "Influence", "money_source": "Money source",
            "why_started": "Reason to start", "source": "Platform", "app": "App", "lang": "Language"}
    c1, c2 = st.columns([4, 1.4], vertical_alignment="bottom")
    dim = c1.segmented_control("Compare by", list(dims), format_func=dims.get, default="income", key="seg_dim") or "income"
    min_n = c2.slider("Minimum posts per segment", 3, 50, 8)
    tb = A.theme_by(D, dim, min_n)
    with S.card("Share of each segment's posts mentioning a theme",
                "Segments come only from what people said about themselves; read small n with care"):
        if len(tb):
            tb[dim] = tb[dim].astype(str).str.replace("_", " ") + tb["n"].map(lambda n: f" (n={n})")
            plot(charts.heatmap(tb, "theme", dim))
        else:
            st.info("No segment reaches the minimum size. Lower it or widen the filters.")
    with S.card("Top themes over time", "Monthly share of posts, months with 5+ posts"):
        tr = A.monthly_trend(D, A.opportunity_table(D).head(4)["theme"].tolist())
        if len(tr) and tr["month"].nunique() > 1:
            plot(charts.trend(tr[tr["n"] >= 5], LABELS))
        else:
            st.caption("Not enough dated posts for a trend.")


def page_evidence():
    D = header("Evidence", "Every coded post, searchable, with the AI's full record.")
    c1, c2, c3 = st.columns([2.2, 1, 1])
    q = c1.text_input("Search", placeholder="mandate, salary date, XIRR, band kar diya", label_visibility="collapsed", icon=":material/search:")
    th = c2.selectbox("Theme", ["All"] + list(LABELS), format_func=lambda k: "All themes" if k == "All" else LABELS[k], label_visibility="collapsed")
    evs = c3.selectbox("Event", ["All"] + schema.EVENTS, format_func=lambda k: "All events" if k == "All" else k.replace("_", " "), label_visibility="collapsed")
    E = D
    if q:
        E = E[E["full_text"].str.contains(q, case=False, regex=False) | E["unmet_need"].str.contains(q, case=False, regex=False)]
    if th != "All":
        E = E[E["themes"].map(lambda v: th in v)]
    if evs != "All":
        E = E[E["event"] == evs]
    with S.card(f"{len(E):,} posts"):
        view = E[["id", "source", "app", "date", "event", "primary_theme", "severity", "evidence_quote", "unmet_need", "url"]].copy()
        view["primary_theme"] = view["primary_theme"].map(lambda k: LABELS.get(k, k))
        view["event"] = view["event"].str.replace("_", " ")
        st.dataframe(view, hide_index=True, width="stretch", height=430, column_config={
            "url": st.column_config.LinkColumn("Link", display_text="open"), "primary_theme": "Root cause",
            "evidence_quote": st.column_config.TextColumn("Evidence quote", width="large"),
            "unmet_need": st.column_config.TextColumn("Unmet need", width="medium"),
            "severity": st.column_config.NumberColumn("Sev.", format="%d")})
        a, b = st.columns([3, 1], vertical_alignment="bottom")
        pick = a.selectbox("Open a record", E["id"].tolist()[:500]) if len(E) else None
        b.download_button("Download CSV", E.drop(columns=LIST_COLS).to_csv(index=False).encode("utf-8"),
                          "sip_evidence.csv", "text/csv", width="stretch", icon=":material/download:")
    if pick:
        r = E[E["id"] == pick].iloc[0]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            with S.card("Original post"):
                st.write(r["full_text"])
                st.markdown(S.chip("quote verified") if r["quote_grounded"] else S.chip("quote not found in text", "bad"),
                            unsafe_allow_html=True)
        with c2:
            with S.card("What the AI coded"):
                st.json({k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()
                         if k not in ("full_text", "text", "title", "context", "kw_hit")}, expanded=False)


def page_extractor():
    header("Try the extractor", "Paste any review, comment or post in English, Hindi or Hinglish and see how the engine codes it.",
           show_filters=False)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        with S.card("Post"):
            txt = st.text_area("Post text", height=170, label_visibility="collapsed",
                               value="Started my first SIP in Jan. Salary comes on 30th but SIP date is 2nd, two times it failed "
                                     "and bank cut 590 rs. App only says failed. Now paused.")
            src = st.selectbox("Where is it from?", ["playstore", "appstore", "reddit", "youtube", "tradingqna", "other"])
            go_ = st.button("Code this post", type="primary", icon=":material/auto_awesome:", width="stretch")
    with c2:
        with S.card("Result"):
            if not go_:
                st.caption("The coded record appears here.")
            elif not llm.available():
                st.info("Needs a free GEMINI_API_KEY in the app's secrets.")
            elif st.session_state["calls"] >= MAX_CALLS:
                st.info(f"{MAX_CALLS} AI calls used this visit.")
            elif txt.strip():
                st.session_state["calls"] += 1
                from engine import extract

                with st.spinner("Coding"):
                    try:
                        rec = extract.extract_one({"id": "LIVE", "source": src, "platform": "unknown", "date": "",
                                                   "context": "", "rating": None, "full_text": txt[:4000]})
                    except Exception as ex:  # noqa: BLE001
                        rec = None
                        st.error(f"The AI call failed: {ex}")
                if rec:
                    st.markdown((S.chip("relevant") if rec["is_relevant"] else S.chip("not a SIP experience", "warn")) + " " +
                                (S.chip("quote verified") if rec["quote_grounded"] else S.chip("quote not found", "bad")),
                                unsafe_allow_html=True)
                    st.markdown(f"**Root cause:** {LABELS.get(rec['primary_theme'], rec['primary_theme'])}  \n"
                                f"**What happened:** {rec['event'].replace('_', ' ')}, {A.STAGE_LABEL.get(rec['journey_stage'], '')}  \n"
                                f"**Unmet need:** {rec['unmet_need'] or '—'}  \n**Workaround:** {rec['workaround'] or '—'}")
                    st.markdown(S.quote_html(rec["evidence_quote"], f"evidence quote, confidence {rec['confidence']:.0%}"), unsafe_allow_html=True)
                    with st.expander("Full record"):
                        st.json(rec)


def page_survey():
    header("Survey", "Responses from the first-SIP survey: who kept going, who didn't, and what differs.", show_filters=False)
    up = st.file_uploader("Upload the responses CSV (Google Sheet > File > Download > CSV). Stays in this session only.", type="csv")
    sv = survey.load(up) if up else survey.load()
    if sv is None or sv.empty:
        st.info("No survey file yet. Upload the CSV, or commit it as data/survey/responses.csv.")
        return
    fl = survey.flags(sv)
    st.markdown(f"""<div class="kpis row" style="margin-bottom:16px">
      <div class="kpi"><div class="v">{len(sv)}</div><div class="l">responses</div></div>
      <div class="kpi"><div class="v">{(fl['group'] == 'Kept going').sum()}</div><div class="l">kept going</div></div>
      <div class="kpi"><div class="v">{(fl['group'] != 'Kept going').sum()}</div><div class="l">paused, stopped or reduced</div></div>
      <div class="kpi"><div class="v">{int(fl['had_issue'].sum())}</div><div class="l">had a payment or SIP issue</div></div>
    </div>""", unsafe_allow_html=True)
    if len(sv) < 30:
        st.caption("Small sample: read percentages as directional. Contact details are removed before display.")
    qcols = [c for c in (survey.question_cols(sv) or list(sv.columns)) if c.lower() != "timestamp"
             and not any(k in c.lower() for k in survey.FREE_TEXT)]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with S.card("Answers by question"):
            qc = st.selectbox("Question", qcols, label_visibility="collapsed")
            cnt = survey.counts(sv, qc)
            cnt["label_n"] = cnt.apply(lambda r: f"{r['share']:.0%} ({r['n']})", axis=1)
            plot(charts.survey_bar(cnt))
    with c2:
        with S.card("Kept going vs didn't"):
            names = [t for t, k in survey.CROSSTABS if survey.find_col(sv, *k)]
            if names:
                pick = st.selectbox("Compare", names, label_visibility="collapsed")
                plot(charts.survey_cross(survey.crosstab(sv, dict(survey.CROSSTABS)[pick])))
    ft = survey.find_col(sv, *survey.FREE_TEXT)
    if ft and sv[ft].notna().any():
        with S.card("Open-text answers"):
            for v in sv[ft].dropna().astype(str):
                st.markdown(S.quote_html(v, "survey respondent"), unsafe_allow_html=True)


KAPPA = [(0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"), (0.2, "fair"), (-1, "slight")]


def page_quality():
    header("Quality checks", "How far the engine can be trusted: agreement with hand labels, verified quotes and documented corrections.",
           show_filters=False)
    if not REPORT:
        with S.card("No evaluation yet", "Label a golden set and run the eval"):
            st.code("python pipeline.py golden-sample --n 20\n# label data/golden/golden_to_label.csv, save as golden_set.csv\n"
                    "python pipeline.py eval\npython pipeline.py theme-audit")
        return
    if REPORT.get("demo"):
        st.markdown('<div class="demo"><b>Demo numbers.</b> Run the eval on your golden set to replace them.</div>', unsafe_allow_html=True)
    f = REPORT["fields"]
    g = REPORT.get("grounding") or {}
    tiles = [("Relevance", f.get("is_relevant", {}).get("agreement")), ("Root cause", f.get("primary_theme", {}).get("agreement")),
             ("What happened", f.get("event", {}).get("agreement")), ("Quotes verified", g.get("relevant", g.get("all")))]
    st.markdown('<div class="kpis row" style="margin-bottom:16px">' + "".join(
        f'<div class="kpi"><div class="v">{"—" if v is None else f"{v:.0%}"}</div><div class="l">{S.e(l)}</div></div>' for l, v in tiles)
        + "</div>", unsafe_allow_html=True)
    rows = []
    for k, v in f.items():
        kp = v.get("kappa")
        rows.append({"Field": k.replace("_", " "), "Items": v["n"], "Scored on": v.get("scored_on", "all items"), "Agreement": v["agreement"],
                     "Lenient": v.get("lenient_agreement"), "Kappa": kp,
                     "Reading": next(lbl for t, lbl in KAPPA if kp >= t) if kp is not None else "n/a"})
    df = pd.DataFrame(rows)
    for c in ["Lenient", "Kappa"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    with S.card("Agreement with my labels", f"{REPORT['golden_n']} hand-labelled items, model {REPORT['model']}, {REPORT['generated']}. Kappa corrects for chance."):
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Agreement": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Lenient": st.column_config.NumberColumn("Lenient (theme in top 3)", format="percent"),
            "Kappa": st.column_config.NumberColumn("Cohen's kappa", format="%.2f")})
    hist = []
    for p in sorted(settings.EVALS.glob("report_*.json")) + [settings.EVALS / "report.json"]:
        r = load_json(str(p), _mtime(p))
        if r and not r.get("demo") and r.get("fields"):
            row = {"Version": r.get("prompt_label", "v1"), "Generated": r.get("generated")}
            row.update({k.replace("_", " "): v.get("agreement") for k, v in r["fields"].items()})
            hist.append(row)
    if len(hist) > 1:
        with S.card("Prompt versions compared", "Agreement with my labels before and after tightening the coding rules"):
            hd = pd.DataFrame(hist)
            st.dataframe(hd, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="percent") for c in hd.columns if c not in ("Version", "Generated")})
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with S.card("Where the AI and I disagreed"):
            dis = pd.DataFrame(REPORT.get("disagreements", []))
            if len(dis):
                st.dataframe(dis, hide_index=True, width="stretch", column_config={"text": st.column_config.TextColumn("Text", width="large")})
            else:
                st.caption("No disagreements.")
    with c2:
        with S.card("Places I rejected or corrected AI output"):
            corr = REPORT.get("corrections") or []
            if corr:
                st.dataframe(pd.DataFrame(corr), hide_index=True, width="stretch")
            else:
                st.caption("Add rows to data/golden/corrections.csv and rerun the eval.")
        if REPORT.get("theme_precision"):
            with S.card("Theme precision (spot-check)"):
                tp = pd.DataFrame(REPORT["theme_precision"])
                tp["primary_theme"] = tp["primary_theme"].map(lambda k: LABELS.get(k, k))
                st.dataframe(tp, hide_index=True, width="stretch", column_config={
                    "precision": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})


def page_method():
    header("How it works", "From public posts to scored opportunities, with checks at every step.", show_filters=False)
    steps = [("Collect", "Play Store, App Store, TradingQ&A and YouTube for INDmoney, Groww and Zerodha, plus survey open text."),
             ("Clean", "Personal details removed, short and old posts dropped, duplicates merged, English, Hindi and Hinglish detected."),
             ("Code with AI", "Gemini's free tier codes batches of posts into 25 fields; replies are checked against allowed values."),
             ("Verify", "Every evidence quote is matched word for word to its source; a golden set measures agreement."),
             ("Theme and score", "Fixed taxonomy plus bottom-up clusters; opportunity = frequency × severity × drop-off link × addressability."),
             ("Explain", "Findings and FinBot answers come from computed numbers and cited posts only.")]
    with S.card("Pipeline"):
        st.markdown('<div class="steps">' + "".join(f'<div class="step"><span class="i">{i + 1}</span><b>{S.e(a)}</b><span>{S.e(b)}</span></div>'
                                                   for i, (a, b) in enumerate(steps)) + "</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with S.card("Sources and volume"):
            raw = FUNNEL.get("raw_by_source") or {}
            if raw:
                vol = pd.DataFrame({"source": list(raw), "collected": list(raw.values())})
                vol["sent to AI"] = vol["source"].map(FUNNEL.get("by_source", {})).fillna(0).astype(int)
                vol["relevant"] = vol["source"].map(REL_ALL["source"].value_counts()).fillna(0).astype(int)
                st.dataframe(vol, hide_index=True, width="stretch")
            fn = FUNNEL.get("funnel", {})
            if fn:
                plot(charts.funnel(["Collected", "Long enough", "In date range", "De-duplicated", "About SIPs / MFs", "Coded", "Relevant"],
                                   [fn.get("raw", 0), fn.get("after_length_filter", 0), fn.get("after_date_filter", 0),
                                    fn.get("after_dedupe", 0), fn.get("after_keyword_filter", 0), len(ALL), len(REL_ALL)]))
    with c2:
        with S.card("Limits of this evidence"):
            st.markdown("""
- Public posts over-represent people with strong feelings; quiet drop-offs rarely post.
- Reddit is not included: its API now needs approval.
- Segment fields are filled only when people describe themselves, so most are unclear.
- Counts show where problems are talked about, not how often they happen in the product.
- This tool describes experiences. It never gives investment advice or recommends funds.
""")
    with S.card("Theme taxonomy and addressability"):
        st.dataframe(pd.DataFrame(settings.taxonomy_cfg()["themes"])[["label", "branch", "addressability", "definition", "rationale"]],
                     hide_index=True, width="stretch")
    with st.expander("The extraction prompt"):
        from engine import extract

        st.text(extract.system_prompt())


# =================================================================== navigation
PAGES = {
    "home": st.Page(page_home, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
    "finbot": st.Page(page_finbot, title="FinBot", icon=":material/smart_toy:", url_path="finbot"),
    "opp": st.Page(page_opportunities, title="Opportunities", icon=":material/trending_up:", url_path="opportunities"),
    "qs": st.Page(page_questions, title="Research questions", icon=":material/quiz:", url_path="questions"),
    "seg": st.Page(page_segments, title="Segments", icon=":material/pie_chart:", url_path="segments"),
    "ev": st.Page(page_evidence, title="Evidence", icon=":material/database:", url_path="evidence"),
    "ex": st.Page(page_extractor, title="Try the extractor", icon=":material/science:", url_path="extractor"),
    "sv": st.Page(page_survey, title="Survey", icon=":material/ballot:", url_path="survey"),
    "qa": st.Page(page_quality, title="Quality checks", icon=":material/verified:", url_path="quality"),
    "how": st.Page(page_method, title="How it works", icon=":material/account_tree:", url_path="method"),
}
nav = st.navigation({"Insights": [PAGES[k] for k in ("home", "finbot", "opp", "qs", "seg")],
                     "Data": [PAGES[k] for k in ("ev", "ex", "sv")],
                     "Trust": [PAGES[k] for k in ("qa", "how")]}, position="sidebar")
with st.sidebar:
    st.markdown('<div class="brand"><b>SIP Discovery Engine</b><span>INDmoney growth research</span></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="navfoot">{len(REL_ALL):,} investor posts, coded by AI and checked against hand labels. '
                'Describes experiences; not investment advice.</p>', unsafe_allow_html=True)
nav.run()
footer()
if nav.url_path != "finbot":
    with st.container(key="finbot_fab"):
        if st.button("Ask FinBot", icon=":material/auto_awesome:"):
            finbot_dialog()
