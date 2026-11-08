"""Visual system aligned with INDmoney's in-app UI language (colours sampled from app screenshots).

Navy header band with rounded bottom, light grey canvas, white rounded cards, green primary
buttons, mint CTA on navy, blue for links and active states, green up / red down, section titles
with a green accent bar. No INDmoney logo or wordmark: this is a research tool, not their product.
"""
from __future__ import annotations

from contextlib import contextmanager
from html import escape

import streamlit as st

# ---- tokens (sampled) -------------------------------------------------------
NAVY, NAVY_2, NAVY_LINE = "#223464", "#1D2C55", "#374778"
MINT, MINT_INK = "#21CE99", "#0E2440"
GREEN_BTN, GREEN = "#33A34D", "#0B975B"
BLUE = "#017AFF"
RED, RED_SOFT = "#D63B3B", "#F7E4E6"
PAPER, SURFACE, SURFACE_2, LINE = "#F5F5F7", "#FFFFFF", "#F6F7F9", "#E6E8EC"
INK, INK_NAVY, SLATE = "#1B1F27", "#1F2E5C", "#626977"
PILL_DARK = "#2D3750"
NOTE_BG = "#EAF4FF"
# legacy names used by charts/app
TEAL, TEAL_SOFT, SAFFRON, SAFFRON_SOFT, RUST, PLUM, NAV = GREEN, "#E3F4EA", RED, RED_SOFT, "#A8262B", "#6E59C7", NAVY
SEQ = [NAVY, GREEN, BLUE, RED, MINT, "#6E59C7", "#8A93A6", "#F2A93B"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select, [data-testid="stWidgetLabel"] {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }}
.stApp {{ background:{PAPER}; color:{INK}; }}
.block-container {{ padding: 0 2rem 6rem; max-width: 1320px; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ color:{INK}; letter-spacing:-0.01em; }}
p, li {{ line-height:1.55; }}

/* sidebar: light, blue active state */
[data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {LINE}; }}
[data-testid="stSidebarNav"] a {{ border-radius:12px; padding:9px 12px; margin:2px 8px; }}
[data-testid="stSidebarNav"] a:hover {{ background:{SURFACE_2}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background:#EAF2FF; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color:{BLUE}; font-weight:700; }}
[data-testid="stSidebarNav"] span {{ color:{SLATE}; font-size:.95rem; }}
[data-testid="stSidebarNavSeparator"] {{ border-color:{LINE}; }}
.brand {{ padding: 6px 10px 12px; }}
.brand b {{ display:block; color:{INK_NAVY}; font-size:1.02rem; font-weight:800; }}
.brand span {{ color:{SLATE}; font-size:.8rem; }}
.navfoot {{ color:{SLATE}; font-size:.8rem; line-height:1.5; padding:0 10px; }}

/* navy header band, rounded bottom (INDmoney home pattern) */
.st-key-band {{ background:{NAVY}; margin: 0 -2rem 22px; padding: 30px 2rem 22px;
  border-radius: 0 0 32px 32px; }}
.st-key-band .phead h1 {{ color:#FFFFFF; font-size:1.9rem; font-weight:700; margin:0; line-height:1.2; }}
.st-key-band .phead p {{ color:#C9D1E6; margin:6px 0 0; font-size:.98rem; max-width:760px; }}
.st-key-band .eyebrow {{ color:{MINT}; font-weight:600; font-size:.9rem; margin-bottom:6px; }}
.st-key-band [data-testid="stPopover"] button {{ background: rgba(255,255,255,.08); border:1px solid {NAVY_LINE};
  color:#FFFFFF; border-radius:12px; }}
.st-key-band [data-testid="stPopover"] button p {{ color:#FFFFFF; }}
.trust {{ display:flex; flex-wrap:wrap; gap:8px; margin: 16px 0 0; }}
.trust span {{ display:inline-flex; align-items:center; gap:7px; background:rgba(255,255,255,.07);
  border:1px solid {NAVY_LINE}; border-radius:999px; padding:6px 12px; font-size:.8rem; color:#C9D1E6; cursor:help; }}
.trust b {{ color:#FFFFFF; font-variant-numeric: tabular-nums; }}
.trust i {{ width:7px; height:7px; border-radius:50%; background:{MINT}; display:inline-block; }}
.trust i.warn {{ background:#F2A93B; }}
.demo {{ background:{NOTE_BG}; color:#0B4A8F; padding:10px 14px; border-radius:14px; font-size:.88rem; margin: 0 0 14px; }}
.st-key-band .demo {{ background:rgba(255,255,255,.1); color:#E6ECFA; }}

/* chips */
.chip {{ display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:999px; font-size:.8rem;
  font-weight:600; background:#E3F4EA; color:#066B3F; white-space:nowrap; }}
.chip.warn {{ background:#FFF1DB; color:#8A5300; }}
.chip.bad {{ background:{RED_SOFT}; color:#A8262B; }}
.chip.gray {{ background:#EEF0F4; color:{SLATE}; }}
.chip.blue {{ background:#EAF2FF; color:{BLUE}; }}

/* cards: keyed containers starting with card */
[class*="st-key-card"] {{ background:{SURFACE}; border:1px solid #EEF0F3; border-radius:22px; padding:22px 24px;
  box-shadow: 0 2px 12px rgba(16,24,40,.04); }}
.ctitle {{ position:relative; padding-left:14px; font-size:1.08rem; font-weight:700; color:{INK}; margin:0 0 2px; }}
.ctitle::before {{ content:""; position:absolute; left:-24px; top:2px; bottom:2px; width:5px; border-radius:0 4px 4px 0; background:{GREEN}; }}
.csub {{ font-size:.86rem; color:{SLATE}; margin:4px 0 14px; line-height:1.5; }}

/* hero (FinBot) card */
.st-key-hero {{ background:{SURFACE}; border:1px solid #EEF0F3; border-radius:24px; padding:26px 28px;
  box-shadow: 0 2px 12px rgba(16,24,40,.04); }}
.hero-k {{ font-size:.86rem; font-weight:700; color:{GREEN}; margin:0 0 6px; }}
.hero-h {{ font-size:1.65rem; font-weight:700; color:{INK_NAVY}; line-height:1.25; margin:0 0 8px; }}
.hero-p {{ color:{SLATE}; margin:0 0 16px; font-size:.95rem; max-width:620px; }}

/* KPI tiles (NIFTY-tile pattern) */
.kpis {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; }}
.kpis.row {{ grid-template-columns: repeat(4, minmax(0,1fr)); }}
.kpi {{ background:{SURFACE}; border:1px solid #EEF0F3; border-radius:18px; padding:16px 18px; box-shadow:0 2px 12px rgba(16,24,40,.04); }}
.kpi .l {{ font-size:.8rem; color:{SLATE}; font-weight:600; letter-spacing:.01em; }}
.kpi .v {{ font-size:1.6rem; font-weight:700; color:{INK_NAVY}; font-variant-numeric: tabular-nums; margin-top:6px; line-height:1.1; }}
.kpi .d {{ font-size:.8rem; color:{GREEN}; font-weight:600; margin-top:4px; }}
.kpi .d.neg {{ color:{RED}; }}
.basis {{ color:{SLATE}; font-size:.78rem; margin-top:8px; }}

/* leak strip */
.strip {{ display:grid; grid-template-columns: repeat(7, minmax(0,1fr)); gap:10px; align-items:end; height:190px; border-bottom:2px solid {INK_NAVY}; }}
.col {{ display:flex; flex-direction:column; justify-content:flex-end; height:100%; }}
.bar {{ background:#F1A7A9; border-radius:10px 10px 0 0; min-height:3px; }}
.bar.peak {{ background:{RED}; }}
.n {{ font-weight:700; font-size:1.15rem; color:{INK_NAVY}; font-variant-numeric: tabular-nums; margin-bottom:4px; }}
.labels {{ display:grid; grid-template-columns: repeat(7, minmax(0,1fr)); gap:10px; margin-top:8px; }}
.labels div {{ font-size:.78rem; color:{SLATE}; line-height:1.3; }}
.labels b {{ display:block; color:{INK}; font-size:.84rem; }}

/* opportunity rows (list pattern) */
.opp {{ display:grid; grid-template-columns: 30px 1fr 150px; gap:12px; align-items:center; padding:13px 0; border-bottom:1px solid {LINE}; }}
.opp:last-child {{ border-bottom:none; }}
.opp .rank {{ width:28px; height:28px; border-radius:50%; background:#EEF0F4; color:{INK_NAVY}; font-weight:700; font-size:.82rem;
  display:flex; align-items:center; justify-content:center; }}
.opp .name {{ font-weight:600; color:{INK}; font-size:.96rem; }}
.opp .meta {{ color:{SLATE}; font-size:.8rem; margin-top:2px; }}
.meter {{ background:#E9ECF1; height:8px; border-radius:4px; overflow:hidden; }}
.meter > span {{ display:block; height:100%; background:{GREEN}; border-radius:4px; }}
.score {{ font-size:.8rem; color:{GREEN}; font-weight:700; margin-top:4px; text-align:right; font-variant-numeric: tabular-nums; }}

/* quotes, findings, notes */
.quote {{ background:{SURFACE_2}; border-radius:16px; padding:12px 14px; margin:8px 0; }}
.quote q {{ color:{INK}; font-size:.92rem; line-height:1.5; }}
.quote .src {{ display:block; color:{SLATE}; font-size:.76rem; margin-top:6px; }}
.finding {{ padding:4px 0 12px; border-bottom:1px solid {LINE}; margin-bottom:12px; }}
.finding:last-child {{ border-bottom:none; margin-bottom:0; }}
.finding h4 {{ margin:0 0 4px; font-size:.98rem; color:{INK_NAVY}; line-height:1.4; }}
.finding p {{ margin:0; color:{SLATE}; font-size:.88rem; line-height:1.5; }}
.note {{ color:{SLATE}; font-size:.86rem; line-height:1.55; }}
.notebox {{ background:{NOTE_BG}; border-radius:16px; padding:14px 16px; }}
.notebox b {{ color:{BLUE}; display:block; margin-bottom:4px; }}
.notebox span {{ color:#2C3A55; font-size:.88rem; line-height:1.5; }}

/* onboarding (checklist pattern) */
.welcome {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px; }}
.welcome div {{ background:{SURFACE_2}; border-radius:16px; padding:14px 16px; }}
.welcome b {{ display:block; color:{INK_NAVY}; margin-bottom:4px; }}
.welcome span {{ color:{SLATE}; font-size:.86rem; line-height:1.5; }}
.steps {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(210px,1fr)); gap:12px; }}
.step {{ background:{SURFACE_2}; border-radius:16px; padding:14px 16px; }}
.step .i {{ display:inline-flex; width:24px; height:24px; border-radius:50%; background:#E3F4EA; color:{GREEN};
  font-weight:700; font-size:.8rem; align-items:center; justify-content:center; }}
.step b {{ display:block; color:{INK_NAVY}; margin:6px 0 4px; }}
.step span {{ color:{SLATE}; font-size:.84rem; line-height:1.5; }}
.fb-hello {{ display:flex; gap:14px; align-items:center; margin-bottom:8px; }}
.fb-ico {{ width:46px; height:46px; border-radius:14px; background:{NAVY}; display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
.foot {{ margin-top:36px; padding-top:16px; border-top:1px solid {LINE}; color:{SLATE}; font-size:.8rem; display:flex; flex-wrap:wrap; gap:6px 22px; }}
.foot b {{ color:{INK}; }}

/* widgets */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius:999px; font-weight:600; min-height:44px; padding: 0 20px; border:1px solid {LINE}; }}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {{
  background:{GREEN_BTN}; border-color:{GREEN_BTN}; color:#FFFFFF; }}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {{ border-radius:14px; }}
[data-testid="stTextInput"] > div > div, [data-testid="stTextArea"] > div > div {{ border-radius:14px; }}
[data-baseweb="select"] > div {{ border-radius:14px; }}
[data-testid="stPills"] button, [data-testid="stButtonGroup"] button {{ border-radius:14px; }}
[data-testid="stPills"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-checked="true"] {{
  background:{PILL_DARK}; color:#FFFFFF; border-color:{PILL_DARK}; }}
[data-testid="stChatMessage"] {{ background:{SURFACE}; border:1px solid #EEF0F3; border-radius:20px; padding:14px 16px; }}
[data-testid="stExpander"] details {{ border-radius:16px; border-color:{LINE}; background:{SURFACE}; }}
[data-testid="stDataFrame"] {{ border-radius:14px; overflow:hidden; }}
[data-testid="stChatInput"] {{ border-radius:999px; }}
:focus-visible {{ outline:2px solid {BLUE} !important; outline-offset:2px; }}

/* FinBot launcher: floating assistant pill (navy glass, mint orb, two-line label) */
.st-key-finbot_fab {{ position:fixed; right:28px; bottom:26px; z-index:1000; width:auto !important; }}
.st-key-finbot_fab button {{
  background: linear-gradient(135deg, #2A3F78 0%, {NAVY} 55%, {NAVY_2} 100%) !important;
  border:1px solid rgba(33,206,153,.35) !important; border-radius:999px !important;
  height:62px !important; min-height:62px; padding:8px 24px 8px 8px !important; overflow:visible !important;
  box-shadow: 0 14px 34px rgba(20,32,70,.38), 0 0 0 5px rgba(33,206,153,.07) !important; }}
.st-key-finbot_fab button {{ position:relative; padding-left:64px !important; justify-content:flex-start !important;
  width:auto !important; min-width:0 !important; }}
.st-key-finbot_fab button > div {{ justify-content:flex-start !important; gap:0 !important; }}
.st-key-finbot_fab button [data-testid="stMarkdownContainer"] {{ text-align:left !important; width:auto !important; }}
.st-key-finbot_fab button::before {{ content:"" !important; position:absolute !important; left:8px !important; top:8px !important;
  right:auto !important; bottom:auto !important; width:46px !important; height:46px !important; min-height:0 !important;
  max-height:46px !important; transform:none !important; border-radius:50% !important;
  background: radial-gradient(circle at 30% 28%, #8DF5D6 0%, {MINT} 55%, #129C73 100%);
  box-shadow: inset 0 -2px 6px rgba(0,0,0,.12); }}
.st-key-finbot_fab button [data-testid="stIconMaterial"] {{ position:absolute; left:20px; top:20px;
  z-index:1; font-size:22px !important; color:{NAVY} !important; }}
.st-key-finbot_fab button p {{ color:#FFFFFF !important; font-weight:700; font-size:.98rem; line-height:1.15; text-align:left; }}
.st-key-finbot_fab button p::after {{ content:"AI research assistant"; display:block; font-size:.72rem; font-weight:500;
  color:#A9B8DA; margin-top:2px; letter-spacing:.01em; }}
.st-key-finbot_fab button:hover {{ border-color:{MINT} !important;
  box-shadow: 0 18px 40px rgba(20,32,70,.45), 0 0 0 6px rgba(33,206,153,.14) !important; }}
.st-key-finbot_fab button:focus-visible {{ outline:3px solid {MINT} !important; outline-offset:3px; }}
@keyframes fbpulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(33,206,153,.6); }} 70% {{ box-shadow: 0 0 0 14px rgba(33,206,153,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(33,206,153,0); }} }}
@media (prefers-reduced-motion: no-preference) {{
  .st-key-finbot_fab button {{ transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }}
  .st-key-finbot_fab button:hover {{ transform: translateY(-3px); }}
  .st-key-finbot_fab button::before {{ animation: fbpulse 2.4s ease-out 1s 3; }} }}

/* mobile */
@media (max-width: 760px) {{
  .block-container {{ padding: 0 1rem 6rem; }}
  .st-key-band {{ margin: 0 -1rem 18px; padding: 22px 1rem 18px; border-radius:0 0 24px 24px; }}
  .st-key-band .phead h1 {{ font-size:1.4rem; }}
  .hero-h {{ font-size:1.3rem; }}
  .kpis, .kpis.row, .welcome {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .welcome {{ grid-template-columns: 1fr; }}
  .strip, .labels {{ gap:4px; }} .n {{ font-size:.9rem; }} .labels div {{ font-size:.66rem; }}
  .opp {{ grid-template-columns: 28px 1fr 90px; }}
  .st-key-finbot_fab {{ right:14px; bottom:14px; }}
  .st-key-finbot_fab button {{ padding:0 !important; height:62px !important; width:62px !important; min-width:62px !important; }}
  .st-key-finbot_fab button p {{ display:none; }}
  .ctitle::before {{ left:-16px; }} [class*="st-key-card"] {{ padding:18px 16px; }}
}}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def e(s) -> str:
    return escape("" if s is None else str(s))


def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{e(text)}</span>'


def quote_html(q: str, src: str) -> str:
    return f'<div class="quote"><q>{e(q)}</q><span class="src">{e(src)}</span></div>'


def card_title(title: str, sub: str = ""):
    st.markdown(f'<div class="ctitle">{e(title)}</div>' + (f'<div class="csub">{e(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)


_n = [0]


@contextmanager
def card(title: str = "", sub: str = "", key: str | None = None):
    _n[0] += 1
    with st.container(key=key or f"card_{_n[0]}"):
        if title:
            card_title(title, sub)
        yield
