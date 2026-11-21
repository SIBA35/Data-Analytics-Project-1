"""Page styling helpers (one palette: navy, teal, amber, coral)."""
import html
import streamlit as st

NAVY, TEAL, AMBER, CORAL, INK = "#1F3A5F", "#2A9D8F", "#E9A23B", "#E76F51", "#1B2733"

CSS = f"""
<style>
.stApp {{ background:#F4F7FB; }}
.block-container {{ max-width:1150px; padding-top:1.4rem; padding-bottom:3rem; }}
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color:{INK}; }}
[data-testid="stSidebar"] {{ background:#FFFFFF; }}
.hero {{ background:linear-gradient(120deg,{NAVY} 0%,#25607A 55%,{TEAL} 100%); border-radius:16px; padding:30px 34px; margin-bottom:18px; }}
.hero h1 {{ color:#FFFFFF !important; font-size:2.0rem; margin:0 0 6px 0; padding:0; line-height:1.2; }}
.hero p {{ color:#E3F1F0 !important; margin:0; font-size:1.02rem; }}
.hero small {{ color:#BFE0DC !important; display:block; margin-top:10px; font-size:.86rem; }}
.kpis {{ display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 22px 0; }}
.kpi {{ flex:1 1 200px; background:#FFFFFF; border-radius:12px; padding:16px 18px; border-left:6px solid {TEAL}; box-shadow:0 1px 4px rgba(31,58,95,.10); }}
.kpi:nth-child(2) {{ border-left-color:{NAVY}; }} .kpi:nth-child(3) {{ border-left-color:{AMBER}; }} .kpi:nth-child(4) {{ border-left-color:{CORAL}; }}
.kpi .lab {{ font-size:.76rem; letter-spacing:.06em; text-transform:uppercase; color:#6B778A !important; }}
.kpi .val {{ font-size:1.6rem; font-weight:700; color:{NAVY} !important; line-height:1.25; }}
.kpi .sub {{ font-size:.82rem; color:#6B778A !important; }}
.sec {{ display:flex; align-items:center; gap:12px; margin:30px 0 10px 0; }}
.sec .num {{ background:{NAVY}; color:#FFFFFF !important; width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; }}
.sec .ttl {{ font-size:1.45rem; font-weight:700; color:{NAVY} !important; }}
.lead {{ color:#4A576B !important; margin:-4px 0 12px 46px; }}
.box {{ border-radius:10px; padding:12px 16px; margin:8px 0 14px 0; font-size:.97rem; line-height:1.5; }}
.box.info {{ background:#E8F5F3; border-left:5px solid {TEAL}; }}
.box.warn {{ background:#FFF4E1; border-left:5px solid {AMBER}; }}
.box.bad  {{ background:#FDECE7; border-left:5px solid {CORAL}; }}
.card {{ background:#FFFFFF; border-radius:12px; padding:16px 20px; box-shadow:0 1px 4px rgba(31,58,95,.10); margin-bottom:14px; }}
.card ul {{ margin:0; padding-left:20px; }} .card li {{ margin:6px 0; line-height:1.5; }}
.card h4 {{ margin:0 0 8px 0; color:{NAVY} !important; }}
.recos {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; }}
.reco {{ background:#FFFFFF; border-radius:12px; padding:16px 18px; box-shadow:0 1px 4px rgba(31,58,95,.10); border-top:5px solid {TEAL}; line-height:1.5; }}
.reco:nth-child(2) {{ border-top-color:{NAVY}; }} .reco:nth-child(3) {{ border-top-color:{AMBER}; }} .reco:nth-child(4) {{ border-top-color:{CORAL}; }}
.reco b {{ color:{NAVY} !important; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def _e(x):
    return html.escape(str(x))


def hero(title, subtitle, caption=""):
    st.markdown(f'<div class="hero"><h1>{_e(title)}</h1><p>{_e(subtitle)}</p><small>{_e(caption)}</small></div>', unsafe_allow_html=True)


def kpi_cards(items):
    cells = "".join(f'<div class="kpi"><div class="lab">{_e(l)}</div><div class="val">{_e(v)}</div><div class="sub">{_e(s)}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def section(num, title, lead=""):
    badge = f'<div class="num">{_e(num)}</div>' if str(num) else ""
    st.markdown(f'<div class="sec">{badge}<div class="ttl">{_e(title)}</div></div>', unsafe_allow_html=True)
    if lead:
        st.markdown(f'<div class="lead">{_e(lead)}</div>', unsafe_allow_html=True)


def callout(text, kind="info", label=""):
    lab = f"<b>{_e(label)}</b> " if label else ""
    st.markdown(f'<div class="box {kind}">{lab}{_e(text)}</div>', unsafe_allow_html=True)


def bullet_card(title, items):
    li = "".join(f"<li>{_e(i)}</li>" for i in items)
    st.markdown(f'<div class="card"><h4>{_e(title)}</h4><ul>{li}</ul></div>', unsafe_allow_html=True)


def reco_cards(items):
    cells = "".join(f'<div class="reco"><b>{i + 1}.</b> {_e(t)}</div>' for i, t in enumerate(items))
    st.markdown(f'<div class="recos">{cells}</div>', unsafe_allow_html=True)


def table(rows, headers, colors=None):
    """Simple styled HTML table. rows: list of lists of strings; colors: optional {(row, col): colour}."""
    colors = colors or {}
    head = "".join(f"<th style='text-align:{'left' if i == 0 else 'right'};padding:9px 12px;background:{NAVY};color:#fff'>{_e(h)}</th>" for i, h in enumerate(headers))
    body = ""
    for r, row in enumerate(rows):
        bg = "#FFFFFF" if r % 2 == 0 else "#F4F7FB"
        cells = "".join(f"<td style='text-align:{'left' if i == 0 else 'right'};padding:8px 12px;background:{bg};color:{colors.get((r, i), INK)};"
                        f"font-weight:{'700' if (r, i) in colors else '400'}'>{_e(c)}</td>" for i, c in enumerate(row))
        body += f"<tr>{cells}</tr>"
    st.markdown(f"<div class='card' style='padding:6px;overflow-x:auto'><table style='width:100%;border-collapse:collapse;font-size:.95rem'><tr>{head}</tr>{body}</table></div>", unsafe_allow_html=True)


def conclusion_cards(items):
    cells = "".join(f'<div class="reco"><b>{i + 1}. {_e(t)}</b><br>{_e(x)}</div>' for i, (t, x) in enumerate(items))
    st.markdown(f'<div class="recos">{cells}</div>', unsafe_allow_html=True)
