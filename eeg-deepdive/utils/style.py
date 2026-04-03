"""Shared CSS injection, color constants, and card helpers for EEG DeepDive."""

import streamlit as st

# ---------------------------------------------------------------------------
# Color palette – scientific / neuroscience dark theme
# ---------------------------------------------------------------------------
BG_PRIMARY = "#0a0e17"
BG_CARD = "#111827"
ACCENT_CYAN = "#06b6d4"
ACCENT_PURPLE = "#8b5cf6"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"

PAPER_COLORS = {
    "Paper 1": ACCENT_AMBER,
    "Paper 2": ACCENT_CYAN,
    "Paper 3": ACCENT_PURPLE,
}


def inject_css():
    """Inject the global CSS theme into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def paper_badge(paper: str) -> str:
    """Return an HTML badge for a paper label."""
    color = PAPER_COLORS.get(paper, ACCENT_CYAN)
    return (
        f'<span style="background:{color}22;color:{color};'
        f'padding:4px 12px;border-radius:20px;font-size:0.8rem;'
        f'font-weight:600;border:1px solid {color}44;">{paper}</span>'
    )


def metric_card(label: str, value: str, delta: str = "") -> str:
    """Return HTML for a compact metric display card."""
    delta_html = ""
    if delta:
        color = ACCENT_GREEN if delta.startswith("+") or delta.startswith("↑") else ACCENT_RED
        delta_html = f'<div style="color:{color};font-size:0.85rem;margin-top:4px;">{delta}</div>'
    return (
        f'<div class="metric-card">'
        f'<div style="color:{TEXT_SECONDARY};font-size:0.8rem;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:4px;">{label}</div>'
        f'<div style="color:{TEXT_PRIMARY};font-size:1.8rem;font-weight:700;">{value}</div>'
        f'{delta_html}</div>'
    )


def callout_box(text: str, icon: str = "💡", color: str = ACCENT_CYAN) -> str:
    """Return HTML for a highlighted insight callout box."""
    return (
        f'<div style="background:{color}11;border-left:4px solid {color};'
        f'padding:16px 20px;border-radius:0 8px 8px 0;margin:16px 0;">'
        f'<span style="font-size:1.2rem;margin-right:8px;">{icon}</span>'
        f'<span style="color:{TEXT_PRIMARY};font-size:0.95rem;">{text}</span></div>'
    )


def section_header(title: str, subtitle: str = "") -> str:
    """Return HTML for a styled section header."""
    sub = ""
    if subtitle:
        sub = f'<div style="color:{TEXT_SECONDARY};font-size:0.95rem;margin-top:4px;">{subtitle}</div>'
    return (
        f'<div style="margin:32px 0 16px 0;">'
        f'<h2 style="color:{TEXT_PRIMARY};margin:0;font-size:1.6rem;">{title}</h2>'
        f'{sub}</div>'
    )


FOOTER_HTML = f"""
<hr style="border-color:{BG_CARD};margin-top:48px;">
<div style="text-align:center;padding:24px 0;color:{TEXT_SECONDARY};font-size:0.8rem;">
    <strong>EEG DeepDive</strong> — An educational companion to three research papers<br>
    Authors: David Darankoum, Romain Thomas &amp; collaborators<br>
    Affiliations: Univ. Grenoble Alpes &middot; SynapCell<br>
    <em>This app is for educational purposes only. No clinical decisions should be based on its content.</em>
</div>
"""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = f"""<style>
/* ---- Global ---- */
.stApp {{
    background-color: {BG_PRIMARY};
}}
section[data-testid="stSidebar"] {{
    background-color: {BG_CARD};
}}

/* ---- Glow header ---- */
.glow-header {{
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, {ACCENT_CYAN}, {ACCENT_PURPLE}, {ACCENT_GREEN});
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 4s ease infinite;
    text-align: center;
    margin-bottom: 0;
}}
@keyframes gradient-shift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* ---- Neuro card ---- */
.neuro-card {{
    background: {BG_CARD};
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px;
    margin: 8px 0;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}}
.neuro-card:hover {{
    transform: translateY(-2px);
    border-color: {ACCENT_CYAN}55;
}}

/* ---- Metric card ---- */
.metric-card {{
    background: {BG_CARD};
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}}

/* ---- Callout box ---- */
.callout-box {{
    background: {ACCENT_CYAN}11;
    border-left: 4px solid {ACCENT_CYAN};
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0;
}}

/* ---- Custom scrollbar ---- */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: {BG_PRIMARY};
}}
::-webkit-scrollbar-thumb {{
    background: #334155;
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: #475569;
}}

/* ---- Links ---- */
a {{
    color: {ACCENT_CYAN} !important;
    text-decoration: none !important;
}}
a:hover {{
    text-decoration: underline !important;
}}

/* ---- Audience card ---- */
.audience-card {{
    background: {BG_CARD};
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 28px 20px;
    text-align: center;
    min-height: 180px;
    transition: transform 0.2s ease, border-color 0.2s ease;
    cursor: default;
}}
.audience-card:hover {{
    transform: translateY(-3px);
    border-color: {ACCENT_CYAN}66;
}}

/* ---- Paper card ---- */
.paper-card {{
    background: {BG_CARD};
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px;
    min-height: 200px;
    transition: border-color 0.2s ease;
}}

/* ---- Tab styling ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {BG_CARD};
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    color: {TEXT_SECONDARY};
}}
.stTabs [aria-selected="true"] {{
    background-color: {ACCENT_CYAN}22;
    color: {ACCENT_CYAN};
}}
</style>"""
