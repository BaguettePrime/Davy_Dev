"""Home -- Landing page for EEG DeepDive."""

import streamlit as st

from utils.style import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_PURPLE,
    BG_CARD,
    FOOTER_HTML,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    inject_css,
    paper_badge,
)

inject_css()

# ---------------------------------------------------------------------------
# Title & subtitle
# ---------------------------------------------------------------------------
st.markdown('<h1 class="glow-header">EEG DeepDive</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p style="text-align:center;color:{TEXT_SECONDARY};font-size:1.15rem;margin-top:-8px;">'
    "A Visual Tour Through Three Papers on EEG Deep Learning</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Narrative paragraph
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div style="max-width:760px;margin:24px auto 32px auto;color:{TEXT_SECONDARY};
            font-size:1rem;line-height:1.7;text-align:center;">
EEG signals are rich, noisy, and deeply personal.  Over the course of three
research papers we tackled a single question from multiple angles:
<em>How can deep learning reliably decode brain activity from scalp
electrodes?</em>  This interactive app walks you through the ideas, the
architectures, and the results&mdash;whether you work in pharma, study
neuroscience, or build machine-learning models.
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Audience selector
# ---------------------------------------------------------------------------
st.markdown(
    f'<div style="text-align:center;margin-bottom:8px;">'
    f'<span style="color:{TEXT_PRIMARY};font-size:1.2rem;font-weight:600;">'
    "Choose your path</span></div>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
<div class="audience-card" style="border-top:3px solid {ACCENT_AMBER};">
    <div style="font-size:2rem;margin-bottom:8px;">🏢</div>
    <div style="color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;margin-bottom:8px;">
        I'm in Pharma</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.5;">
        Start with <strong>EEG Fundamentals</strong>, then
        <strong>Detection Challenge</strong>, and finish at
        <strong>Results &amp; Impact</strong>.
    </div>
</div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
<div class="audience-card" style="border-top:3px solid {ACCENT_CYAN};">
    <div style="font-size:2rem;margin-bottom:8px;">🎓</div>
    <div style="color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;margin-bottom:8px;">
        I'm a Student</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.5;">
        Follow all seven modules in order for a comprehensive,
        step-by-step learning experience.
    </div>
</div>""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
<div class="audience-card" style="border-top:3px solid {ACCENT_PURPLE};">
    <div style="font-size:2rem;margin-bottom:8px;">🔬</div>
    <div style="color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;margin-bottom:8px;">
        I'm a Researcher</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.5;">
        Jump directly to any technical module&mdash;each one is
        self-contained with architecture diagrams and ablations.
    </div>
</div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Paper cards
# ---------------------------------------------------------------------------
st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center;margin-bottom:8px;">'
    f'<span style="color:{TEXT_PRIMARY};font-size:1.2rem;font-weight:600;">'
    "The Papers</span></div>",
    unsafe_allow_html=True,
)

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown(
        f"""
<div class="paper-card" style="border-top:3px solid {ACCENT_AMBER};">
    <div style="margin-bottom:12px;">{paper_badge("Paper 1")}</div>
    <div style="color:{TEXT_PRIMARY};font-size:1.05rem;font-weight:700;margin-bottom:6px;">
        The Detection Challenge</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.85rem;line-height:1.6;">
        Thomas et al.<br>
        <em>Neuroscience Informatics</em>, 2026
    </div>
</div>""",
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        f"""
<div class="paper-card" style="border-top:3px solid {ACCENT_CYAN};">
    <div style="margin-bottom:12px;">{paper_badge("Paper 2")}</div>
    <div style="color:{TEXT_PRIMARY};font-size:1.05rem;font-weight:700;margin-bottom:6px;">
        CoSupFormer</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.85rem;line-height:1.6;">
        Darankoum et al.<br>
        <em>arXiv</em>, 2025
    </div>
</div>""",
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        f"""
<div class="paper-card" style="border-top:3px solid {ACCENT_PURPLE};">
    <div style="margin-bottom:12px;">{paper_badge("Paper 3")}</div>
    <div style="color:{TEXT_PRIMARY};font-size:1.05rem;font-weight:700;margin-bottom:6px;">
        SpecMoE</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.85rem;line-height:1.6;">
        Darankoum et al.<br>
        <em>arXiv</em>, 2026
    </div>
</div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
