"""Progress tracker custom Streamlit component."""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

_component_func = components.declare_component("progress_tracker", path=_COMPONENT_DIR)

MODULE_NAMES = [
    "EEG Fundamentals",
    "Detection Challenge",
    "Multi-Scale Encoding",
    "Attention & Gating",
    "Learning Representations",
    "Foundation Models",
    "Results & Impact",
]

def progress_tracker(current: int, visited: list[int] | None = None, key: str | None = None):
    """Render the module progress tracker.

    Args:
        current: 1-indexed current module number (1-7).
        visited: List of 1-indexed module numbers the user has visited.
        key: Streamlit component key.
    """
    if visited is None:
        visited = []
    return _component_func(
        current=current,
        visited=visited,
        modules=MODULE_NAMES,
        key=key,
        default=None,
    )
