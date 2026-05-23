from __future__ import annotations

import pandas as pd
import streamlit as st


def render_metrics(probabilities: dict[str, float], fps: float, latency_ms: float) -> None:
    st.metric("FPS", f"{fps:.1f}")
    st.metric("Latency", f"{latency_ms:.1f} ms")
    chart_data = pd.DataFrame(
        {
            "class": list(probabilities.keys()),
            "confidence": [value * 100.0 for value in probabilities.values()],
        }
    )
    st.bar_chart(chart_data, x="class", y="confidence", height=240)
