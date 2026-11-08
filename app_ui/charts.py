"""Plotly figures in the app's style."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .style import BLUE, INK, LINE, NAVY, SAFFRON, SEQ, SLATE, TEAL

FONT = "Inter, system-ui, sans-serif"


def _style(fig: go.Figure, height: int = 360, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=30, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK, size=13), colorway=SEQ,
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(font_family=FONT, bgcolor="white"),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(color=SLATE), title=None)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", zeroline=False, linecolor=LINE, tickfont=dict(color=INK), title=None)
    if legend:
        fig.update_layout(margin=dict(l=8, r=16, t=70, b=8))
    try:
        fig.update_traces(marker_cornerradius=5, selector=dict(type="bar"))
    except Exception:  # older plotly
        pass
    return fig


def hbar(df: pd.DataFrame, label: str, value: str = "share", color: str = TEAL, pct: bool = True,
         height: int | None = None, text: str | None = None) -> go.Figure:
    d = df.sort_values(value)
    fig = go.Figure(go.Bar(
        x=d[value], y=d[label], orientation="h", marker_color=color,
        text=d[text] if text else (d[value].map(lambda v: f"{v:.0%}") if pct else d[value]),
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.0%}<extra></extra>" if pct else "%{y}<br>%{x}<extra></extra>"))
    fig = _style(fig, height or max(220, 34 * len(d) + 60))
    if pct:
        fig.update_xaxes(tickformat=".0%", range=[0, max(d[value].max() * 1.22, 0.05)] if len(d) else None)
    return fig


def grouped(df: pd.DataFrame, label: str, series: list[str], names: list[str], colors=(TEAL, SAFFRON),
            height: int | None = None) -> go.Figure:
    d = df.sort_values(series[-1])
    fig = go.Figure()
    for s, n, c in zip(series, names, colors):
        fig.add_bar(x=d[s], y=d[label], name=n, orientation="h", marker_color=c,
                    hovertemplate=f"{n}<br>%{{y}}: %{{x:.0%}}<extra></extra>")
    fig = _style(fig, height or max(260, 30 * len(d) * len(series) // 2 + 90), legend=True)
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(tickformat=".0%")
    return fig


def heatmap(df: pd.DataFrame, row: str, col: str, value: str = "share", height: int | None = None) -> go.Figure:
    pv = df.pivot_table(index=row, columns=col, values=value, aggfunc="mean").fillna(0)
    fig = px.imshow(pv, color_continuous_scale=[[0, "#F6F7F9"], [0.5, "#8C9AC0"], [1, NAVY]],
                    aspect="auto", text_auto=".0%")
    fig = _style(fig, height or max(320, 28 * len(pv) + 120))
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(side="top", tickangle=0, gridcolor="rgba(0,0,0,0)")
    fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:.0%}<extra></extra>")
    return fig


def funnel(stages: list[str], values: list[int]) -> go.Figure:
    fig = go.Figure(go.Funnel(y=stages, x=values, textinfo="value", marker_color=[NAVY] * (len(values) - 1) + [TEAL],
                              connector={"line": {"color": LINE}}))
    return _style(fig, 300)


def opp_components(opp: pd.DataFrame) -> go.Figure:
    d = opp.head(10).iloc[::-1]
    fig = go.Figure()
    for comp, name, c, mx in [("F", "Frequency /5", TEAL, 5), ("S", "Severity /5", SAFFRON, 5),
                              ("L", "Metric link /3", BLUE, 3), ("A", "Addressability /3", "#8A4F9E", 3)]:
        fig.add_bar(y=d["label"], x=d[comp] / mx, name=name, orientation="h", marker_color=c,
                    customdata=d[comp], hovertemplate=f"{name}: %{{customdata:.1f}}<extra></extra>")
    fig = _style(fig, 420, legend=True)
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(tickformat=".0%")
    return fig


def branch_bar(opp: pd.DataFrame) -> go.Figure:
    g = (opp.groupby("branch_label")
            .agg(score=("score", "sum"), themes=("label", lambda x: "<br>".join(x)))
            .reset_index().sort_values("score"))
    fig = go.Figure(go.Bar(x=g["score"], y=g["branch_label"], orientation="h", marker_color=NAVY,
                           text=g["score"].map(lambda v: f"{v:.0f}"), textposition="outside", cliponaxis=False,
                           customdata=g["themes"], hovertemplate="%{y}<br>total score %{x:.0f}<br>%{customdata}<extra></extra>"))
    fig = _style(fig, 300)
    fig.update_xaxes(range=[0, g["score"].max() * 1.2 if len(g) else 1])
    return fig


def trend(df: pd.DataFrame, labels: dict) -> go.Figure:
    d = df.assign(theme=df["theme"].map(labels))
    fig = px.line(d, x="month", y="share", color="theme", markers=True, color_discrete_sequence=SEQ)
    fig = _style(fig, 340, legend=True)
    fig.update_yaxes(tickformat=".0%", gridcolor=LINE)
    return fig


def survey_bar(df: pd.DataFrame) -> go.Figure:
    return hbar(df.rename(columns={"answer": "label"}), "label", "share", BLUE,
                text="label_n" if "label_n" in df else None)


def survey_cross(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, y="answer", x="share", color="group", orientation="h", barmode="group",
                 color_discrete_map={"Kept going": TEAL, "Paused / stopped / reduced": SAFFRON})
    fig = _style(fig, max(240, 44 * df["answer"].nunique() + 80), legend=True)
    fig.update_xaxes(tickformat=".0%")
    return fig
