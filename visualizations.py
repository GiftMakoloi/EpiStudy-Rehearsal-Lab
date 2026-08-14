from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_recruitment_funnel(
    summary: dict,
):
    labels = [
        "Selected",
        "Recruited",
        "Retained",
        "Outcome events",
    ]

    values = [
        summary["n_selected"],
        summary["n_recruited"],
        summary["n_retained"],
        summary["events"],
    ]

    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            textinfo="value+percent initial",
        )
    )

    fig.update_layout(
        title=(
            "Virtual Study Recruitment "
            "and Outcome Funnel"
        ),
        height=420,
    )

    return fig


def plot_estimate_distribution(
    results: pd.DataFrame,
    true_rr: float,
):
    fig = px.histogram(
        results,
        x="risk_ratio",
        nbins=30,
        title=(
            "Distribution of Estimated "
            "Risk Ratios Across Rehearsals"
        ),
    )

    fig.add_vline(
        x=true_rr,
        line_dash="dash",
        annotation_text=(
            f"True RR = {true_rr:.2f}"
        ),
    )

    fig.update_layout(
        height=420,
    )

    return fig


def plot_retention_distribution(
    results: pd.DataFrame,
):
    fig = px.histogram(
        results,
        x="n_retained",
        nbins=25,
        title=(
            "Distribution of Final "
            "Analysable Sample Sizes"
        ),
    )

    fig.update_layout(
        height=360,
    )

    return fig


def plot_events_distribution(
    results: pd.DataFrame,
):
    fig = px.histogram(
        results,
        x="events",
        nbins=25,
        title=(
            "Distribution of Outcome "
            "Events Across Rehearsals"
        ),
    )

    fig.update_layout(
        height=360,
    )

    return fig


def plot_sensitivity_surface(
    grid_df: pd.DataFrame,
):
    fig = px.scatter(
        grid_df,
        x="true_rr",
        y="dropout_rate",
        size="success_probability",
        color="success_probability",
        hover_data=[
            "mean_events",
            "mean_retained",
        ],
        title=(
            "Study Resilience to "
            "Effect Size and Dropout"
        ),
    )

    fig.update_layout(
        height=440,
    )

    return fig


def plot_assumption_curve(
    curve_df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_title: str,
    y_title: str,
):
    fig = px.line(
        curve_df,
        x=x,
        y=y,
        markers=True,
        title=title,
    )

    fig.update_xaxes(
        title=x_title
    )

    fig.update_yaxes(
        title=y_title
    )

    fig.update_layout(
        height=420
    )

    return fig
