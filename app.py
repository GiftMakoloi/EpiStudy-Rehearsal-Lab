from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from analysis import (
    analyse_rehearsal,
    run_repeated_rehearsals,
)

from simulation import (
    generate_virtual_population,
    run_study_rehearsal,
)

from visualizations import (
    plot_events_distribution,
    plot_estimate_distribution,
    plot_recruitment_funnel,
    plot_retention_distribution,
    plot_sensitivity_surface,
)


st.set_page_config(
    page_title="EpiStudy Rehearsal Lab",
    page_icon="R",
    layout="wide",
)


st.sidebar.title(
    "EpiStudy Rehearsal Lab"
)

st.sidebar.caption(
    "Run your proposed epidemiological study virtually before you run it in the real world."
)


page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Study Rehearsal",
        "Assumption Stress Test",
        "Reproducibility Report",
    ],
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "All data are synthetic. No real patient data are required."
)


def get_parameters() -> dict:

    with st.sidebar:

        st.subheader(
            "Study parameters"
        )

        population_size = st.slider(
            "Virtual population",
            1000,
            100000,
            20000,
            1000,
        )

        sample_target = st.slider(
            "Target sample",
            100,
            10000,
            2000,
            100,
        )

        exposure_prevalence = st.slider(
            "Exposure prevalence",
            0.02,
            0.90,
            0.20,
            0.01,
        )

        baseline_risk = st.slider(
            "Outcome risk in unexposed",
            0.005,
            0.50,
            0.08,
            0.005,
        )

        true_rr = st.slider(
            "True risk ratio",
            0.80,
            4.00,
            1.80,
            0.05,
        )

        recruitment_rate = st.slider(
            "Recruitment probability",
            0.50,
            1.00,
            0.90,
            0.01,
        )

        dropout_rate = st.slider(
            "Loss to follow-up",
            0.00,
            0.50,
            0.15,
            0.01,
        )

        repetitions = st.slider(
            "Virtual study rehearsals",
            25,
            1000,
            250,
            25,
        )

        seed = st.number_input(
            "Random seed",
            1,
            999999,
            42,
        )

    return {
        "population_size":
            population_size,
        "sample_target":
            sample_target,
        "exposure_prevalence":
            exposure_prevalence,
        "baseline_risk":
            baseline_risk,
        "true_risk_ratio":
            true_rr,
        "recruitment_rate":
            recruitment_rate,
        "dropout_rate":
            dropout_rate,
        "repetitions":
            repetitions,
        "seed":
            int(seed),
    }


def build_report(
    parameters: dict,
    summary: pd.DataFrame,
) -> str:

    row = summary.iloc[0]

    lines = [
        "EpiStudy Rehearsal Lab",
        "Reproducibility Report",
        "",
        "Study parameters",
        json.dumps(
            parameters,
            indent=2,
        ),
        "",
        "Simulation summary",
        row.to_json(indent=2),
        "",
        "Interpretation note",
        (
            "This report describes synthetic "
            "study rehearsals. Results depend "
            "on the specified data-generating "
            "assumptions and should not be "
            "interpreted as real-world clinical "
            "or public-health evidence."
        ),
    ]

    return "\n".join(
        lines
    )


if page == "Home":

    st.title(
        "EpiStudy Rehearsal Lab"
    )

    st.subheader(
        "Run your study before you run your study"
    )

    st.markdown(
        """
        **EpiStudy Rehearsal Lab** is a free,
        no-data-needed Streamlit application that
        lets a researcher specify a proposed
        epidemiological study and repeatedly
        rehearse it on synthetic populations.

        Instead of only calculating a sample-size
        number, the platform simulates a simplified
        research process: participant selection,
        recruitment, loss to follow-up, outcome
        events, final analysable sample and
        statistical estimation.
        """
    )

    st.info(
        """
        The purpose is not to predict exactly what
        will happen in a real study. It is to expose
        how your proposed study behaves under the
        assumptions you provide.
        """
    )

    st.markdown(
        "### The experiment"
    )

    cols = st.columns(5)

    for col, text in zip(
        cols,
        [
            "1. Specify study",
            "2. Create virtual population",
            "3. Rehearse study",
            "4. Measure failures",
            "5. Stress-test assumptions",
        ],
    ):
        col.success(text)

    st.markdown(
        "### What the platform measures"
    )

    st.write(
        """
        Recruitment, final sample size, outcome
        events, estimated risk ratio, sampling
        variability, bias relative to the generating
        truth, and the frequency with which a
        simulated study meets an educational
        evidence-success criterion.
        """
    )

    st.warning(
        """
        This is an educational simulation environment.
        It is not a substitute for formal prospective
        sample-size or power calculations, protocol
        review, statistical consultation, or regulatory
        and ethical review.
        """
    )


elif page == "Study Rehearsal":

    st.title(
        "Study Rehearsal"
    )

    st.write(
        """
        Configure a proposed study and run it
        virtually.
        """
    )

    params = get_parameters()

    if st.button(
        "Rehearse Study",
        type="primary",
        width="stretch",
    ):

        population = (
            generate_virtual_population(
                population_size=
                    params[
                        "population_size"
                    ],
                exposure_prevalence=
                    params[
                        "exposure_prevalence"
                    ],
                baseline_risk=
                    params[
                        "baseline_risk"
                    ],
                true_risk_ratio=
                    params[
                        "true_risk_ratio"
                    ],
                dropout_probability=
                    params[
                        "dropout_rate"
                    ],
                recruitment_probability=
                    params[
                        "recruitment_rate"
                    ],
                seed=
                    params["seed"],
            )
        )

        rehearsal = (
            run_study_rehearsal(
                population=
                    population,
                sample_target=
                    params[
                        "sample_target"
                    ],
                recruitment_rate=
                    params[
                        "recruitment_rate"
                    ],
                dropout_rate=
                    params[
                        "dropout_rate"
                    ],
                seed=
                    params["seed"] + 1,
            )
        )

        metrics = analyse_rehearsal(
            rehearsal["retained"],
            true_risk_ratio=
                params["true_risk_ratio"],
            true_risk_difference=(
                params["baseline_risk"]
                * params["true_risk_ratio"]
                - params["baseline_risk"]
            ),
        )

        st.session_state[
            "last_rehearsal"
        ] = rehearsal

        st.session_state[
            "last_metrics"
        ] = metrics

        st.session_state[
            "last_params"
        ] = params

    rehearsal = st.session_state.get(
        "last_rehearsal"
    )

    metrics = st.session_state.get(
        "last_metrics"
    )

    params = st.session_state.get(
        "last_params"
    )

    if rehearsal is None:

        st.info(
            """
            Set your assumptions in the sidebar
            and click Rehearse Study.
            """
        )

    else:

        c1, c2, c3, c4, c5 = (
            st.columns(5)
        )

        c1.metric(
            "Selected",
            f"{rehearsal['n_selected']:,}",
        )

        c2.metric(
            "Recruited",
            f"{rehearsal['n_recruited']:,}",
        )

        c3.metric(
            "Retained",
            f"{rehearsal['n_retained']:,}",
        )

        c4.metric(
            "Outcome events",
            f"{rehearsal['events']:,}",
        )

        c5.metric(
            "Estimated RR",
            (
                f"{rehearsal['risk_ratio']:.2f}"
                if pd.notna(
                    rehearsal["risk_ratio"]
                )
                else "N/A"
            ),
        )

        st.plotly_chart(
            plot_recruitment_funnel(
                rehearsal
            ),
            width="stretch",
        )

        st.markdown(
            "### Single-study result"
        )

        result_table = pd.DataFrame(
            {
                "Measure": [
                    "True risk ratio",
                    "Estimated risk ratio",
                    "Risk-ratio bias (%)",
                    "True risk difference",
                    "Estimated risk difference",
                    "Absolute RD error",
                    "Odds ratio",
                ],
                "Value": [
                    params[
                        "true_risk_ratio"
                    ],
                    metrics[
                        "risk_ratio"
                    ],
                    metrics[
                        "rr_bias_pct"
                    ],
                    (
                        params[
                            "baseline_risk"
                        ]
                        * params[
                            "true_risk_ratio"
                        ]
                        - params[
                            "baseline_risk"
                        ]
                    ),
                    metrics[
                        "risk_difference"
                    ],
                    metrics[
                        "rd_bias_abs"
                    ],
                    metrics[
                        "odds_ratio"
                    ],
                ],
            }
        )

        st.dataframe(
            result_table,
            hide_index=True,
            width="stretch",
        )

        st.markdown(
            "### Rehearse the same study repeatedly"
        )

        with st.spinner(
            "Running virtual study rehearsals..."
        ):

            results, summary = (
                run_repeated_rehearsals(
                    **params
                )
            )

        st.session_state[
            "last_results"
        ] = results

        st.session_state[
            "last_summary"
        ] = summary

        row = summary.iloc[0]

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "Mean retained",
            f"{row['mean_retained']:.0f}",
        )

        c2.metric(
            "Mean events",
            f"{row['mean_events']:.0f}",
        )

        c3.metric(
            "Mean RR",
            f"{row['mean_rr']:.2f}",
        )

        c4.metric(
            "Approx. success probability",
            (
                f"{row['probability_80pct_power'] * 100:.1f}%"
            ),
        )

        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
        )

        st.plotly_chart(
            plot_estimate_distribution(
                results,
                params[
                    "true_risk_ratio"
                ],
            ),
            width="stretch",
        )

        left, right = st.columns(2)

        with left:

            st.plotly_chart(
                plot_retention_distribution(
                    results
                ),
                width="stretch",
            )

        with right:

            st.plotly_chart(
                plot_events_distribution(
                    results
                ),
                width="stretch",
            )

        st.markdown(
            "### What did the rehearsal reveal?"
        )

        if (
            row[
                "probability_80pct_power"
            ]
            < 0.80
        ):

            st.error(
                """
                The proposed study is fragile
                under these assumptions: fewer
                than 80% of simulated rehearsals
                met the application's educational
                evidence-success criterion.
                """
            )

        else:

            st.success(
                """
                Under the specified assumptions,
                at least 80% of simulated rehearsals
                met the application's educational
                evidence-success criterion.
                """
            )

        if (
            row[
                "probability_under_100_events"
            ]
            > 0.20
        ):

            st.warning(
                """
                A substantial fraction of rehearsals
                produced fewer than 100 outcome events.
                Consider whether the proposed study is
                likely to generate enough information
                for the planned analysis.
                """
            )


elif page == "Assumption Stress Test":

    st.title(
        "Assumption Stress Test"
    )

    st.write(
        """
        Change the assumptions that matter most
        and see whether your proposed study remains
        viable.
        """
    )

    params = get_parameters()

    base_reps = min(
        params["repetitions"],
        250,
    )

    rr_values = [
        max(
            1.05,
            params["true_risk_ratio"] - 0.60,
        ),
        max(
            1.05,
            params["true_risk_ratio"] - 0.30,
        ),
        params["true_risk_ratio"],
        params["true_risk_ratio"] + 0.30,
        params["true_risk_ratio"] + 0.60,
    ]

    dropout_values = [
        max(
            0.0,
            params["dropout_rate"] - 0.10,
        ),
        max(
            0.0,
            params["dropout_rate"] - 0.05,
        ),
        params["dropout_rate"],
        min(
            0.50,
            params["dropout_rate"] + 0.10,
        ),
        min(
            0.50,
            params["dropout_rate"] + 0.15,
        ),
    ]

    rows = []

    total = (
        len(rr_values)
        * len(dropout_values)
    )

    count = 0

    progress = st.progress(
        0
    )

    for rr in rr_values:

        for dropout in dropout_values:

            _, summary = (
                run_repeated_rehearsals(
                    population_size=
                        params[
                            "population_size"
                        ],
                    exposure_prevalence=
                        params[
                            "exposure_prevalence"
                        ],
                    baseline_risk=
                        params[
                            "baseline_risk"
                        ],
                    true_risk_ratio=
                        float(rr),
                    sample_target=
                        params[
                            "sample_target"
                        ],
                    recruitment_rate=
                        params[
                            "recruitment_rate"
                        ],
                    dropout_rate=
                        float(dropout),
                    repetitions=
                        base_reps,
                    seed=
                        params["seed"]
                        + count * 17,
                )
            )

            row = summary.iloc[0]

            rows.append(
                {
                    "True RR":
                        float(rr),
                    "Dropout %":
                        float(dropout * 100),
                    "Success probability %":
                        float(
                            row[
                                "probability_80pct_power"
                            ] * 100
                        ),
                    "Mean events":
                        row[
                            "mean_events"
                        ],
                    "Mean retained":
                        row[
                            "mean_retained"
                        ],
                }
            )

            count += 1

            progress.progress(
                count / total
            )

    progress.empty()

    grid_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        grid_df,
        hide_index=True,
        width="stretch",
    )

    surface = grid_df.rename(
        columns={
            "True RR":
                "true_rr",
            "Dropout %":
                "dropout_rate",
            "Success probability %":
                "success_probability_pct",
        }
    )

    import plotly.express as px

    fig = px.scatter(
        surface,
        x="true_rr",
        y="dropout_rate",
        size="success_probability_pct",
        color="success_probability_pct",
        hover_data=[
            "Mean events",
            "Mean retained",
        ],
        title=(
            "Study Resilience to "
            "Effect Size and Dropout"
        ),
    )

    fig.update_layout(
        height=440,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    worst = grid_df.loc[
        grid_df[
            "Success probability %"
        ].idxmin()
    ]

    best = grid_df.loc[
        grid_df[
            "Success probability %"
        ].idxmax()
    ]

    st.markdown(
        "### What the stress test reveals"
    )

    st.write(
        f"""
        The strongest simulated scenario achieved
        **{best['Success probability %']:.1f}%**
        success probability, while the weakest
        achieved **{worst['Success probability %']:.1f}%**.
        """
    )

    st.info(
        """
        This is a sensitivity analysis around the
        assumptions you specified. It is not a formal
        guarantee of study performance.
        """
    )


elif page == "Reproducibility Report":

    st.title(
        "Reproducibility Report"
    )

    params = st.session_state.get(
        "last_params"
    )

    summary = st.session_state.get(
        "last_summary"
    )

    if (
        params is None
        or summary is None
    ):

        st.info(
            "Run a Study Rehearsal first."
        )

    else:

        report = build_report(
            params,
            summary,
        )

        st.code(
            report,
            language="text",
        )

        st.download_button(
            "Download rehearsal report",
            data=report,
            file_name=(
                "epistudy_rehearsal_report.txt"
            ),
            mime="text/plain",
            width="stretch",
        )

        st.download_button(
            "Download simulation summary CSV",
            data=summary.to_csv(
                index=False
            ),
            file_name=(
                "epistudy_rehearsal_summary.csv"
            ),
            mime="text/csv",
            width="stretch",
        )
