from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def odds_ratio_ci(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate an odds ratio with a simple Wald-type 95% CI.
    A 0.5 continuity correction is applied when a cell is zero.
    """

    d = df.dropna(
        subset=[
            "exposure",
            "outcome",
        ]
    ).copy()

    a = (
        (d["exposure"] == 1)
        & (d["outcome"] == 1)
    ).sum()

    b = (
        (d["exposure"] == 1)
        & (d["outcome"] == 0)
    ).sum()

    c = (
        (d["exposure"] == 0)
        & (d["outcome"] == 1)
    ).sum()

    dd = (
        (d["exposure"] == 0)
        & (d["outcome"] == 0)
    ).sum()

    cells = (
        np.array(
            [a, b, c, dd],
            dtype=float,
        )
        + 0.5
    )

    a, b, c, dd = cells

    odds_ratio = (
        a * dd
    ) / (
        b * c
    )

    se = np.sqrt(
        1 / a
        + 1 / b
        + 1 / c
        + 1 / dd
    )

    log_or = np.log(
        odds_ratio
    )

    return {
        "estimate": float(
            odds_ratio
        ),
        "ci_low": float(
            np.exp(
                log_or
                - 1.96 * se
            )
        ),
        "ci_high": float(
            np.exp(
                log_or
                + 1.96 * se
            )
        ),
        "n": int(len(d)),
    }


def risk_measures(
    df: pd.DataFrame,
) -> dict:
    d = df.dropna(
        subset=[
            "exposure",
            "outcome",
        ]
    ).copy()

    exposed = d[
        d["exposure"] == 1
    ]["outcome"]

    unexposed = d[
        d["exposure"] == 0
    ]["outcome"]

    risk_exposed = (
        exposed.mean()
    )

    risk_unexposed = (
        unexposed.mean()
    )

    risk_ratio = (
        risk_exposed
        / risk_unexposed
        if risk_unexposed > 0
        else np.nan
    )

    risk_difference = (
        risk_exposed
        - risk_unexposed
    )

    return {
        "risk_exposed":
            float(risk_exposed),
        "risk_unexposed":
            float(risk_unexposed),
        "risk_ratio":
            float(risk_ratio),
        "risk_difference":
            float(risk_difference),
    }


def logistic_or(
    df: pd.DataFrame,
) -> dict | None:
    d = df.dropna(
        subset=[
            "exposure",
            "outcome",
        ]
    ).copy()

    if (
        len(d) < 30
        or d["exposure"].nunique() < 2
        or d["outcome"].nunique() < 2
    ):
        return None

    try:
        x = sm.add_constant(
            d[["exposure"]],
            has_constant="add",
        )

        model = sm.Logit(
            d["outcome"],
            x,
        ).fit(
            disp=0,
            maxiter=200,
        )

        beta = float(
            model.params["exposure"]
        )

        ci = (
            model.conf_int()
            .loc["exposure"]
            .to_numpy(
                dtype=float
            )
        )

        return {
            "estimate":
                float(np.exp(beta)),
            "ci_low":
                float(np.exp(ci[0])),
            "ci_high":
                float(np.exp(ci[1])),
            "p_value":
                float(
                    model.pvalues[
                        "exposure"
                    ]
                ),
        }

    except (
        ValueError,
        np.linalg.LinAlgError,
    ):
        return None


def analyse_rehearsal(
    retained: pd.DataFrame,
    true_risk_ratio: float,
    true_risk_difference: float,
) -> dict:
    """
    Analyse one virtual study.
    """

    measures = risk_measures(
        retained
    )

    odds = odds_ratio_ci(
        retained
    )

    logistic = logistic_or(
        retained
    )

    rr = measures["risk_ratio"]

    rd = measures["risk_difference"]

    return {
        "risk_ratio": rr,
        "risk_difference": rd,
        "odds_ratio":
            odds["estimate"],
        "odds_ratio_ci_low":
            odds["ci_low"],
        "odds_ratio_ci_high":
            odds["ci_high"],
        "logistic_or":
            (
                np.nan
                if logistic is None
                else logistic["estimate"]
            ),
        "logistic_p_value":
            (
                np.nan
                if logistic is None
                else logistic["p_value"]
            ),
        "rr_bias_pct":
            (
                (
                    rr
                    - true_risk_ratio
                )
                / true_risk_ratio
                * 100
                if np.isfinite(rr)
                else np.nan
            ),
        "rd_bias_abs":
            (
                abs(
                    rd
                    - true_risk_difference
                )
                if np.isfinite(rd)
                else np.nan
            ),
    }


def estimate_success_probability(
    results: pd.DataFrame,
) -> float:
    """
    Educational simulation diagnostic.

    A rehearsal is treated as successful when:
    - at least 20 outcome events occur; and
    - the estimated effect is sufficiently separated from 1
      under an approximate event-based criterion.

    This is not a replacement for formal prospective power analysis.
    """

    success = []

    for _, row in results.iterrows():

        rr = row["risk_ratio"]
        events = row["events"]

        successful = (
            np.isfinite(rr)
            and events >= 20
            and abs(
                np.log(rr)
            )
            > (
                1.96
                / np.sqrt(
                    max(events, 1)
                )
            )
        )

        success.append(
            bool(successful)
        )

    return (
        float(
            np.mean(success)
        )
        if success
        else np.nan
    )


def run_repeated_rehearsals(
    population_size: int,
    exposure_prevalence: float,
    baseline_risk: float,
    true_risk_ratio: float,
    sample_target: int,
    recruitment_rate: float,
    dropout_rate: float,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeatedly execute the same proposed study.
    """

    from simulation import (
        generate_virtual_population,
        run_study_rehearsal,
    )

    true_risk_difference = (
        baseline_risk
        * true_risk_ratio
        - baseline_risk
    )

    rows = []

    for i in range(
        repetitions
    ):

        population = (
            generate_virtual_population(
                population_size=
                    population_size,
                exposure_prevalence=
                    exposure_prevalence,
                baseline_risk=
                    baseline_risk,
                true_risk_ratio=
                    true_risk_ratio,
                dropout_probability=
                    dropout_rate,
                recruitment_probability=
                    recruitment_rate,
                seed=
                    seed
                    + (i * 2),
            )
        )

        rehearsal = (
            run_study_rehearsal(
                population=
                    population,
                sample_target=
                    sample_target,
                recruitment_rate=
                    recruitment_rate,
                dropout_rate=
                    dropout_rate,
                seed=
                    seed
                    + (i * 2)
                    + 1,
            )
        )

        metrics = analyse_rehearsal(
            rehearsal["retained"],
            true_risk_ratio=
                true_risk_ratio,
            true_risk_difference=
                true_risk_difference,
        )

        rows.append(
            {
                "run": i + 1,
                "n_selected":
                    rehearsal[
                        "n_selected"
                    ],
                "n_recruited":
                    rehearsal[
                        "n_recruited"
                    ],
                "n_retained":
                    rehearsal[
                        "n_retained"
                    ],
                "events":
                    rehearsal[
                        "events"
                    ],
                **metrics,
            }
        )

    results = pd.DataFrame(
        rows
    )

    summary = pd.DataFrame(
        [
            {
                "rehearsals":
                    len(results),
                "mean_retained":
                    results[
                        "n_retained"
                    ].mean(),
                "mean_events":
                    results[
                        "events"
                    ].mean(),
                "mean_rr":
                    results[
                        "risk_ratio"
                    ].mean(),
                "mean_rr_bias_pct":
                    results[
                        "rr_bias_pct"
                    ].mean(),
                "mean_absolute_rr_error":
                    (
                        results[
                            "risk_ratio"
                        ]
                        - true_risk_ratio
                    )
                    .abs()
                    .mean(),
                "probability_80pct_power":
                    estimate_success_probability(
                        results
                    ),
                "probability_under_100_events":
                    float(
                        (
                            results[
                                "events"
                            ]
                            < 100
                        ).mean()
                    ),
            }
        ]
    )

    return (
        results,
        summary,
    )
