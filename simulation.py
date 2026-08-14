from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_probability(p: float) -> float:
    return float(np.clip(p, 1e-6, 1 - 1e-6))


def _logit(p: float) -> float:
    p = _clip_probability(p)
    return float(np.log(p / (1 - p)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def generate_virtual_population(
    population_size: int,
    exposure_prevalence: float,
    baseline_risk: float,
    true_risk_ratio: float,
    dropout_probability: float,
    recruitment_probability: float,
    seed: int,
) -> pd.DataFrame:
    """
    Generate a synthetic longitudinal population with a known
    underlying exposure-outcome relationship.
    """

    if population_size < 100:
        raise ValueError("population_size must be at least 100.")

    if not 0 < exposure_prevalence < 1:
        raise ValueError(
            "exposure_prevalence must be between 0 and 1."
        )

    if not 0 < baseline_risk < 1:
        raise ValueError(
            "baseline_risk must be between 0 and 1."
        )

    if true_risk_ratio <= 0:
        raise ValueError(
            "true_risk_ratio must be positive."
        )

    if not 0 <= dropout_probability < 1:
        raise ValueError(
            "dropout_probability must be between 0 and 1."
        )

    if not 0 < recruitment_probability <= 1:
        raise ValueError(
            "recruitment_probability must be between 0 and 1."
        )

    rng = np.random.default_rng(seed)

    age = np.clip(
        rng.normal(45, 15, population_size),
        18,
        90,
    )

    exposure = rng.binomial(
        1,
        exposure_prevalence,
        population_size,
    )

    baseline_risk = _clip_probability(
        baseline_risk
    )

    exposed_risk = _clip_probability(
        baseline_risk * true_risk_ratio
    )

    risk = np.where(
        exposure == 1,
        exposed_risk,
        baseline_risk,
    )

    outcome = rng.binomial(
        1,
        risk,
        population_size,
    )

    recruited = rng.binomial(
        1,
        recruitment_probability,
        population_size,
    )

    dropped_out = np.zeros(
        population_size,
        dtype=int,
    )

    recruited_idx = recruited == 1

    dropped_out[recruited_idx] = rng.binomial(
        1,
        dropout_probability,
        recruited_idx.sum(),
    )

    retained = (
        recruited == 1
    ) & (
        dropped_out == 0
    )

    return pd.DataFrame(
        {
            "id": np.arange(
                1,
                population_size + 1,
            ),
            "age": age,
            "exposure": exposure,
            "outcome": outcome,
            "eligible": np.ones(
                population_size,
                dtype=int,
            ),
            "recruited": recruited,
            "dropped_out": dropped_out,
            "retained": retained.astype(int),
            "follow_up_complete":
                retained.astype(int),
        }
    )


def run_virtual_cohort(
    population: pd.DataFrame,
    sample_target: int,
    recruitment_rate: float,
    dropout_rate: float,
    seed: int,
) -> pd.DataFrame:
    """
    Run one proposed recruitment and follow-up process.
    """

    rng = np.random.default_rng(seed)

    df = population.copy()

    eligible_ids = df.loc[
        df["eligible"] == 1,
        "id",
    ].to_numpy()

    rng.shuffle(eligible_ids)

    selected_ids = eligible_ids[
        : min(
            sample_target,
            len(eligible_ids),
        )
    ]

    df["selected"] = 0

    df.loc[
        df["id"].isin(selected_ids),
        "selected",
    ] = 1

    selected = df["selected"].eq(1)

    recruited = selected & (
        rng.random(len(df))
        < recruitment_rate
    )

    df["recruited"] = (
        recruited.astype(int)
    )

    dropout = np.zeros(
        len(df),
        dtype=int,
    )

    recruited_idx = recruited.to_numpy()

    dropout[recruited_idx] = rng.binomial(
        1,
        dropout_rate,
        recruited_idx.sum(),
    )

    df["dropped_out"] = dropout

    df["retained"] = (
        df["recruited"].eq(1)
        & df["dropped_out"].eq(0)
    ).astype(int)

    return df


def run_study_rehearsal(
    population: pd.DataFrame,
    sample_target: int,
    recruitment_rate: float,
    dropout_rate: float,
    seed: int,
) -> dict:
    """
    Run one complete virtual study.
    """

    study = run_virtual_cohort(
        population=population,
        sample_target=sample_target,
        recruitment_rate=recruitment_rate,
        dropout_rate=dropout_rate,
        seed=seed,
    )

    retained = study[
        study["retained"] == 1
    ].copy()

    n_selected = int(
        study["selected"].sum()
    )

    n_recruited = int(
        study["recruited"].sum()
    )

    n_retained = int(
        study["retained"].sum()
    )

    events = int(
        retained["outcome"].sum()
    )

    exposed = retained[
        retained["exposure"] == 1
    ]

    unexposed = retained[
        retained["exposure"] == 0
    ]

    risk_exposed = (
        exposed["outcome"].mean()
        if len(exposed)
        else np.nan
    )

    risk_unexposed = (
        unexposed["outcome"].mean()
        if len(unexposed)
        else np.nan
    )

    risk_ratio = (
        risk_exposed
        / risk_unexposed
        if np.isfinite(risk_exposed)
        and np.isfinite(risk_unexposed)
        and risk_unexposed > 0
        else np.nan
    )

    risk_difference = (
        risk_exposed
        - risk_unexposed
        if np.isfinite(risk_exposed)
        and np.isfinite(risk_unexposed)
        else np.nan
    )

    return {
        "study": study,
        "retained": retained,
        "n_selected": n_selected,
        "n_recruited": n_recruited,
        "n_retained": n_retained,
        "events": events,
        "risk_exposed": risk_exposed,
        "risk_unexposed": risk_unexposed,
        "risk_ratio": risk_ratio,
        "risk_difference": risk_difference,
    }
