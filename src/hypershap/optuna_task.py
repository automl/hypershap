"""Optuna integration for HyperSHAP.

Provides utilities to convert an optuna ``Study`` into an :class:`~hypershap.task.ExplanationTask`
so that optuna HPO results can be analysed with HyperSHAP directly.

Optuna is an **optional** dependency. Install it with::

    pip install optuna
    # or
    pip install hypershap[optuna]
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna  # type: ignore[import-untyped]
    from ConfigSpace import Configuration, ConfigurationSpace
    from sklearn.base import BaseEstimator

from hypershap.task import ExplanationTask


def _require_optuna() -> None:
    """Raise a helpful ``ImportError`` when optuna is not installed."""
    try:
        import optuna  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        msg = "optuna is required for this functionality. Install it with: pip install optuna  (or: pip install hypershap[optuna])"
        raise ImportError(msg) from exc


def _distribution_to_hp(name: str, dist: Any) -> Any:
    """Convert a single optuna distribution to a ConfigSpace hyperparameter.

    Args:
        name: The hyperparameter name.
        dist: The optuna distribution object.

    Returns:
        A ConfigSpace hyperparameter instance.

    Raises:
        TypeError: If ``dist`` is not a supported distribution type.

    """
    from ConfigSpace import Categorical, Float, Integer
    from optuna.distributions import (  # type: ignore[import-untyped]
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    if isinstance(dist, FloatDistribution):
        if dist.step is not None:
            msg = (
                f"Hyperparameter '{name}' has a step size ({dist.step}) which is not "
                "supported by ConfigSpace. The step will be ignored."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
        return Float(name, bounds=(dist.low, dist.high), log=dist.log)

    if isinstance(dist, IntDistribution):
        if dist.step != 1:
            msg = (
                f"Hyperparameter '{name}' has a step size ({dist.step}) which is not "
                "supported by ConfigSpace. The step will be ignored."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
        return Integer(name, bounds=(dist.low, dist.high), log=dist.log)

    if isinstance(dist, CategoricalDistribution):
        return Categorical(name, items=list(dist.choices))

    msg = (
        f"Unsupported optuna distribution type for hyperparameter '{name}': {type(dist).__name__}. "
        "Supported types are FloatDistribution, IntDistribution, CategoricalDistribution."
    )
    raise TypeError(msg)


def study_to_config_space(study: optuna.Study) -> ConfigurationSpace:
    """Build a :class:`~ConfigSpace.ConfigurationSpace` from an optuna :class:`~optuna.Study`.

    The hyperparameter distributions are inferred from the completed trials in the study.
    Supported distribution types:

    * :class:`~optuna.distributions.FloatDistribution` → ``Float`` hyperparameter
    * :class:`~optuna.distributions.IntDistribution` → ``Integer`` hyperparameter
    * :class:`~optuna.distributions.CategoricalDistribution` → ``Categorical`` hyperparameter

    Args:
        study: A completed (or partially completed) optuna study.

    Returns:
        A :class:`~ConfigSpace.ConfigurationSpace` covering all hyperparameters
        observed in the study's completed trials.

    Raises:
        ImportError: If ``optuna`` or ``ConfigSpace`` are not installed.
        ValueError: If no completed trials are found in the study.
        TypeError: If an unsupported optuna distribution type is encountered.

    """
    _require_optuna()

    import optuna as opt  # type: ignore[import-untyped]
    from ConfigSpace import ConfigurationSpace

    completed_trials = [t for t in study.trials if t.state == opt.trial.TrialState.COMPLETE]
    if not completed_trials:
        msg = "No completed trials found in the study."
        raise ValueError(msg)

    # Collect distributions; later trials may add new HPs (conditional case).
    distributions: dict[str, object] = {}
    for trial in completed_trials:
        for name, dist in trial.distributions.items():
            if name not in distributions:
                distributions[name] = dist

    hyperparameters = [_distribution_to_hp(name, dist) for name, dist in distributions.items()]

    cs = ConfigurationSpace()
    cs.add(hyperparameters)
    return cs


def study_to_data(
    study: optuna.Study,
    config_space: ConfigurationSpace | None = None,
    negate: bool | None = None,
) -> list[tuple[Configuration, float]]:
    """Convert completed optuna trials into hypershap-compatible ``(Configuration, float)`` pairs.

    Args:
        study: A completed (or partially completed) optuna study.
        config_space: The target :class:`~ConfigSpace.ConfigurationSpace`. If ``None``, it is
            inferred automatically via :func:`study_to_config_space`.
        negate: Whether to negate the objective values before storing them.
            HyperSHAP follows the convention that *higher values are better*, so for
            minimisation studies the values should be negated.
            If ``None`` (default), the sign is determined automatically from
            ``study.direction``.

    Returns:
        A list of ``(Configuration, float)`` tuples suitable for
        :meth:`~hypershap.task.ExplanationTask.from_data`.

    Raises:
        ImportError: If ``optuna`` or ``ConfigSpace`` are not installed.
        ValueError: If no completed trials can be converted to valid configurations.

    """
    _require_optuna()

    import optuna as opt  # type: ignore[import-untyped]
    from ConfigSpace import Configuration

    if config_space is None:
        config_space = study_to_config_space(study)

    # Auto-detect sign from study direction (study.directions is always a list in optuna >= 3).
    if negate is None:
        if len(study.directions) > 1:
            msg = "Multi-objective optuna studies are not supported. Please select a single objective before creating an ExplanationTask."
            raise ValueError(msg)
        negate = study.directions[0] == opt.study.StudyDirection.MINIMIZE

    hp_names = list(config_space.keys())
    completed_trials = [t for t in study.trials if t.state == opt.trial.TrialState.COMPLETE and t.value is not None]

    data: list[tuple[Configuration, float]] = []
    skipped = 0
    for trial in completed_trials:
        # Some HPs may be absent in trials that use conditional search (e.g. early stopping).
        missing = [name for name in hp_names if name not in trial.params]
        if missing:
            skipped += 1
            continue

        try:
            values = {name: trial.params[name] for name in hp_names}
            config = Configuration(config_space, values=values)
            trial_value: float = trial.value  # type: ignore[assignment]  # filtered above
            value = -trial_value if negate else trial_value
            data.append((config, value))
        except Exception:  # noqa: BLE001
            skipped += 1
            continue

    if skipped:
        msg = (
            f"{skipped} trial(s) were skipped because they could not be converted to valid "
            "ConfigSpace configurations (missing hyperparameter values or constraint violations)."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    if not data:
        msg = (
            "No trials could be converted to valid configurations. "
            "Check that the study has completed trials and that the hyperparameter "
            "distributions are supported."
        )
        raise ValueError(msg)

    return data


def from_optuna_study(
    study: optuna.Study,
    negate: bool | None = None,
    base_model: BaseEstimator | None = None,
) -> ExplanationTask:
    """Create an :class:`~hypershap.task.ExplanationTask` directly from an optuna study.

    This is the main entry point for the optuna integration.  It automatically:

    1. Extracts the hyperparameter search space from the study's trial distributions.
    2. Converts completed trial results into ``(Configuration, float)`` pairs.
    3. Fits a surrogate model on those pairs and wraps everything in an
       :class:`~hypershap.task.ExplanationTask`.

    For minimisation studies (``study.direction == StudyDirection.MINIMIZE``) the
    objective values are negated so that HyperSHAP's *higher-is-better* convention
    is respected.  Pass ``negate=False`` to disable this behaviour.

    Example::

        import optuna
        from hypershap import HyperSHAP
        from hypershap.optuna_task import from_optuna_study

        study = optuna.load_study(study_name="my_study", storage="sqlite:///my_study.db")
        task = from_optuna_study(study)

        hs = HyperSHAP(task)
        iv = hs.tunability()
        hs.plot_stacked_bar(iv)

    Args:
        study: A completed (or partially completed) optuna study.
        negate: Whether to negate objective values.  Defaults to ``None``, meaning
            minimisation studies are negated automatically.
        base_model: An optional sklearn-compatible regressor to use as the surrogate
            model.  Defaults to ``RandomForestRegressor`` (same as the rest of
            HyperSHAP).

    Returns:
        An :class:`~hypershap.task.ExplanationTask` ready for downstream HyperSHAP
        analysis.

    Raises:
        ImportError: If ``optuna`` is not installed.
        ValueError: If the study contains no usable completed trials.
        TypeError: If the study uses unsupported distribution types.

    """
    _require_optuna()

    config_space = study_to_config_space(study)
    data = study_to_data(study, config_space=config_space, negate=negate)
    return ExplanationTask.from_data(config_space, data, base_model=base_model)
