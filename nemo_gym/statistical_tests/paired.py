# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The `paired` statistical test (`--test paired`, default): a paired-difference t-test."""

from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, model_validator
from scipy import stats

from nemo_gym.comparison.loading import LoadedRun
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import MEAN_PREFIX, TASK_INDEX_KEY_NAME
from nemo_gym.statistical_tests.common import fmt, fmt_bool, fmt_p, load_run_pair, sanitize_filename_part
from nemo_gym.statistical_tests.schema import StatTestConfig, StatTestReport


class PairedTestConfig(StatTestConfig):
    test: Literal["paired"] = "paired"
    metric: Optional[List[str]] = Field(default=None, description="Metric(s) to test, e.g. `reward`.")
    margin: Optional[float] = Field(default=None, description="Non-inferiority margin, e.g. 0.01 for 1pp.")

    @model_validator(mode="after")
    def _check_margin(self) -> "PairedTestConfig":
        if self.margin is not None and self.margin <= 0:
            raise ValueError(f"--margin must be a positive number (got {self.margin}).")
        return self

    def filename_parts(self) -> List[str]:
        parts = ["two-sided" if self.margin is None else f"margin-{self.margin:g}"]
        if self.metric:
            parts.insert(0, "metric-" + "+".join(sanitize_filename_part(m) for m in self.metric))
        return parts


class PairedTestResult(BaseModel):
    metric: str
    margin: Optional[float] = None
    alpha: float
    n_pairs: int
    mean_diff: Optional[float] = None
    se: Optional[float] = None
    p_value: Optional[float] = None
    significant: Optional[bool] = None
    note: Optional[str] = None


class PairedTestReport(StatTestReport):
    results: List[PairedTestResult] = Field(default_factory=list)


def _groups_by_task(run: LoadedRun) -> dict:
    return {group[TASK_INDEX_KEY_NAME]: group for group in run.group_level_metrics if TASK_INDEX_KEY_NAME in group}


def paired_task_deltas(baseline: LoadedRun, candidate: LoadedRun, metric: str) -> Optional[List[float]]:
    key = f"{MEAN_PREFIX}{metric}"
    baseline_groups, candidate_groups = _groups_by_task(baseline), _groups_by_task(candidate)
    deltas: List[float] = []
    for task_index in sorted(set(baseline_groups) & set(candidate_groups)):
        b, c = baseline_groups[task_index].get(key), candidate_groups[task_index].get(key)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            deltas.append(float(c) - float(b))
    return deltas or None


def resolve_metrics(baseline: LoadedRun, candidate: LoadedRun, requested: Optional[List[str]]) -> Tuple[List, List]:
    if requested:
        return list(dict.fromkeys(requested)), []

    resolved: List[str] = []
    skipped: List[str] = []
    for name in sorted(set(baseline.key_metrics) | set(candidate.key_metrics)):
        if not name.startswith(MEAN_PREFIX):
            skipped.append(name)
            continue
        metric = name[len(MEAN_PREFIX) :]
        (resolved if paired_task_deltas(baseline, candidate, metric) else skipped).append(metric)
    return resolved, skipped


def run_metric(baseline: LoadedRun, candidate: LoadedRun, *, metric: str, margin: Optional[float], alpha: float):
    def result(**kw) -> PairedTestResult:
        return PairedTestResult(metric=metric, margin=margin, alpha=alpha, **kw)

    deltas = paired_task_deltas(baseline, candidate, metric)
    if not deltas:
        return result(n_pairs=0, note=f"no per-task `mean/{metric}` value on both sides for any common task.")

    n = len(deltas)
    mean_diff = sum(deltas) / n
    if n < 2:
        return result(n_pairs=n, mean_diff=mean_diff, note="only 1 paired task: cannot estimate a standard error.")

    se = (sum((d - mean_diff) ** 2 for d in deltas) / (n - 1)) ** 0.5 / n**0.5
    if se < 1e-12:
        threshold = 0.0 if margin is None else -margin
        significant = mean_diff != threshold if margin is None else mean_diff > threshold
        p_value = 0.0 if significant else 1.0
        note = "every paired delta was identical (zero variance)."
        return result(n_pairs=n, mean_diff=mean_diff, se=0.0, p_value=p_value, significant=significant, note=note)

    df = n - 1
    p_value = 2 * stats.t.sf(abs(mean_diff / se), df) if margin is None else stats.t.sf((mean_diff + margin) / se, df)
    return result(n_pairs=n, mean_diff=mean_diff, se=se, p_value=float(p_value), significant=p_value < 0.05)


def build_report(config: PairedTestConfig, command: str) -> PairedTestReport:
    pair = load_run_pair(config)

    notes: List[str] = []
    if config.metric:
        metrics = list(dict.fromkeys(config.metric))
        for metric in metrics:
            if not paired_task_deltas(pair.baseline, pair.candidate, metric):
                raise ConfigError(f"--metric '{metric}' has no per-task `mean/{metric}` value on both sides.")
    else:
        metrics, skipped = resolve_metrics(pair.baseline, pair.candidate, None)
        if not metrics:
            raise ConfigError("No key metric has per-task pairing data to test. Pass --metric explicitly.")
        if skipped:
            notes.append(f"Skipped {len(skipped)} key metric(s) with no per-task pairing data: {', '.join(skipped)}.")

    results = [
        run_metric(pair.baseline, pair.candidate, metric=metric, margin=config.margin, alpha=config.alpha)
        for metric in metrics
    ]
    return PairedTestReport(**pair.report_identity(config, command), notes=notes, results=results)


def _result_line(result: PairedTestResult) -> str:
    if result.p_value is None:
        return f"{result.metric}: {result.note}"
    return (
        f"{result.metric}: n={result.n_pairs} mean_diff={fmt(result.mean_diff)} se={fmt(result.se)} "
        f"p={fmt_p(result.p_value)} significant={fmt_bool(result.significant)}"
    )


def render_markdown(report: PairedTestReport) -> str:
    lines = [
        "gym eval stat-test: paired",
        f"Baseline:  {report.baseline_rollouts_jsonl_fpath} (agent {report.baseline_agent}, "
        f"{report.baseline_task_count} tasks)",
        f"Candidate: {report.candidate_rollouts_jsonl_fpath} (agent {report.candidate_agent}, "
        f"{report.candidate_task_count} tasks)",
        f"Generated {report.generated_at} by nemo-gym {report.nemo_gym_version}.",
        f"Command: {report.command}",
        "",
        *([_result_line(result) for result in report.results] or ["No metrics were tested."]),
        *(f"Note: {note}" for note in report.notes),
        *(f"Warning: {warning}" for warning in report.warnings),
    ]
    return "\n".join(lines) + "\n"


def summary(report: PairedTestReport, written: Sequence[Path]) -> Tuple[str, ...]:
    lines = [f"Baseline:  {report.baseline_agent}", f"Candidate: {report.candidate_agent}"]
    lines += [_result_line(result) for result in report.results]
    lines += [f"Wrote: {path}" for path in written]
    return tuple(lines)
