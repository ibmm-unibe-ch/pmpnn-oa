"""
metrics.py — Per-model/seed sequence-design and structure-prediction metrics
for Experiment 2 (finetuning).

For each model/seed directory under models/exp2_finetuning/ (e.g.
``oa1_seed0``, ``vanilla3``), this script reads:

  eval/seqs/*.fa
      pmpnn-gen FASTA output, one record per designed sequence, with headers
      like ``>4uoo_D, score=0.9310, recovery=0.5036, blosum=2.735, T=0.10,
      sample=1``.  ``score`` and ``recovery`` are read off each header.

  eval/structs_boltz/boltz_results_seqs_boltz/predictions/<stem>/
  confidence_<stem>_model_0.json
      Boltz structure-prediction confidence files (one per FASTA record,
      matched by the same ``<stem>_T<temp>_sample<sample>`` naming used by
      ``gen.to_boltz``).  ``ptm``, ``complex_plddt`` and ``complex_pde`` are
      read off each file.

  eval/structs_boltz/TMscore
      Per-structure TM-score/RMSD against the native structure, written by
      scripts/tmscore.py (one row per ``stem_tag``).

and reports, for the full dataset, the per-model/seed distribution of all
seven quantities as three markdown tables (mean +/- std, Q1 / median / Q3,
and the fraction of designs with TM-score > 0.5), a fourth table with the
Pearson and Spearman correlation between PTM and TM-score (over records
that have both), and a fifth section testing oa1 (oa1_seed0/1/2) vs oa3
(oa3_seed0/1/2) for a significant difference — both a seed-level
Mann-Whitney U test (3 vs 3 per-seed means) and a structure-paired Wilcoxon
signed-rank test (seed-averaged per structure, ~500 matched pairs).

A sixth table gives the mean score/recovery per model/fraction/seed, one
row per model/seed directory whose label matches the
``<model>_frac<fraction>_seed<seed>`` naming convention used by
Experiment 1's data-fraction sweep (scripts/train_exp1_data_fraction.sbatch,
scripts/gen_exp1_data_fraction.sbatch) — e.g. when run with
``--root models/exp1_data_fraction``.

The same six-table report is then repeated twice more, split by whether
the ``native`` model/seed directory's own tm_score (Boltz folding the true
sequence, scored against the true structure) is above or at/below
``--native_tm_threshold`` (default 0.5) — i.e. structures Boltz can vs.
cannot successfully re-fold from the ground-truth sequence, so both the
"clean" and "hard" subsets can be compared side by side.

Two more sections report mean +/- std score/recovery (only — no
Boltz/TM-score data exists for these yet) for every model/seed directory
with a populated ``eval/seqs_homomers`` or ``eval/seqs_heteromers``
folder, written by scripts/gen_exp2_finetuning_multimers.sbatch.

Finally, a recovery-vs-TM-score scatter is saved to ``--pareto_out``
(default ``pareto_recovery_tmscore.svg`` next to ``--out``), pooling every
record with both values across every resolved model/seed directory,
colored by architecture (seeds, and Experiment 1 data fractions, pooled
together). It overlays the strict empirical Pareto frontier (non-dominated
in both recovery and TM-score) with a bootstrapped 90% confidence band on
that frontier (``--pareto_bootstrap`` resamples, default 200) — since the
empirical frontier from a single ~500-structure sample is noisy, the band
shows how much of its shape is signal vs. sampling variance — plus
per-architecture marginal density histograms of recovery (top) and
TM-score (right). Each architecture's legend entry carries its own
Pearson r and Spearman rho between recovery and TM-score.

Usage
-----
python -m vis.metrics  [model_dir ...]  [--root models/exp2_finetuning]  [--out out.md]

With no model_dir arguments, every subdirectory of --root containing an
``eval/seqs`` folder is used (sorted alphabetically).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Context — configuration + validated CLI parsing
# ---------------------------------------------------------------------------

@dataclass
class MetricsContext:
    """All configuration needed to compute the per-model/seed metrics table."""

    root:       str
    model_dirs: list
    out:        str
    native_tm_threshold: float
    pareto_bootstrap: int
    pareto_seed:      int

    labels:     list = None
    pareto_out: str  = None

    resolved_dirs:   list = field(default=None, repr=False)
    resolved_labels: list = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not os.path.isdir(self.root):
            raise ValueError(f"root does not exist or is not a directory: {self.root!r}")

        if self.model_dirs:
            dirs = list(self.model_dirs)
        else:
            dirs = sorted(
                os.path.join(self.root, name)
                for name in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, name, "eval", "seqs"))
            )

        for d in dirs:
            if not os.path.isdir(os.path.join(d, "eval", "seqs")):
                raise ValueError(f"model_dir missing eval/seqs: {d!r}")
        if len(dirs) == 0:
            raise ValueError(
                f"No model/seed directories with an eval/seqs folder found "
                f"under {self.root!r}"
            )

        if self.labels is not None and len(self.labels) != len(dirs):
            raise ValueError(
                f"--labels must have the same length as the resolved model "
                f"directories ({len(dirs)}), got {len(self.labels)}"
            )

        out_parent = os.path.dirname(os.path.abspath(self.out)) or "."
        if not os.path.isdir(out_parent):
            raise ValueError(f"Parent directory of --out does not exist: {out_parent!r}")

        if self.pareto_out is None:
            self.pareto_out = os.path.join(out_parent, "pareto_recovery_tmscore.svg")
        pareto_out_parent = os.path.dirname(os.path.abspath(self.pareto_out)) or "."
        if not os.path.isdir(pareto_out_parent):
            raise ValueError(f"Parent directory of --pareto_out does not exist: {pareto_out_parent!r}")

        self.resolved_dirs   = dirs
        self.resolved_labels = (
            list(self.labels) if self.labels is not None
            else [os.path.basename(os.path.normpath(d)) for d in dirs]
        )

    @classmethod
    def from_args(cls) -> "MetricsContext":
        p = argparse.ArgumentParser(
            description=(
                "Compute per-model/seed mean +/- std sequence-design "
                "(score, recovery), Boltz structure-prediction confidence "
                "(ptm, complex_plddt, complex_pde) and native structural "
                "similarity (tm_score, rmsd) metrics for Experiment 2 "
                "(finetuning)."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument(
            "model_dirs", nargs="*",
            help=(
                "Model/seed directories to summarize (each must contain "
                "eval/seqs and eval/structs_boltz), e.g. "
                "'models/exp2_finetuning/oa1_seed0'. Defaults to every "
                "subdirectory of --root with an eval/seqs folder."
            ),
        )
        p.add_argument(
            "--root", default="models/exp2_finetuning",
            help="Experiment root directory, used to auto-discover model_dirs "
                 "when none are given on the command line.",
        )
        p.add_argument(
            "--out", default="models/exp2_finetuning/metrics.md",
            help="Output markdown file.",
        )
        p.add_argument(
            "--labels", nargs="+", default=None,
            help="Optional display label per resolved model_dir (same count "
                 "and order). Defaults to the directory's basename.",
        )
        p.add_argument(
            "--native_tm_threshold", type=float, default=0.5,
            help="Threshold on the native self-consistency TM-score "
                 "(the 'native' model/seed directory's own tm_score, i.e. "
                 "Boltz-from-true-sequence vs. true structure) used to "
                 "additionally report every table restricted to structures "
                 "where native folds successfully.",
        )
        p.add_argument(
            "--pareto_out", default=None,
            help="Output path for the recovery-vs-TM-score Pareto scatter "
                 "figure (.svg/.png/...). Defaults to "
                 "'pareto_recovery_tmscore.svg' next to --out.",
        )
        p.add_argument(
            "--pareto_bootstrap", type=int, default=200,
            help="Number of bootstrap resamples used to estimate the "
                 "Pareto frontier's sampling uncertainty (90%% CI band).",
        )
        p.add_argument(
            "--pareto_seed", type=int, default=0,
            help="Seed for the Pareto frontier bootstrap's random resampling.",
        )
        args = p.parse_args()
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# FASTA parsing — score / recovery, and the stem used to find the matching
# Boltz confidence file
# ---------------------------------------------------------------------------

_SCORE_RECOVERY_RE = re.compile(r"score=([0-9.eE+-]+).*?recovery=([0-9.eE+-]+)")
_TAG_RE = re.compile(r"T=([0-9.]+).*?sample=(\d+)")


def _parse_fasta_records(path: str) -> list[tuple[str, float | None, float | None, str]]:
    """
    Parse *path* and return a list of (stem_tag, score, recovery, header)
    tuples, one per FASTA record (one per header line).

    ``stem_tag`` matches the naming convention ``gen.to_boltz.record_tag``
    uses for the Boltz YAML/prediction directory derived from this record:
    ``"<fasta_stem>_T<temp>_sample<sample>"`` for headers with
    ``T=.../sample=...`` fields (pmpnn-gen's designed-sequence headers), or
    ``"<fasta_stem>_seq<index>"`` (1-indexed) for headers without them —
    e.g. the plain ``>pdbid_chain`` headers scripts/extract_native.py
    writes for the native sequences.

    ``score``/``recovery`` are ``None`` when the header has no such
    fields (again, the native case).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    headers = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line[1:].strip())

    records = []
    for index, header in enumerate(headers, start=1):
        tag_match = _TAG_RE.search(header)
        if tag_match:
            temp, sample = tag_match.groups()
            stem_tag = f"{stem}_T{temp}_sample{sample}"
        else:
            stem_tag = f"{stem}_seq{index}"

        score_match = _SCORE_RECOVERY_RE.search(header)
        score, recovery = (
            (float(score_match.group(1)), float(score_match.group(2)))
            if score_match else (None, None)
        )

        records.append((stem_tag, score, recovery, header))
    return records


# ---------------------------------------------------------------------------
# Boltz confidence JSON lookup
# ---------------------------------------------------------------------------

def _confidence_path(model_dir: str, stem_tag: str) -> str:
    return os.path.join(
        model_dir, "eval", "structs_boltz", "boltz_results_seqs_boltz",
        "predictions", stem_tag, f"confidence_{stem_tag}_model_0.json",
    )


def _load_confidence(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TMscore file lookup (scripts/tmscore.py output)
# ---------------------------------------------------------------------------

def _tmscore_path(model_dir: str) -> str:
    return os.path.join(model_dir, "eval", "structs_boltz", "TMscore")


def _load_tmscore(model_dir: str) -> dict:
    """Return {stem_tag: (tm_score, rmsd)} parsed from eval/structs_boltz/TMscore."""
    path = _tmscore_path(model_dir)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {
            row["stem_tag"]: (float(row["tm_score"]), float(row["rmsd"]))
            for row in csv.DictReader(f, delimiter="\t")
        }


# ---------------------------------------------------------------------------
# Per-model/seed aggregation
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    label:          str
    n_sequences:    int
    n_confidence:   int
    n_tmscore:      int
    score:          np.ndarray
    recovery:       np.ndarray
    ptm:            np.ndarray
    complex_plddt:  np.ndarray
    complex_pde:    np.ndarray
    tm_score:       np.ndarray
    rmsd:           np.ndarray
    # PTM/TM-score pairs, kept only for records with both values, so they
    # can be correlated against each other (np.corrcoef-style pairing).
    ptm_paired:      np.ndarray
    tm_score_paired: np.ndarray
    # {stem_tag: {metric_name: value}}, only metrics actually available for
    # that record — used to match the same structure across different
    # model/seed directories (e.g. oa1 vs oa3 significance testing).
    per_structure: dict = field(default_factory=dict)


def _collect_model_metrics(model_dir: str, label: str, seqs_dir: str | None = None) -> ModelMetrics:
    """
    Collect metrics for *model_dir*. *seqs_dir* defaults to the model's
    ``eval/seqs`` (test_monomers) FASTA directory; pass
    ``eval/seqs_homomers`` or ``eval/seqs_heteromers`` (written by
    scripts/gen_exp2_finetuning_multimers.sbatch) to instead collect
    score/recovery for those test sets. Boltz confidence and TM-score
    lookups stay keyed off *model_dir*'s own eval/structs_boltz, which only
    has data for the monomers set; for other *seqs_dir* values those
    lookups will simply find nothing, leaving ptm/complex_plddt/
    complex_pde/tm_score/rmsd empty (score/recovery are read directly from
    the FASTA headers regardless of *seqs_dir*).
    """
    seqs_dir = seqs_dir or os.path.join(model_dir, "eval", "seqs")
    fasta_files = sorted(
        os.path.join(seqs_dir, f) for f in os.listdir(seqs_dir) if f.lower().endswith(".fa")
    )
    tmscores = _load_tmscore(model_dir)

    n_records = 0
    scores, recoveries = [], []
    ptms, plddts, pdes = [], [], []
    tm_scores, rmsds = [], []
    ptms_paired, tm_scores_paired = [], []
    per_structure: dict = {}
    n_missing_confidence = 0
    n_missing_tmscore = 0

    for fasta_path in fasta_files:
        for stem_tag, score, recovery, _header in _parse_fasta_records(fasta_path):
            n_records += 1
            record: dict = {}
            if score is not None:
                scores.append(score)
                record["score"] = score
            if recovery is not None:
                recoveries.append(recovery)
                record["recovery"] = recovery

            conf = _load_confidence(_confidence_path(model_dir, stem_tag))
            if conf is None:
                n_missing_confidence += 1
            else:
                ptms.append(conf["ptm"])
                plddts.append(conf["complex_plddt"])
                pdes.append(conf["complex_pde"])
                record["ptm"] = conf["ptm"]
                record["complex_plddt"] = conf["complex_plddt"]
                record["complex_pde"] = conf["complex_pde"]

            tm_rmsd = tmscores.get(stem_tag)
            if tm_rmsd is None:
                n_missing_tmscore += 1
            else:
                tm_scores.append(tm_rmsd[0])
                rmsds.append(tm_rmsd[1])
                record["tm_score"] = tm_rmsd[0]
                record["rmsd"] = tm_rmsd[1]

            if conf is not None and tm_rmsd is not None:
                ptms_paired.append(conf["ptm"])
                tm_scores_paired.append(tm_rmsd[0])

            if record:
                per_structure[stem_tag] = record

    if n_missing_confidence:
        print(f"[metrics] {label}: {n_missing_confidence} sequence(s) missing a "
              f"Boltz confidence file, excluded from ptm/complex_plddt/complex_pde")
    if n_missing_tmscore:
        print(f"[metrics] {label}: {n_missing_tmscore} sequence(s) missing a "
              f"TMscore row, excluded from tm_score/rmsd")

    return ModelMetrics(
        label=label,
        n_sequences=n_records,
        n_confidence=len(ptms),
        n_tmscore=len(tm_scores),
        score=np.asarray(scores, dtype=np.float64),
        recovery=np.asarray(recoveries, dtype=np.float64),
        ptm=np.asarray(ptms, dtype=np.float64),
        complex_plddt=np.asarray(plddts, dtype=np.float64),
        complex_pde=np.asarray(pdes, dtype=np.float64),
        tm_score=np.asarray(tm_scores, dtype=np.float64),
        rmsd=np.asarray(rmsds, dtype=np.float64),
        ptm_paired=np.asarray(ptms_paired, dtype=np.float64),
        tm_score_paired=np.asarray(tm_scores_paired, dtype=np.float64),
        per_structure=per_structure,
    )


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def _mean_std(values: np.ndarray) -> str:
    if len(values) == 0:
        return "n/a"
    return f"{float(np.mean(values)):.3g} +/- {float(np.std(values)):.3g}"


def _quartiles(values: np.ndarray) -> str:
    if len(values) == 0:
        return "n/a"
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return f"{q1:.3g} / {q2:.3g} / {q3:.3g}"


def _format_table(all_metrics: list[ModelMetrics], summary) -> str:
    header = [
        "Model", "N", "Score", "Recovery", "PTM", "Complex pLDDT", "Complex PDE",
        "TM-score", "RMSD",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for m in all_metrics:
        n_cell = str(m.n_sequences) if m.n_confidence == m.n_sequences else (
            f"{m.n_sequences} ({m.n_confidence})"
        )
        lines.append(
            "| " + " | ".join([
                m.label,
                n_cell,
                summary(m.score),
                summary(m.recovery),
                summary(m.ptm),
                summary(m.complex_plddt),
                summary(m.complex_pde),
                summary(m.tm_score),
                summary(m.rmsd),
            ]) + " |"
        )
    return "\n".join(lines)


def _format_metrics_table(all_metrics: list[ModelMetrics]) -> str:
    return _format_table(all_metrics, _mean_std)


def _format_quartiles_table(all_metrics: list[ModelMetrics]) -> str:
    return _format_table(all_metrics, _quartiles)


_MODEL_FRAC_SEED_RE = re.compile(r"^(?P<model>.+)_frac(?P<frac>[0-9.]+)_seed(?P<seed>\d+)$")


def _parse_model_frac_seed(label: str) -> tuple | None:
    """
    Parse a ``<model>_frac<fraction>_seed<seed>`` label — the naming
    convention used by scripts/train_exp1_data_fraction.sbatch and
    scripts/gen_exp1_data_fraction.sbatch for Experiment 1's data-fraction
    sweep — into ``(model, fraction, seed)``. Returns ``None`` if *label*
    doesn't match (e.g. Experiment 2's ``oa1_seed0`` labels).
    """
    match = _MODEL_FRAC_SEED_RE.match(label)
    if match is None:
        return None
    return match.group("model"), float(match.group("frac")), int(match.group("seed"))


def _format_model_frac_seed_table(all_metrics: list[ModelMetrics]) -> str:
    """Mean +/- std score/recovery per model/fraction/seed, one row per
    model/seed directory whose label matches _parse_model_frac_seed."""
    rows = []
    for m in all_metrics:
        parsed = _parse_model_frac_seed(m.label)
        if parsed is not None:
            rows.append((parsed, m))
    if not rows:
        return (
            "n/a — no model/seed directory label matches the "
            "'<model>_frac<fraction>_seed<seed>' naming convention "
            "in the resolved set."
        )
    rows.sort(key=lambda row: row[0])

    header = ["Model", "Fraction", "Seed", "N", "Score", "Recovery"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for (model, frac, seed), m in rows:
        lines.append(
            "| " + " | ".join([
                model, f"{frac:g}", str(seed), str(len(m.score)),
                _mean_std(m.score), _mean_std(m.recovery),
            ]) + " |"
        )
    return "\n".join(lines)


def _format_score_recovery_table(all_metrics: list[ModelMetrics]) -> str:
    """Mean +/- std score/recovery only, one row per model/seed directory —
    used for test sets (homomers, heteromers) that have no Boltz/TM-score
    data, so the other five metric columns would just be a wall of n/a."""
    if not all_metrics:
        return "n/a"
    header = ["Model", "N", "Score", "Recovery"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for m in all_metrics:
        lines.append(
            "| " + " | ".join([
                m.label, str(len(m.score)), _mean_std(m.score), _mean_std(m.recovery),
            ]) + " |"
        )
    return "\n".join(lines)


def _format_tm_threshold_table(all_metrics: list[ModelMetrics], threshold: float = 0.5) -> str:
    header = ["Model", "N", f"Fraction TM-score > {threshold:g}"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for m in all_metrics:
        if len(m.tm_score) == 0:
            frac_cell = "n/a"
        else:
            frac_cell = f"{float(np.mean(m.tm_score > threshold)):.3f}"
        lines.append(f"| {m.label} | {len(m.tm_score)} | {frac_cell} |")
    return "\n".join(lines)


def _correlation_cell(r: float, p: float) -> str:
    return f"{r:.3f} (p={p:.2g})"


def _format_correlation_table(all_metrics: list[ModelMetrics]) -> str:
    """Pearson/Spearman correlation between PTM (Boltz confidence) and
    TM-score (vs. the native structure), over records with both values."""
    header = ["Model", "N", "Pearson r", "Spearman rho"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for m in all_metrics:
        n = len(m.ptm_paired)
        if n < 2:
            pearson_cell = spearman_cell = "n/a"
        else:
            pearson_r, pearson_p = sp_stats.pearsonr(m.ptm_paired, m.tm_score_paired)
            spearman_r, spearman_p = sp_stats.spearmanr(m.ptm_paired, m.tm_score_paired)
            pearson_cell = _correlation_cell(pearson_r, pearson_p)
            spearman_cell = _correlation_cell(spearman_r, spearman_p)
        lines.append(f"| {m.label} | {n} | {pearson_cell} | {spearman_cell} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OA1 vs OA3 significance testing
# ---------------------------------------------------------------------------
#
# Two complementary tests, since there are only 3 seeds per architecture but
# ~500 shared structures per seed:
#
#   - Seed-level: collapse each seed to one mean per metric, then compare the
#     3 oa1 means against the 3 oa3 means with a Mann-Whitney U test. Honest
#     about "would this hold with more random seeds", but n=3 per group
#     means it will rarely reach significance even for a real effect.
#   - Structure-paired: average each metric per structure across the 3 seeds
#     within each architecture (all models share the same test structures),
#     then run a paired Wilcoxon signed-rank test over the ~500 matched
#     structures. Much higher power, but it answers a narrower question
#     ("do these two 3-seed ensembles differ on these structures"), not
#     "does the architecture choice generalize".
#
# Report both, since agreement/disagreement between them is itself
# informative.

_OA1_LABELS = ("oa1_seed0", "oa1_seed1", "oa1_seed2")
_OA3_LABELS = ("oa3_seed0", "oa3_seed1", "oa3_seed2")

_COMPARISON_METRICS = [
    ("score",         "Score"),
    ("recovery",      "Recovery"),
    ("ptm",           "PTM"),
    ("complex_plddt", "Complex pLDDT"),
    ("complex_pde",   "Complex PDE"),
    ("tm_score",      "TM-score"),
    ("rmsd",          "RMSD"),
]


def _select_group(all_metrics: list[ModelMetrics], labels: tuple) -> list[ModelMetrics]:
    by_label = {m.label: m for m in all_metrics}
    return [by_label[label] for label in labels if label in by_label]


def _fmt_list(values: list) -> str:
    if not values:
        return "n/a"
    return ", ".join(f"{v:.3g}" for v in values)


def _seed_means(group: list[ModelMetrics], metric: str) -> list:
    means = []
    for m in group:
        values = getattr(m, metric)
        if len(values) > 0:
            means.append(float(np.mean(values)))
    return means


def _mannwhitney_p(a: list, b: list) -> float | None:
    if len(a) < 1 or len(b) < 1:
        return None
    try:
        return float(sp_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return None


def _format_seed_level_table(all_metrics: list[ModelMetrics]) -> str:
    group_a = _select_group(all_metrics, _OA1_LABELS)
    group_b = _select_group(all_metrics, _OA3_LABELS)
    header = ["Metric", "OA1 seed means", "OA3 seed means", "Mann-Whitney p"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for metric, display in _COMPARISON_METRICS:
        a_means = _seed_means(group_a, metric)
        b_means = _seed_means(group_b, metric)
        p = _mannwhitney_p(a_means, b_means)
        p_cell = f"{p:.3g}" if p is not None else "n/a"
        lines.append(f"| {display} | {_fmt_list(a_means)} | {_fmt_list(b_means)} | {p_cell} |")
    return "\n".join(lines)


def _paired_structure_values(group: list[ModelMetrics], metric: str) -> dict:
    """{stem_tag: mean(metric) across this group's seeds that have a value for it}."""
    acc: dict = {}
    for m in group:
        for stem_tag, record in m.per_structure.items():
            if metric in record:
                acc.setdefault(stem_tag, []).append(record[metric])
    return {stem_tag: float(np.mean(v)) for stem_tag, v in acc.items()}


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 1:
        return None
    try:
        return float(sp_stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return None


def _format_structure_paired_table(all_metrics: list[ModelMetrics]) -> str:
    group_a = _select_group(all_metrics, _OA1_LABELS)
    group_b = _select_group(all_metrics, _OA3_LABELS)
    header = [
        "Metric", "N pairs", "OA1 median", "OA3 median",
        "Median delta (OA3-OA1)", "Wilcoxon p",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for metric, display in _COMPARISON_METRICS:
        a_vals = _paired_structure_values(group_a, metric)
        b_vals = _paired_structure_values(group_b, metric)
        common = sorted(set(a_vals) & set(b_vals))
        if not common:
            lines.append(f"| {display} | 0 | n/a | n/a | n/a | n/a |")
            continue
        a_arr = np.array([a_vals[k] for k in common])
        b_arr = np.array([b_vals[k] for k in common])
        p = _wilcoxon_p(a_arr, b_arr)
        p_cell = f"{p:.3g}" if p is not None else "n/a"
        lines.append(
            f"| {display} | {len(common)} | {np.median(a_arr):.3g} | {np.median(b_arr):.3g} | "
            f"{np.median(b_arr - a_arr):.3g} | {p_cell} |"
        )
    return "\n".join(lines)


def _format_significance_section(all_metrics: list[ModelMetrics]) -> str:
    group_a = _select_group(all_metrics, _OA1_LABELS)
    group_b = _select_group(all_metrics, _OA3_LABELS)
    if not group_a or not group_b:
        missing = [
            label for label in (*_OA1_LABELS, *_OA3_LABELS)
            if label not in {m.label for m in all_metrics}
        ]
        return (
            f"n/a — missing model/seed director(y/ies) from the resolved set: "
            f"{', '.join(missing)}"
        )
    return (
        "### Seed-level (Mann-Whitney U on per-seed means, n=3 vs n=3)\n\n"
        + _format_seed_level_table(all_metrics) + "\n\n"
        "### Structure-paired (Wilcoxon signed-rank on seed-averaged, "
        "per-structure values)\n\n"
        + _format_structure_paired_table(all_metrics)
    )


# ---------------------------------------------------------------------------
# Native-quality filtering — split every table into structures where
# native's own Boltz self-consistency check succeeds (tm_score > threshold)
# vs. fails (tm_score <= threshold), as a complement to the full-dataset
# report: a design evaluated against a native structure Boltz itself can't
# re-fold from the true sequence is a weaker signal than one evaluated
# against a structure Boltz folds well, so the two views are reported
# side by side.
# ---------------------------------------------------------------------------

_BASE_ID_RE = re.compile(r"_(?:T[0-9.]+_sample\d+|seq\d+)$")


def _base_id(stem_tag: str) -> str:
    """Strip the '_T<temp>_sample<n>' / '_seq<n>' suffix, leaving '<pdbid_chain>'."""
    return _BASE_ID_RE.sub("", stem_tag)


def _native_quality_split(all_metrics: list[ModelMetrics], threshold: float) -> tuple:
    """
    Split base structure IDs by the 'native' model/seed's own tm_score vs.
    threshold. Returns (above, at_or_below) sets; both empty if there is no
    'native' entry (or it has no tm_score data) in all_metrics.
    """
    native = next((m for m in all_metrics if m.label == "native"), None)
    if native is None:
        return set(), set()
    above, at_or_below = set(), set()
    for stem_tag, record in native.per_structure.items():
        tm_score = record.get("tm_score")
        if tm_score is None:
            continue
        base = _base_id(stem_tag)
        (above if tm_score > threshold else at_or_below).add(base)
    return above, at_or_below


def _filter_model_metrics(m: ModelMetrics, allowed_base_ids: set) -> ModelMetrics:
    """Restrict *m* to structures whose base id is in allowed_base_ids, rebuilt
    from per_structure (so it stays consistent with the full-dataset view)."""
    filtered = {
        stem_tag: record for stem_tag, record in m.per_structure.items()
        if _base_id(stem_tag) in allowed_base_ids
    }

    def _column(key: str) -> np.ndarray:
        return np.asarray([r[key] for r in filtered.values() if key in r], dtype=np.float64)

    ptm_paired, tm_score_paired = [], []
    for r in filtered.values():
        if "ptm" in r and "tm_score" in r:
            ptm_paired.append(r["ptm"])
            tm_score_paired.append(r["tm_score"])

    return ModelMetrics(
        label=m.label,
        n_sequences=len(filtered),
        n_confidence=sum(1 for r in filtered.values() if "ptm" in r),
        n_tmscore=sum(1 for r in filtered.values() if "tm_score" in r),
        score=_column("score"),
        recovery=_column("recovery"),
        ptm=_column("ptm"),
        complex_plddt=_column("complex_plddt"),
        complex_pde=_column("complex_pde"),
        tm_score=_column("tm_score"),
        rmsd=_column("rmsd"),
        ptm_paired=np.asarray(ptm_paired, dtype=np.float64),
        tm_score_paired=np.asarray(tm_score_paired, dtype=np.float64),
        per_structure=filtered,
    )


# ---------------------------------------------------------------------------
# Full report assembly (shared between the all-structures and
# native-quality-filtered views)
# ---------------------------------------------------------------------------

def _build_report(all_metrics: list[ModelMetrics]) -> str:
    mean_std_md         = _format_metrics_table(all_metrics)
    model_frac_seed_md  = _format_model_frac_seed_table(all_metrics)
    quartiles_md        = _format_quartiles_table(all_metrics)
    tm_threshold_md = _format_tm_threshold_table(all_metrics)
    correlation_md  = _format_correlation_table(all_metrics)
    significance_md = _format_significance_section(all_metrics)
    return (
        "## Mean +/- std\n\n" + mean_std_md + "\n\n"
        "## Score / Recovery by model / fraction / seed\n\n" + model_frac_seed_md + "\n\n"
        "## Q1 / median / Q3\n\n" + quartiles_md + "\n\n"
        "## Fraction TM-score > 0.5\n\n" + tm_threshold_md + "\n\n"
        "## PTM vs TM-score correlation\n\n" + correlation_md + "\n\n"
        "## OA1 vs OA3 significance\n\n" + significance_md
    )


# ---------------------------------------------------------------------------
# Pareto frontier — recovery (x) vs TM-score (y), both maximized
# ---------------------------------------------------------------------------
#
# Sequence-design quality (recovery) and structural fidelity (TM-score) are
# measured per design, but a model that scores higher on one need not score
# higher on the other — the question this scatter answers is whether some
# models trade one off for the other, or whether one model dominates
# (higher recovery AND higher TM-score) across the board. The empirical
# frontier from a single ~500-structure sample is noisy, so its sampling
# uncertainty is estimated by bootstrap resampling the pooled per-structure
# (recovery, tm_score) pairs and recomputing the frontier each time. Points
# are colored by architecture (oa1/oa3/vanilla3/gat3), pooling seeds (and,
# for Experiment 1 labels, data fractions) together, since per-seed color
# would just split each architecture's cloud into indistinguishable shades.

_ARCHITECTURE_PREFIXES = ("oa1", "oa3", "vanilla3", "gat3")


def _architecture_label(label: str) -> str:
    """Strip a '_seed<seed>' (and, for Experiment 1 labels, '_frac<fraction>')
    suffix off *label*, leaving the bare architecture name — e.g.
    'oa1_seed0' -> 'oa1', 'oa1_frac0.1_seed0' -> 'oa1', 'vanilla3' -> 'vanilla3'
    (no suffix to strip). Labels that don't start with a known architecture
    prefix (e.g. 'native') are returned unchanged."""
    for prefix in _ARCHITECTURE_PREFIXES:
        if label == prefix or label.startswith(prefix + "_"):
            return prefix
    return label


def _pareto_recovery_tmscore_frame(all_metrics: list[ModelMetrics]) -> pd.DataFrame:
    """Pooled per-structure (recovery, tm_score, Architecture) rows, one per
    record that has both values, across every model/seed directory in
    *all_metrics* (e.g. 'native', which has no recovery, is naturally
    excluded). Seeds (and data fractions) are pooled under their shared
    architecture via _architecture_label, so color reflects architecture
    only."""
    rows = []
    for m in all_metrics:
        architecture = _architecture_label(m.label)
        for record in m.per_structure.values():
            if "recovery" in record and "tm_score" in record:
                rows.append({
                    "recovery": record["recovery"],
                    "tm_score": record["tm_score"],
                    "Architecture": architecture,
                })
    return pd.DataFrame(rows, columns=["recovery", "tm_score", "Architecture"])


def _pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Non-dominated frontier of *df* maximizing both *x_col* and *y_col*:
    scan points from highest x_col to lowest, keeping one whenever its
    y_col reaches a new high (so no kept point is dominated in y_col by an
    earlier, higher-x_col one), then return the kept points sorted by
    x_col."""
    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[False, False])
    frontier = []
    max_y = -float("inf")
    for _, row in sorted_df.iterrows():
        if row[y_col] >= max_y:
            frontier.append(row)
            max_y = row[y_col]
    return pd.DataFrame(frontier).sort_values(by=x_col)


def _bootstrap_frontier_band(
    df: pd.DataFrame, x_col: str, y_col: str,
    n_iterations: int, x_eval: np.ndarray, rng: np.random.Generator,
) -> tuple:
    """Bootstrap-resample *df* (with replacement) *n_iterations* times,
    recompute the Pareto frontier each time, linearly interpolate it onto
    the shared *x_eval* grid, and return the (5th, 50th, 95th) percentile
    curves across resamples — a 90% confidence band on where the true
    frontier lies, NaN outside the range any given resample's frontier
    covers."""
    curves = []
    for _ in range(n_iterations):
        sample = df.sample(frac=1.0, replace=True, random_state=rng.integers(2**32 - 1))
        frontier = _pareto_frontier(sample, x_col, y_col)
        if len(frontier) > 1:
            curves.append(np.interp(
                x_eval, frontier[x_col], frontier[y_col], left=np.nan, right=np.nan,
            ))
    if not curves:
        nan_curve = np.full_like(x_eval, np.nan)
        return nan_curve, nan_curve, nan_curve
    curves = np.array(curves)
    lower  = np.nanpercentile(curves, 5, axis=0)
    median = np.nanpercentile(curves, 50, axis=0)
    upper  = np.nanpercentile(curves, 95, axis=0)
    return lower, median, upper


def _plot_pareto_recovery_tmscore(
    all_metrics: list[ModelMetrics], out_path: str, n_iterations: int, seed: int,
) -> bool:
    """Scatter recovery (x) vs TM-score (y) across every model/seed
    directory with both values, overlaid with the strict empirical Pareto
    frontier and a bootstrapped 90% CI band on it, plus per-architecture
    marginal density histograms of recovery (top) and TM-score (right).
    Saves to *out_path* and returns True, or returns False (saving
    nothing) if there are fewer than 2 paired (recovery, tm_score)
    records to plot."""
    df = _pareto_recovery_tmscore_frame(all_metrics)
    if len(df) < 2:
        return False

    rng = np.random.default_rng(seed)
    x_eval = np.linspace(df["recovery"].min(), df["recovery"].max(), 100)
    lower, median, upper = _bootstrap_frontier_band(
        df, "recovery", "tm_score", n_iterations, x_eval, rng,
    )
    strict_frontier = _pareto_frontier(df, "recovery", "tm_score")

    architectures = sorted(df["Architecture"].unique())
    cycle_colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {a: cycle_colors[i % len(cycle_colors)] for i, a in enumerate(architectures)}

    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
        left=0.1, right=0.9, bottom=0.1, top=0.92, wspace=0.05, hspace=0.05,
    )
    ax       = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    recovery_bins = np.linspace(df["recovery"].min(), df["recovery"].max(), 31)
    tm_score_bins = np.linspace(df["tm_score"].min(), df["tm_score"].max(), 31)

    for architecture, group in df.groupby("Architecture"):
        color = colors[architecture]
        if len(group) >= 2:
            pearson_r, _  = sp_stats.pearsonr(group["recovery"], group["tm_score"])
            spearman_r, _ = sp_stats.spearmanr(group["recovery"], group["tm_score"])
            label = f"{architecture} (r={pearson_r:.2f}, ρ={spearman_r:.2f})"
        else:
            label = architecture
        ax.scatter(group["recovery"], group["tm_score"], label=label,
                   color=color, alpha=0.4, s=14)
        ax_histx.hist(group["recovery"], bins=recovery_bins, color=color,
                      alpha=0.4, density=True, histtype="stepfilled")
        ax_histy.hist(group["tm_score"], bins=tm_score_bins, color=color,
                      alpha=0.4, density=True, histtype="stepfilled",
                      orientation="horizontal")

    ax.plot(x_eval, median, color="red", linewidth=2, label="Median frontier (bootstrap)")
    ax.fill_between(x_eval, lower, upper, color="red", alpha=0.15, label="90% frontier CI")
    ax.plot(
        strict_frontier["recovery"], strict_frontier["tm_score"],
        color="black", linestyle=":", alpha=0.7, label="Strict empirical frontier",
    )
    ax.set_xlabel("Recovery")
    ax.set_ylabel("TM-score")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize="small")
    ax_histx.set_ylabel("Density")
    ax_histy.set_xlabel("Density")
    fig.suptitle("Recovery vs TM-score Pareto frontier")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = MetricsContext.from_args()

    print(f"[metrics] Computing metrics for {len(ctx.resolved_dirs)} model/seed "
          f"director(y/ies) under {ctx.root!r}")

    all_metrics = []
    for model_dir, label in zip(ctx.resolved_dirs, ctx.resolved_labels):
        print(f"  · {label}: {model_dir}")
        all_metrics.append(_collect_model_metrics(model_dir, label))

    full_report = _build_report(all_metrics)

    above_ids, at_or_below_ids = _native_quality_split(all_metrics, ctx.native_tm_threshold)

    def _filtered_section(ids: set) -> str:
        if not ids:
            return (
                "n/a — no 'native' model/seed directory with TM-score data found "
                "in the resolved set, or no structures fall in this range."
            )
        return _build_report([_filter_model_metrics(m, ids) for m in all_metrics])

    above_report      = _filtered_section(above_ids)
    at_or_below_report = _filtered_section(at_or_below_ids)

    def _collect_extra_seqs(seqs_subdir: str) -> list[ModelMetrics]:
        """Score/recovery for a second test set (e.g. eval/seqs_homomers,
        written by scripts/gen_exp2_finetuning_multimers.sbatch), for
        every resolved model_dir that has it populated."""
        collected = []
        for model_dir, label in zip(ctx.resolved_dirs, ctx.resolved_labels):
            seqs_dir = os.path.join(model_dir, "eval", seqs_subdir)
            if not os.path.isdir(seqs_dir) or not os.listdir(seqs_dir):
                continue
            collected.append(_collect_model_metrics(model_dir, label, seqs_dir=seqs_dir))
        return collected

    def _extra_section(seqs_subdir: str) -> str:
        collected = _collect_extra_seqs(seqs_subdir)
        if not collected:
            return (
                f"n/a — no model/seed directory has a populated "
                f"eval/{seqs_subdir} folder in the resolved set."
            )
        return _format_score_recovery_table(collected)

    homomers_report   = _extra_section("seqs_homomers")
    heteromers_report = _extra_section("seqs_heteromers")

    pareto_plotted = _plot_pareto_recovery_tmscore(
        all_metrics, ctx.pareto_out, ctx.pareto_bootstrap, ctx.pareto_seed,
    )
    if pareto_plotted:
        print(f"[metrics] saved {ctx.pareto_out}")
        pareto_section = f"![Recovery vs TM-score Pareto frontier]({os.path.basename(ctx.pareto_out)})"
    else:
        pareto_section = (
            "n/a — fewer than 2 records with both recovery and tm_score "
            "in the resolved set."
        )

    document = (
        "# All structures\n\n" + full_report + "\n\n"
        f"# Filtered: native TM-score > {ctx.native_tm_threshold:g} "
        f"(N={len(above_ids)} structures)\n\n" + above_report + "\n\n"
        f"# Filtered: native TM-score <= {ctx.native_tm_threshold:g} "
        f"(N={len(at_or_below_ids)} structures)\n\n" + at_or_below_report + "\n\n"
        "# Homomers (score/recovery)\n\n" + homomers_report + "\n\n"
        "# Heteromers (score/recovery)\n\n" + heteromers_report + "\n\n"
        "# Pareto frontier — recovery vs TM-score\n\n" + pareto_section + "\n"
    )
    with open(ctx.out, "w") as f:
        f.write(document)
    print(f"[metrics] saved {ctx.out}")
    print(document)


if __name__ == "__main__":
    main()
