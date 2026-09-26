"""Write metrics, a results table and a model card."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def _fmt_ci(v, ci):
    if ci and ci[0] is not None:
        return f"{v:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"
    return f"{v:.3f}"


def save_metrics(payload: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def results_table(metrics: dict, labels: dict) -> str:
    """Markdown table of the headline test-set metrics."""
    horizons = list(next(iter(metrics.values()))["by_horizon"])
    head = (
        "| Model | Uno's C (95% CI) | Harrell's C | IBS (95% CI) | IBS skill vs KM | "
        + " | ".join(f"AUC {h}m" for h in horizons)
        + " |"
    )
    sep = "|" + "---|" * (5 + len(horizons))
    rows = [head, sep]
    for name, m in metrics.items():
        rows.append(
            f"| {labels[name]} | {_fmt_ci(m['c_index_uno'], m['c_index_uno_ci95'])} | "
            f"{m['c_index_harrell']:.3f} | {_fmt_ci(m['ibs'], m['ibs_ci95'])} | "
            f"{m['ibs_skill_vs_km']:+.1%} | " + " | ".join(f"{m['by_horizon'][h]['auc']:.3f}" for h in horizons) + " |"
        )
    return "\n".join(rows)


def save_model_card(
    path, *, metrics, labels, best, tuning, features, split_sizes, n_rows, n_events, horizons, importance
):
    m = metrics[best]
    top = ", ".join(f for f, _, _ in importance[:5])
    lines = [
        "# Model card - breast cancer survival models",
        "",
        f"_Generated automatically by `run_pipeline.py` on {date.today().isoformat()}._",
        "",
        "## Intended use",
        "Research and teaching: comparing survival models on a public SEER breast cancer extract.",
        "**Not** validated for clinical decision-making.",
        "",
        "## Data",
        f"- {n_rows} patients after cleaning, {n_events} deaths ({n_events / n_rows:.1%}); the rest are censored.",
        f"- Split (stratified by stage and outcome): train {split_sizes['train']}, "
        f"validation {split_sizes['val']}, test {split_sizes['test']}.",
        f"- Features ({len(features)}): " + ", ".join(f"`{f}`" for f in features),
        "",
        "## Models",
        "Hyper-parameters were chosen by Uno's C-index on the validation split; test was used only once.",
        "",
    ]
    for name, t in tuning.items():
        lines.append(
            f"- **{labels[name]}** - best params `{t['best_params']}`, validation C = {t['val_c_index_uno']:.3f}"
        )
    lines += [
        "",
        "## Test-set results",
        "",
        results_table(metrics, labels),
        "",
        f"Best model (chosen by validation Uno's C): **{labels[best]}** (C = {m['c_index_uno']:.3f}, IBS = {m['ibs']:.3f}). "
        f"Evaluation horizons: {', '.join(str(int(h)) for h in horizons)} months.",
        "",
        f"Most important variables (permutation, {labels[best]}): {top}.",
        "",
        "## Limitations",
        "- Single registry extract (SEER, diagnoses 2006-2010); no external validation.",
        "- Follow-up is capped at about 9 years, so long-term risk is extrapolated.",
        "- The outcome is all-cause death, not breast-cancer-specific death.",
        "- No treatment, HER2, or comorbidity data; important predictors are missing.",
        "- Confidence intervals come from bootstrapping the test set only (models are not refitted).",
    ]
    Path(path).write_text("\n".join(lines) + "\n")
