import json
from pathlib import Path


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    print(f"[save] Metrics saved to {path}")


def save_model_card(path, models, inputs=None, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Model Card")
    lines.append("")
    lines.append("## Models")
    for m in models:
        lines.append(f"- {m}")
    lines.append("")
    lines.append("## Inputs")
    if inputs:
        for col in inputs:
            lines.append(f"- {col}")
    else:
        lines.append("Input features not listed.")
    lines.append("")
    lines.append("## Notes")
    lines.append("This model card was generated automatically by the pipeline.")

    path.write_text("\n".join(lines))
    print(f"[save] Model card saved to {path}")