#!/usr/bin/env python3
"""
Benchmark Suite: Oncology Tumor Diagnostic Classifier (WDBC Cytological Dataset)
Evaluates Deep Multi-Layer Perceptron (MLP), Notebook Shallow Neural Network,
and Regularized Logistic Regression across 5-Fold Stratified Cross-Validation.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def render_svg_chart(results: list[dict], output_path: Path) -> None:
    """Generates a dark-theme vector SVG chart of 5-fold cross-validated oncology metrics."""
    width, height = 880, 440
    metrics = ["roc_auc", "accuracy", "f1", "sensitivity"]
    metric_labels = {
        "roc_auc": "ROC-AUC",
        "accuracy": "Diagnostic Accuracy",
        "f1": "F1-Score",
        "sensitivity": "Sensitivity (Recall)",
    }
    colors = {
        "roc_auc": "#38bdf8",
        "accuracy": "#34d399",
        "f1": "#a78bfa",
        "sensitivity": "#f43f5e",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        "  <defs>",
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#0f172a"/>',
        '      <stop offset="100%" stop-color="#1e293b"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{width}" height="{height}" rx="12" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5"/>',
        '  <text x="32" y="42" fill="#f8fafc" font-family="Segoe UI, Inter, sans-serif" font-size="20" font-weight="700">'
        "Oncology Tumor Diagnostic Classifier — 5-Fold Stratified CV Benchmark</text>",
        '  <text x="32" y="66" fill="#94a3b8" font-family="Segoe UI, Inter, sans-serif" font-size="13">'
        "Wisconsin Diagnostic Breast Cancer (WDBC) | N = 569 Biopsies, d = 30 FNA Cytological Features</text>",
    ]

    lx = 32
    for m in metrics:
        svg.append(f'  <rect x="{lx}" y="84" width="12" height="12" rx="3" fill="{colors[m]}"/>')
        svg.append(
            f'  <text x="{lx + 18}" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" '
            f'font-size="12" font-weight="600">{metric_labels[m]}</text>'
        )
        lx += 200

    chart_left = 290
    chart_right = 810
    chart_width = chart_right - chart_left
    top_y = 125
    group_height = 94

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = chart_left + tick * chart_width
        svg.append(
            f'  <line x1="{gx:.1f}" y1="{top_y - 10}" x2="{gx:.1f}" y2="{top_y + 3 * group_height - 10}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>'
        )
        svg.append(
            f'  <text x="{gx:.1f}" y="{top_y + 3 * group_height + 10}" fill="#64748b" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="11" text-anchor="middle">{tick:.2f}</text>'
        )

    for idx, entry in enumerate(results):
        gy = top_y + idx * group_height
        model_name = entry["model"]
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 32}" fill="#f1f5f9" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="12.5" font-weight="600" text-anchor="end">{model_name}</text>'
        )
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 50}" fill="#64748b" '
            f'font-family="Consolas, monospace" font-size="11" text-anchor="end">Brier: {entry["brier_score"]:.4f} | Spec: {entry["specificity"]:.4f}</text>'
        )
        for m_idx, m in enumerate(metrics):
            val = entry[m]
            bar_w = max(2.0, val * chart_width)
            by = gy + m_idx * 18
            svg.append(
                f'  <rect x="{chart_left}" y="{by}" width="{bar_w:.1f}" height="14" rx="3" fill="{colors[m]}" opacity="0.92"/>'
            )
            svg.append(
                f'  <text x="{chart_left + bar_w + 8:.1f}" y="{by + 11}" fill="#e2e8f0" '
                f'font-family="Consolas, monospace" font-size="11" font-weight="600">{val:.4f}</text>'
            )

    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Deep MLP (64-32 ReLU + L2)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.15,
                        learning_rate_init=0.003,
                        max_iter=800,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Shallow MLP (Notebook 20-Unit)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(20,),
                        activation="relu",
                        solver="adam",
                        alpha=0.08,
                        learning_rate_init=0.003,
                        max_iter=800,
                        random_state=3,
                    ),
                ),
            ]
        ),
        "L2 Logistic Regression Baseline": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, max_iter=500, random_state=42)),
            ]
        ),
    }

    results = []
    for name, pipeline in models.items():
        oof_probs = np.zeros(len(y))
        oof_preds = np.zeros(len(y), dtype=int)
        fold_aucs = []

        for train_idx, val_idx in cv.split(X, y):
            pipeline.fit(X[train_idx], y[train_idx])
            probs = pipeline.predict_proba(X[val_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = preds
            fold_aucs.append(roc_auc_score(y[val_idx], probs))

        tn, fp, fn, tp = confusion_matrix(y, oof_preds).ravel()
        specificity = tn / max(1, (tn + fp))

        results.append(
            {
                "model": name,
                "roc_auc": round(float(roc_auc_score(y, oof_probs)), 4),
                "roc_auc_std": round(float(np.std(fold_aucs)), 4),
                "accuracy": round(float(accuracy_score(y, oof_preds)), 4),
                "precision": round(float(precision_score(y, oof_preds)), 4),
                "sensitivity": round(float(recall_score(y, oof_preds)), 4),
                "specificity": round(float(specificity), 4),
                "f1": round(float(f1_score(y, oof_preds)), 4),
                "brier_score": round(float(brier_score_loss(y, oof_probs)), 4),
            }
        )

    results.sort(key=lambda r: (r["roc_auc"], r["f1"]), reverse=True)

    root_dir = Path(__file__).resolve().parent.parent
    json_path = root_dir / "benchmarks" / "benchmark_results.json"
    svg_path = root_dir / "assets" / "benchmark_chart.svg"

    payload = {
        "project": "Oncology Tumor Diagnostic Classifier (Deep Learning Project 1)",
        "dataset": "Wisconsin Diagnostic Breast Cancer (WDBC, N=569, d=30)",
        "validation_protocol": "5-Fold Stratified Cross-Validation",
        "class_distribution": {"malignant_0": int(np.sum(y == 0)), "benign_1": int(np.sum(y == 1))},
        "results": results,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_svg_chart(results, svg_path)
    print(f"[OK] Saved oncology benchmark JSON -> {json_path}")
    print(f"[OK] Saved oncology benchmark SVG  -> {svg_path}")


if __name__ == "__main__":
    main()
