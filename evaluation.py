from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from .utils import header, save_fig

def evaluate_classifier_binary(
    model,
    X_train, y_train, X_test, y_test,
    model_name: str,
    figures_dir: Path,
    reports_dir: Path
):
    header(f"CLASSIFICATION — {model_name} (binary_drop1)")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    lines = []
    lines.append(f"Model: {model_name}")
    lines.append("Problem: binary_drop1 (0=non-diabetic, 1=diabetic)")
    lines.append("")
    lines.append("Classification report:")
    lines.append(classification_report(y_test, y_pred, digits=4))
    lines.append(f"Balanced accuracy: {bal_acc:.4f}")
    lines.append(f"Macro F1:         {f1_macro:.4f}")

    print("\n".join(lines))
    (reports_dir / f"classification_{model_name}.txt").write_text("\n".join(lines), encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix — {model_name}")
    save_fig(figures_dir / f"cm_{model_name}.png")

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        pos = y_proba[:, 1]

        roc_auc = roc_auc_score(y_test, pos)
        pr_auc = average_precision_score(y_test, pos)

        extra = [
            "",
            "Probability sample (first 5 rows):",
            str(y_proba[:5]),
            f"ROC-AUC: {roc_auc:.4f}",
            f"PR-AUC:  {pr_auc:.4f}",
        ]
        print("\n".join(extra))
        with open(reports_dir / f"classification_{model_name}.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(extra) + "\n")

        RocCurveDisplay.from_predictions(y_test, pos)
        plt.title(f"ROC Curve — {model_name}")
        save_fig(figures_dir / f"roc_{model_name}.png")

        PrecisionRecallDisplay.from_predictions(y_test, pos)
        plt.title(f"Precision–Recall Curve — {model_name}")
        save_fig(figures_dir / f"pr_{model_name}.png")

    return model
