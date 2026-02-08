import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.config import Paths, CSV_NAME, TARGET_COL, RANDOM_STATE, TEST_SIZE
from src.utils import ensure_dir, header
from src.data_io import load_csv
from src.eda import basic_eda
from src.preprocess import make_target_binary_drop1, make_features
from src.classification import get_models, get_param_grids, tune_model
from src.evaluation import evaluate_classifier_binary
from src.clustering import minibatch_kmeans_fit_and_report

def main():
    paths = Paths()
    for p in [paths.outputs, paths.figures, paths.reports, paths.models]:
        ensure_dir(p)

    header("0) LOAD DATA")
    csv_path = paths.data_raw / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_csv(csv_path)

    eda_info = basic_eda(df, TARGET_COL)
    (paths.reports / "eda_summary.txt").write_text(str(eda_info), encoding="utf-8")

    df_used, y, problem_type = make_target_binary_drop1(df, TARGET_COL)
    X = make_features(df_used, TARGET_COL)

    header("3) AFTER binary_drop1 FILTERING")
    print(f"Problem: {problem_type}")
    print(f"Rows used: {len(df_used)}")
    print("Class counts (0=non-diabetic, 1=diabetic):")
    print(pd.Series(y).value_counts())

    header("4) TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    header("5) CLASSIFICATION")
    models = get_models()
    grids = get_param_grids()

    _ = evaluate_classifier_binary(
        models["DummyMostFreq"],
        X_train, y_train, X_test, y_test,
        model_name="DummyMostFreq",
        figures_dir=paths.figures,
        reports_dir=paths.reports
    )

    best_name, best_model, best_test_f1 = None, None, -1.0

    for name in ["LogReg", "RandomForest"]:
        tuned, best_params, cv_score = tune_model(models[name], grids[name], X_train, y_train, scoring="f1_macro")

        header(f"TUNING — {name}")
        print(f"Best params: {best_params}")
        print(f"Best CV f1_macro: {cv_score:.4f}")

        fitted = evaluate_classifier_binary(
            tuned, X_train, y_train, X_test, y_test,
            model_name=name,
            figures_dir=paths.figures,
            reports_dir=paths.reports
        )

        y_pred = fitted.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="macro")

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_name = name
            best_model = fitted

    if best_model is not None:
        model_path = paths.models / f"best_model_{best_name}.joblib"
        joblib.dump(best_model, model_path)

        header("BEST MODEL SAVED")
        print(f"Best model: {best_name}")
        print(f"Best test macro-F1: {best_test_f1:.4f}")
        print(f"Saved to: {model_path}")

    header("6) CLUSTERING")
    cluster_labels, best_k, _profile = minibatch_kmeans_fit_and_report(X, paths.figures, paths.reports)

    header("7) CLUSTER vs TARGET DISTRIBUTION")
    temp = df_used.reset_index(drop=True).copy()
    temp["cluster"] = cluster_labels
    temp["target"] = y.reset_index(drop=True)
    dist = pd.crosstab(temp["cluster"], temp["target"], normalize="index").round(4)
    print(dist)
    dist.to_csv(paths.reports / "cluster_vs_target_distribution.csv")

    header("DONE")
    print("Outputs saved to:")
    print(f"- Figures: {paths.figures}")
    print(f"- Reports: {paths.reports}")
    print(f"- Models : {paths.models}")

if __name__ == "__main__":
    main()
