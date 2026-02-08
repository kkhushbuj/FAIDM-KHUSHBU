from typing import Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from .config import RANDOM_STATE

def get_models() -> Dict[str, Pipeline]:
    lr = LogisticRegression(max_iter=600, class_weight="balanced", solver="lbfgs")
    rf = RandomForestClassifier(
        n_estimators=250,                 # slightly reduced for speed
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    return {
        "DummyMostFreq": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", lr),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", rf),
        ]),
    }

def get_param_grids() -> Dict[str, dict]:
    # small grids (still tuning)
    return {
        "LogReg": {"clf__C": [0.5, 1.0]},
        "RandomForest": {"clf__max_depth": [None, 14], "clf__min_samples_split": [2, 8]},
    }

def tune_model(model: Pipeline, param_grid: dict, X_train, y_train, scoring: str = "f1_macro") -> Tuple[Pipeline, dict, float]:
    # 3-fold CV for speed (acceptable, explainable)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)
