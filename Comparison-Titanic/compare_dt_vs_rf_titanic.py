"""
Comparison: Decision Tree (Model 3) vs Random Forest (Model 4) on the Titanic dataset.

Both models get the same data, cleaning, split and hyperparameter grid, so the model is the only difference.
They are compared in two ways:
  A. The single test set used by both model scripts (223 passengers, random_state=3).
  B. Repeated nested cross-validation: 5-fold CV repeated 5 times, giving 25 different test sets.
     The tuning is redone inside every fold, so test passengers never influence the settings.
A is simple to read, but one split can be lucky or unlucky. B is the more reliable result.
Takes about 10 minutes on an 8-core machine because B trains many forests.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV, RepeatedStratifiedKFold,
                                     cross_validate, cross_val_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             log_loss, confusion_matrix, roc_curve, ConfusionMatrixDisplay)

HERE = Path(__file__).resolve().parent     # pictures are saved next to this file
N_TREES = 200
DT_COLOR, RF_COLOR = "#eb6834", "#2a78d6"  # each model keeps the same color in every chart
PARAM_GRID = {"max_depth": list(range(1, 11)), "min_samples_leaf": [1, 5, 10, 20]}
MODELS = {
    "Decision tree": DecisionTreeClassifier(random_state=3),
    "Random forest": RandomForestClassifier(n_estimators=N_TREES, random_state=3),
}
COLORS = {"Decision tree": DT_COLOR, "Random forest": RF_COLOR}

# ---------------- STEP 1: SAME DATA, CLEANING AND SPLIT AS BOTH MODEL SCRIPTS ----------------
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
try:
    df = pd.read_csv(URL).rename(columns={"sex": "gender"})   # the CSV's own column name
except Exception as e:
    raise SystemExit(f"Could not download the dataset (check your internet): {e}")
data = df[["survived", "pclass", "gender", "age", "sibsp", "parch", "fare"]].copy()
data["gender"] = (data["gender"] == "female").astype(int)
data["age"] = data["age"].fillna(data["age"].median())
X = data.drop(columns="survived")
y = data["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3, stratify=y)
print("=== STEP 1: Data ===")
print("Train:", len(X_train), "| Test:", len(X_test), "\n")

# ---------------- STEP 2: TUNE BOTH MODELS ON THE TRAINING SET ----------------
fitted = {}
print("=== STEP 2: Tuned settings (5-fold CV on training set) ===")
for name, model in MODELS.items():
    grid = GridSearchCV(model, PARAM_GRID, cv=5, n_jobs=-1).fit(X_train, y_train)
    fitted[name] = grid.best_estimator_
    print(f"{name}: {grid.best_params_} | CV accuracy {grid.best_score_:.3f}")
print()

# ---------------- STEP 3 (A): METRICS ON THE SHARED TEST SET ----------------
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
rows, proba, preds = {}, {}, {}
for name, model in fitted.items():
    preds[name] = model.predict(X_test)
    proba[name] = model.predict_proba(X_test)[:, 1]
    rows[name] = {
        "Accuracy": accuracy_score(y_test, preds[name]),
        "Precision": precision_score(y_test, preds[name]),
        "Recall": recall_score(y_test, preds[name]),
        "F1": f1_score(y_test, preds[name]),
        "ROC AUC": roc_auc_score(y_test, proba[name]),
        "Log loss (lower is better)": log_loss(y_test, proba[name]),
        "Train accuracy": model.score(X_train, y_train),
    }
table_a = pd.DataFrame(rows)
table_a.loc["Train - test gap"] = table_a.loc["Train accuracy"] - table_a.loc["Accuracy"]
table_a["RF - DT"] = table_a["Random forest"] - table_a["Decision tree"]
print("=== STEP 3 (A): Shared test set, 223 passengers. Precision, recall and F1 are for 'survived' ===")
print(table_a.round(3).to_string(), "\n")

for name in fitted:
    print(f"{name} confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_test, preds[name]))
print()

# McNemar test: only the passengers the two models disagree on carry any evidence.
dt_right = preds["Decision tree"] == y_test.to_numpy()
rf_right = preds["Random forest"] == y_test.to_numpy()
only_dt, only_rf = int((dt_right & ~rf_right).sum()), int((rf_right & ~dt_right).sum())
n_disagree = only_dt + only_rf
p_mcnemar = binomtest(only_dt, n_disagree, 0.5).pvalue if n_disagree else 1.0
print("Passengers where exactly one model was right:")
print(f"  only the tree: {only_dt} | only the forest: {only_rf} | exact McNemar p-value: {p_mcnemar:.3f}\n")

# ---------------- STEP 4: OVERFITTING AND COST ----------------
unlimited = {
    "Decision tree, no limit": DecisionTreeClassifier(random_state=3),
    "Random forest, no limit": RandomForestClassifier(n_estimators=N_TREES, random_state=3),
}
cost_rows = {}
for label, model in {**fitted, **unlimited}.items():
    model = model.__class__(**model.get_params())   # fresh copy, so the fit below is what gets timed
    if isinstance(model, RandomForestClassifier):
        model.set_params(n_jobs=1)
    t0 = perf_counter(); model.fit(X_train, y_train); fit_ms = (perf_counter() - t0) * 1000
    t0 = perf_counter(); model.predict(X_test); pred_ms = (perf_counter() - t0) * 1000
    trees = getattr(model, "estimators_", [model])
    cost_rows[label] = {
        "Train acc": model.score(X_train, y_train),
        "Test acc": model.score(X_test, y_test),
        "Trees": len(trees),
        "Total nodes": sum(t.tree_.node_count for t in trees),
        "Fit (ms)": fit_ms,
        "Predict (ms)": pred_ms,
    }
cost = pd.DataFrame(cost_rows).T
print("=== STEP 4: Overfitting and cost (times are from one run on one CPU core) ===")
print(cost.round({"Train acc": 3, "Test acc": 3, "Trees": 0, "Total nodes": 0, "Fit (ms)": 1, "Predict (ms)": 1}).to_string(), "\n")

# ---------------- STEP 5 (B): REPEATED NESTED CROSS-VALIDATION ----------------
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=3)
SCORING = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1": "f1", "ROC AUC": "roc_auc"}
print("=== STEP 5 (B): Repeated nested CV, 25 test sets (this is the slow step) ===")
cv_scores = {}
for name, model in MODELS.items():
    inner = GridSearchCV(model, PARAM_GRID, cv=5)       # re-tunes on each outer training fold
    res = cross_validate(inner, X, y, cv=outer, scoring=SCORING, n_jobs=-1)
    cv_scores[name] = pd.DataFrame({m: res[f"test_{m}"] for m in METRICS})
diff = cv_scores["Random forest"] - cv_scores["Decision tree"]
table_b = pd.DataFrame({
    "DT mean": cv_scores["Decision tree"].mean(),
    "DT std": cv_scores["Decision tree"].std(),
    "RF mean": cv_scores["Random forest"].mean(),
    "RF std": cv_scores["Random forest"].std(),
    "RF - DT": diff.mean(),
    "RF wins (of 25)": (diff > 0).sum(),
    "DT wins (of 25)": (diff < 0).sum(),
})
print(table_b.round(3).to_string())
print("The 25 folds reuse the same passengers, so they are not independent and the win counts are descriptive only.\n")

# ---------------- STEP 6: WHY THE FOREST HANDLES DEPTH BETTER ----------------
# min_samples_leaf=1 lets every tree grow freely, which shows the overfitting clearly.
depths = list(range(1, 16))
curves = {name: {"train": [], "cv": []} for name in MODELS}
for d in depths:
    for name, model in MODELS.items():
        m = model.__class__(**{**model.get_params(), "max_depth": d, "min_samples_leaf": 1})
        if isinstance(m, RandomForestClassifier):
            m.set_params(n_jobs=-1)
        curves[name]["train"].append(m.fit(X_train, y_train).score(X_train, y_train))
        curves[name]["cv"].append(cross_val_score(m, X_train, y_train, cv=5).mean())

# ---------------- STEP 7: CHARTS ----------------
# Fig 1: every metric, on the shared test set and across the 25 test sets. Dots, so the axis need not start at 0.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
positions = np.arange(len(METRICS))[::-1]
panels = [("A. Shared test set (223 passengers)", {n: table_a.loc[METRICS, n] for n in MODELS}, None),
          ("B. Nested CV (mean and std of 25 test sets)", {n: cv_scores[n].mean() for n in MODELS},
           {n: cv_scores[n].std() for n in MODELS})]
for ax, (title, values, errors) in zip(axes, panels):
    for i, name in enumerate(MODELS):
        y_pos = positions + (0.14 if i == 0 else -0.14)
        err = errors[name].to_numpy() if errors else None
        ax.errorbar(values[name].to_numpy(), y_pos, xerr=err, fmt="o", ms=8, color=COLORS[name],
                    capsize=3, lw=1.5, label=name, markeredgecolor="white")
        for x, yp in zip(values[name].to_numpy(), y_pos):
            ax.annotate(f"{x:.3f}", (x, yp), xytext=(0, 7 if i == 0 else -14), textcoords="offset points",
                        ha="center", fontsize=8, color="#52514e")
    ax.set_yticks(positions, METRICS); ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.6, len(METRICS) - 0.4)
    ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Score (axis does not start at 0)")
axes[0].legend(loc="lower left", frameon=False)
fig.suptitle("Decision tree vs random forest: metrics", fontsize=13)
fig.tight_layout()
fig.savefig(HERE / "comparison_fig1_metrics.png", dpi=120)
plt.close(fig)

# Fig 2: confusion matrices on one shared color scale.
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
vmax = max(confusion_matrix(y_test, p).max() for p in preds.values())
for ax, name in zip(axes, MODELS):
    ConfusionMatrixDisplay.from_predictions(y_test, preds[name], display_labels=["died", "survived"], ax=ax,
                                            cmap="Blues", colorbar=False, im_kw={"vmin": 0, "vmax": vmax})
    ax.set_title(f"{name} (accuracy {table_a.loc['Accuracy', name]:.3f})", fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "comparison_fig2_confusion.png", dpi=120)
plt.close(fig)

# Fig 3: ROC curves.
fig, ax = plt.subplots(figsize=(5.8, 5.2))
ax.plot([0, 1], [0, 1], color="#9a9a94", lw=1, ls="--", label="Random guessing")
for name in MODELS:
    fpr, tpr, _ = roc_curve(y_test, proba[name])
    ax.plot(fpr, tpr, lw=2, color=COLORS[name], label=f"{name} (AUC {table_a.loc['ROC AUC', name]:.3f})")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
ax.set_title("ROC curves on the shared test set"); ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(HERE / "comparison_fig3_roc.png", dpi=120)
plt.close(fig)

# Fig 4: results for each of the 25 test sets, and the gap between the models on each one.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})
rng = np.random.default_rng(3)
for i, name in enumerate(MODELS):
    vals = cv_scores[name]["Accuracy"].to_numpy()
    axes[0].scatter(i + rng.uniform(-0.08, 0.08, len(vals)), vals, s=55, color=COLORS[name], edgecolor="white", alpha=0.85)
    axes[0].hlines(vals.mean(), i - 0.25, i + 0.25, color="#0b0b0b", lw=2)
    axes[0].annotate(f"mean {vals.mean():.3f}", (i + 0.27, vals.mean()), va="center", fontsize=9, color="#52514e")
axes[0].axhline(table_a.loc["Accuracy", "Decision tree"], color="#9a9a94", ls=":", lw=1.5)
axes[0].annotate("Tree on the shared test set", (-0.45, table_a.loc["Accuracy", "Decision tree"]), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color="#52514e")
axes[0].set_xticks([0, 1], list(MODELS)); axes[0].set_xlim(-0.5, 1.7)
axes[0].set_ylabel("Test accuracy"); axes[0].set_title("Accuracy on each of the 25 test sets", fontsize=11)
axes[0].grid(axis="y", alpha=0.25); axes[0].spines[["top", "right"]].set_visible(False)

gaps = np.sort(diff["Accuracy"].to_numpy())
axes[1].bar(range(1, 26), gaps, color=[RF_COLOR if g > 0 else DT_COLOR for g in gaps], width=0.8)
axes[1].axhline(0, color="#0b0b0b", lw=1)
axes[1].set_xlabel("The 25 test sets, sorted by gap"); axes[1].set_ylabel("RF accuracy - DT accuracy")
axes[1].set_title(f"Forest better on {(gaps > 0).sum()}, tree better on {(gaps < 0).sum()}, "
                  f"tied on {(gaps == 0).sum()}", fontsize=11)
axes[1].grid(axis="y", alpha=0.25); axes[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "comparison_fig4_stability.png", dpi=120)
plt.close(fig)

# Fig 5: feature importance.
imp = pd.DataFrame({name: m.feature_importances_ for name, m in fitted.items()}, index=X.columns)
imp = imp.sort_values("Random forest")
print("=== Feature importance ===")
print(imp.sort_values("Random forest", ascending=False).round(3).to_string(), "\n")
fig, ax = plt.subplots(figsize=(7, 4.2))
pos = np.arange(len(imp))
for i, name in enumerate(MODELS):
    bars = ax.barh(pos + (0.2 if i == 0 else -0.2), imp[name], height=0.36, color=COLORS[name], label=name)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color="#52514e")
ax.set_yticks(pos, imp.index); ax.set_xlabel("Importance (each model's values sum to 1)")
ax.set_title("Feature importance"); ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "comparison_fig5_importance.png", dpi=120)
plt.close(fig)

# Fig 6: depth curves for both models.
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for name in MODELS:
    ax.plot(depths, curves[name]["train"], color=COLORS[name], lw=2, ls="--", marker="o", ms=4, label=f"{name}, train")
    ax.plot(depths, curves[name]["cv"], color=COLORS[name], lw=2, marker="o", ms=4, label=f"{name}, CV")
ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy")
ax.set_title("Deeper models: the tree overfits more than the forest (min_samples_leaf = 1)", fontsize=11)
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "comparison_fig6_depth.png", dpi=120)
plt.close(fig)

print("Saved 6 comparison charts")

# ---------------- STEP 8: SAVE THE RESULTS FOR THE FRONTEND ----------------
results = {
    "test_size": len(X_test),
    "shared_test": table_a.to_dict(orient="index"),
    "mcnemar": {"only_dt": only_dt, "only_rf": only_rf, "p_value": p_mcnemar},
    "cost": cost.to_dict(orient="index"),
    "nested_cv": table_b.to_dict(orient="index"),
    "accuracy_gaps": {"rf_better": int((gaps > 0).sum()), "dt_better": int((gaps < 0).sum()), "tied": int((gaps == 0).sum())},
    "importance": imp.to_dict(orient="index"),
}
(HERE / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
print("Saved results.json")
