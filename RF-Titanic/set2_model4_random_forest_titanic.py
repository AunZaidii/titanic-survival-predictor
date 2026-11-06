"""
Set 2 - Model 4: Random Forest - CLASSIFICATION
Dataset: Titanic passengers (loaded from a CSV on GitHub - needs internet)
  CSV link:      https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv
  Original data: https://www.kaggle.com/c/titanic/data
Task: predict whether a passenger survived (1) or not (0) from class, gender, age, family and fare.

A random forest is many decision trees that vote. Two sources of randomness make the trees different:
  1. Bagging: each tree trains on a random sample of the passengers, drawn with replacement.
  2. Feature sampling: each split may only consider a random subset of the features.
Each tree overfits in its own way, so averaging them cancels out much of that error.

The data, cleaning, train/test split and search grid match the decision tree script (Model 3),
so the two models can be compared fairly. See ../Comparison-Titanic/compare_dt_vs_rf_titanic.py for the comparison.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

HERE = Path(__file__).resolve().parent     # pictures are saved next to this file
N_TREES = 200                              # step 7's chart shows that accuracy has levelled off well before this
RF_COLOR, TRAIN_COLOR = "#2a78d6", "#eb6834"

# ---------------- STEP 1: LOAD DATA ----------------
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
try:
    df = pd.read_csv(URL).rename(columns={"sex": "gender"})   # the CSV's own column name
except Exception as e:
    raise SystemExit(f"Could not download the dataset (check your internet): {e}\n"
                     f"Download it manually from the CSV link in the header and use pd.read_csv('titanic.csv').")
print("=== STEP 1: Dataset ===")
print("Passengers:", len(df))
print("Survived:", df["survived"].sum(), "| Did not survive:", (df["survived"] == 0).sum())
print("Missing values (columns that have any):\n", df.isna().sum()[df.isna().sum() > 0], "\n")

# ---------------- STEP 2: PREPARE THE DATA (same as the decision tree) ----------------
data = df[["survived", "pclass", "gender", "age", "sibsp", "parch", "fare"]].copy()
data["gender"] = (data["gender"] == "female").astype(int)          # female = 1, male = 0
data["age"] = data["age"].fillna(data["age"].median())       # fill missing ages with the median age
X = data.drop(columns="survived")
y = data["survived"]
print("=== STEP 2: Prepared features ===")
print(list(X.columns), "\n")

# ---------------- STEP 3: SPLIT (same random_state, so the same passengers as the decision tree) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3, stratify=y
)
print("=== STEP 3: Split ===")
print("Train:", len(X_train), "| Test:", len(X_test), "\n")

# ---------------- STEP 4: FOREST WITH NO DEPTH LIMIT ----------------
# oob_score: each tree skips about a third of the passengers, so those can act as a built-in test set.
full = RandomForestClassifier(n_estimators=N_TREES, oob_score=True, random_state=3, n_jobs=-1).fit(X_train, y_train)
print("=== STEP 4: Forest with no depth limit ===")
print("Trees:", len(full.estimators_), "| Average tree depth: %.1f" % np.mean([t.get_depth() for t in full.estimators_]))
print("Train accuracy: %.3f" % full.score(X_train, y_train))
print("Out-of-bag accuracy: %.3f" % full.oob_score_)
print("Test accuracy:  %.3f\n" % full.score(X_test, y_test))

# ---------------- STEP 5: TUNE WITH CROSS-VALIDATION (train data only, same grid as the decision tree) ----------------
grid = GridSearchCV(
    RandomForestClassifier(n_estimators=N_TREES, random_state=3),
    param_grid={"max_depth": list(range(1, 11)), "min_samples_leaf": [1, 5, 10, 20]},
    cv=5,
    n_jobs=-1,
)
grid.fit(X_train, y_train)
print("=== STEP 5: Hyperparameter search (5-fold CV on training set) ===")
print("Best settings:", grid.best_params_)
print("Best CV accuracy: %.3f\n" % grid.best_score_)

# depth-vs-accuracy chart (min_samples_leaf fixed at the best value)
msl = grid.best_params_["min_samples_leaf"]
depths = list(range(1, 11))
tr, te = [], []
for d in depths:
    m = RandomForestClassifier(n_estimators=N_TREES, max_depth=d, min_samples_leaf=msl, random_state=3, n_jobs=-1).fit(X_train, y_train)
    tr.append(m.score(X_train, y_train)); te.append(m.score(X_test, y_test))
plt.figure(figsize=(7, 4))
plt.plot(depths, tr, marker="o", color=TRAIN_COLOR, label="Train")
plt.plot(depths, te, marker="o", color=RF_COLOR, label="Test")
plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.title("Titanic forest: depth vs accuracy")
plt.grid(axis="y", alpha=0.25); plt.legend(); plt.tight_layout()
plt.savefig(HERE / "titanic_rf_fig1_depth.png", dpi=120)
plt.close()

# ---------------- STEP 6: FINAL MODEL + EVALUATION ----------------
forest = grid.best_estimator_
pred = forest.predict(X_test)
print("=== STEP 6: Final forest ===")
print("Trees:", len(forest.estimators_), "| Depth limit:", forest.max_depth,
      "| Total leaves:", sum(t.get_n_leaves() for t in forest.estimators_))
print("Train accuracy: %.3f" % forest.score(X_train, y_train))
print("Test accuracy:  %.3f" % accuracy_score(y_test, pred))
print("\nConfusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_test, pred))
print("\n", classification_report(y_test, pred, target_names=["died", "survived"]))
print("(Baseline: always predicting 'died' would score %.3f)\n" % (1 - y_test.mean()))

# ---------------- STEP 7: HOW MANY TREES ARE ENOUGH? ----------------
# Adding trees does not cause overfitting; accuracy simply levels off once the vote is stable.
tree_counts = [1, 2, 5, 10, 25, 50, 100, 200, 300, 500]
best = grid.best_params_
cv_acc = [cross_val_score(RandomForestClassifier(n_estimators=n, random_state=3, n_jobs=-1, **best),
                          X_train, y_train, cv=5).mean() for n in tree_counts]
print("=== STEP 7: Number of trees vs CV accuracy ===")
for n, a in zip(tree_counts, cv_acc):
    print(f"{n:>4} trees: {a:.3f}")
print()
plt.figure(figsize=(7, 4))
plt.plot(tree_counts, cv_acc, marker="o", color=RF_COLOR)
plt.xscale("log"); plt.xticks(tree_counts, tree_counts); plt.minorticks_off()
plt.xlabel("Number of trees (log scale)"); plt.ylabel("Accuracy (5-fold CV on training set)")
plt.title("Titanic forest: number of trees vs accuracy")
plt.grid(axis="y", alpha=0.25); plt.tight_layout()
plt.savefig(HERE / "titanic_rf_fig4_n_trees.png", dpi=120)
plt.close()

# ---------------- STEP 8: VISUALIZE ----------------
# A forest cannot be drawn as one tree, so this shows one member. The others differ from it.
plt.figure(figsize=(18, 9))
plot_tree(forest.estimators_[0], feature_names=list(X.columns), class_names=["died", "survived"],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title(f"One of the {N_TREES} trees in the forest (top 3 levels shown)")
plt.tight_layout()
plt.savefig(HERE / "titanic_rf_fig2_one_tree.png", dpi=110)
plt.close()

imp = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== STEP 8: Feature importance (averaged over all trees) ===")
print(imp.round(3), "\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
imp.sort_values().plot(kind="barh", ax=axes[0], color=RF_COLOR); axes[0].set_title("Feature importance")
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["died", "survived"], ax=axes[1], cmap="Blues")
axes[1].set_title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(HERE / "titanic_rf_fig3_importance_confusion.png", dpi=120)
plt.close()

# ---------------- STEP 9: PREDICT NEW PASSENGERS ----------------
print("=== STEP 9: Predicting two made-up passengers (same two as the decision tree) ===")
people = pd.DataFrame(
    [[1, 1, 29, 0, 0, 100.0],     # 1st class woman, age 29
     [3, 0, 25, 0, 0, 8.0]],      # 3rd class man, age 25
    columns=X.columns)
# Individual trees were fitted on plain arrays by the forest, so they get plain arrays here too.
votes = np.mean([t.predict(people.to_numpy()) for t in forest.estimators_], axis=0)
for desc, p, pr, v in zip(["1st-class woman, 29", "3rd-class man, 25"], forest.predict(people), forest.predict_proba(people)[:, 1], votes):
    print(f"{desc}: {'survived' if p else 'died'} (probability of survival {pr:.2f}; {v * N_TREES:.0f} of {N_TREES} trees say survived)")

# ---------------- STEP 10: SAVE THE MODEL AND RESULTS FOR THE FRONTEND ----------------
joblib.dump(forest, HERE / "random_forest_model.joblib", compress=3)
results = {
    "passengers": len(df), "survived": int(df["survived"].sum()), "train_size": len(X_train), "test_size": len(X_test),
    "n_trees": N_TREES,
    "unlimited": {"avg_depth": float(np.mean([t.get_depth() for t in full.estimators_])),
                  "train_acc": full.score(X_train, y_train), "oob_acc": full.oob_score_, "test_acc": full.score(X_test, y_test)},
    "best_params": grid.best_params_, "best_cv_acc": grid.best_score_,
    "final": {"depth_limit": forest.max_depth, "total_leaves": int(sum(t.get_n_leaves() for t in forest.estimators_)),
              "train_acc": forest.score(X_train, y_train), "test_acc": accuracy_score(y_test, pred)},
    "baseline_acc": 1 - y_test.mean(),
    "confusion": confusion_matrix(y_test, pred).tolist(),
    "report": classification_report(y_test, pred, target_names=["died", "survived"], output_dict=True),
    "importance": imp.to_dict(),
    "trees_vs_cv": dict(zip(map(str, tree_counts), cv_acc)),
}
(HERE / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
print("Saved the model and results.json")
