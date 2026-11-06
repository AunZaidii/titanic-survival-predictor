"""
Set 2 - Model 3: Decision Tree - CLASSIFICATION
Dataset: Titanic passengers (loaded from a CSV on GitHub - needs internet)
  CSV link:      https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv
  Original data: https://www.kaggle.com/c/titanic/data
Task: predict whether a passenger survived (1) or not (0) from class, gender, age, family and fare.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

HERE = Path(__file__).resolve().parent     # pictures and the saved model go next to this file

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

# ---------------- STEP 2: PREPARE THE DATA ----------------
# Trees in sklearn need numbers, and no missing values.
data = df[["survived", "pclass", "gender", "age", "sibsp", "parch", "fare"]].copy()
data["gender"] = (data["gender"] == "female").astype(int)          # female = 1, male = 0
data["age"] = data["age"].fillna(data["age"].median())       # fill missing ages with the median age
X = data.drop(columns="survived")
y = data["survived"]
print("=== STEP 2: Prepared features ===")
print(list(X.columns), "\n")

# ---------------- STEP 3: SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3, stratify=y
)
print("=== STEP 3: Split ===")
print("Train:", len(X_train), "| Test:", len(X_test), "\n")

# ---------------- STEP 4: UNLIMITED TREE (overfits) ----------------
full = DecisionTreeClassifier(random_state=3).fit(X_train, y_train)
print("=== STEP 4: Tree with no depth limit ===")
print("Depth:", full.get_depth(), "| Leaves:", full.get_n_leaves())
print("Train accuracy: %.3f" % full.score(X_train, y_train))
print("Test accuracy:  %.3f\n" % full.score(X_test, y_test))

# ---------------- STEP 5: TUNE WITH CROSS-VALIDATION (train data only) ----------------
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=3),
    param_grid={"max_depth": list(range(1, 11)), "min_samples_leaf": [1, 5, 10, 20]},
    cv=5,
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
    t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=msl, random_state=3).fit(X_train, y_train)
    tr.append(t.score(X_train, y_train)); te.append(t.score(X_test, y_test))
plt.figure(figsize=(7, 4))
plt.plot(depths, tr, marker="o", label="Train"); plt.plot(depths, te, marker="o", label="Test")
plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.title("Titanic tree: depth vs accuracy")
plt.legend(); plt.tight_layout()
plt.savefig(HERE / "titanic_tree_fig1_depth.png", dpi=120)
plt.close()

# ---------------- STEP 6: FINAL MODEL + EVALUATION ----------------
tree = grid.best_estimator_
pred = tree.predict(X_test)
print("=== STEP 6: Final tree ===")
print("Depth:", tree.get_depth(), "| Leaves:", tree.get_n_leaves())
print("Train accuracy: %.3f" % tree.score(X_train, y_train))
print("Test accuracy:  %.3f" % accuracy_score(y_test, pred))
print("\nConfusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_test, pred))
print("\n", classification_report(y_test, pred, target_names=["died", "survived"]))
print("(Baseline: always predicting 'died' would score %.3f)\n" % (1 - y_test.mean()))

# ---------------- STEP 7: VISUALIZE ----------------
plt.figure(figsize=(18, 9))
plot_tree(tree, feature_names=list(X.columns), class_names=["died", "survived"],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title("Titanic decision tree (top 3 levels shown)")
plt.tight_layout()
plt.savefig(HERE / "titanic_tree_fig2_tree.png", dpi=110)
plt.close()

print("=== STEP 7: Top of the tree as text rules ===")
print(export_text(tree, feature_names=list(X.columns), max_depth=3))

imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== Feature importance ===")
print(imp.round(3), "\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
imp.sort_values().plot(kind="barh", ax=axes[0], color="steelblue"); axes[0].set_title("Feature importance")
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["died", "survived"], ax=axes[1], cmap="Blues")
axes[1].set_title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(HERE / "titanic_tree_fig3_importance_confusion.png", dpi=120)
plt.close()

# ---------------- STEP 8: PREDICT NEW PASSENGERS ----------------
print("=== STEP 8: Predicting two made-up passengers ===")
people = pd.DataFrame(
    [[1, 1, 29, 0, 0, 100.0],     # 1st class woman, age 29
     [3, 0, 25, 0, 0, 8.0]],      # 3rd class man, age 25
    columns=X.columns)
for desc, p, pr in zip(["1st-class woman, 29", "3rd-class man, 25"], tree.predict(people), tree.predict_proba(people)[:, 1]):
    print(f"{desc}: {'survived' if p else 'died'} (probability of survival {pr:.2f})")

# ---------------- STEP 9: SAVE THE MODEL AND RESULTS FOR THE FRONTEND ----------------
joblib.dump(tree, HERE / "decision_tree_model.joblib", compress=3)
results = {
    "passengers": len(df), "survived": int(df["survived"].sum()), "train_size": len(X_train), "test_size": len(X_test),
    "unlimited": {"depth": full.get_depth(), "leaves": int(full.get_n_leaves()),
                  "train_acc": full.score(X_train, y_train), "test_acc": full.score(X_test, y_test)},
    "best_params": grid.best_params_, "best_cv_acc": grid.best_score_,
    "final": {"depth": tree.get_depth(), "leaves": int(tree.get_n_leaves()),
              "train_acc": tree.score(X_train, y_train), "test_acc": accuracy_score(y_test, pred)},
    "baseline_acc": 1 - y_test.mean(),
    "confusion": confusion_matrix(y_test, pred).tolist(),
    "report": classification_report(y_test, pred, target_names=["died", "survived"], output_dict=True),
    "importance": imp.to_dict(),
    "rules": export_text(tree, feature_names=list(X.columns), max_depth=3),
}
(HERE / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
print("Saved the model and results.json")
