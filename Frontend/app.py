"""
Streamlit frontend for the Titanic decision tree and random forest.

It only reads what the three analysis folders produced (saved models, results.json, charts, run logs),
so nothing is trained when the app starts. Run from the project root:  streamlit run Frontend/app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DT_DIR, RF_DIR, CMP_DIR = ROOT / "DT-Titanic", ROOT / "RF-Titanic", ROOT / "Comparison-Titanic"
DT_SCRIPT = DT_DIR / "set2_model3_decision_tree_titanic.py"
RF_SCRIPT = RF_DIR / "set2_model4_random_forest_titanic.py"
CMP_SCRIPT = CMP_DIR / "compare_dt_vs_rf_titanic.py"

CLASSES = {"1st class": 1, "2nd class": 2, "3rd class": 3}
DEFAULTS = {"pclass": "2nd class", "gender": "Female", "age": 28.0, "sibsp": 0, "parch": 0, "fare": 14.45}
PRESETS = {  # film characters; the family counts and fares are approximate
    "Rose": {"pclass": "1st class", "gender": "Female", "age": 17.0, "sibsp": 0, "parch": 1, "fare": 100.0},
    "Jack": {"pclass": "3rd class", "gender": "Male", "age": 20.0, "sibsp": 0, "parch": 0, "fare": 8.05},
}


@st.cache_resource
def load_models():
    return joblib.load(DT_DIR / "decision_tree_model.joblib"), joblib.load(RF_DIR / "random_forest_model.joblib")


@st.cache_data
def load_results():
    return tuple(json.loads((d / "results.json").read_text(encoding="utf-8")) for d in (DT_DIR, RF_DIR, CMP_DIR))


# ---------------- Small display helpers ----------------
def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def html(markup):
    st.html(markup)


def kpis(items):
    cards = "".join(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                    f'<div class="note">{note}</div></div>' for label, value, note in items)
    html(f'<div class="kpis">{cards}</div>')


def step(number, title, body):
    html(f'<div class="step"><div class="num">{number}</div><div><h4>{title}</h4><p>{body}</p></div></div>')


def picture(path, caption=None):
    st.image(str(path), caption=caption, width="stretch")


def table(headers, rows, best=None):
    """rows are lists of strings. best maps (row index, column index) cells to highlight."""
    best = best or set()
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f'<td class="best">{c}</td>' if (i, j) in best else f"<td>{c}</td>"
                                    for j, c in enumerate(row)) + "</tr>" for i, row in enumerate(rows))
    html(f'<div class="table-wrap"><table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def report_table(report):
    rows = [[label.capitalize(), f'{report[label]["precision"]:.3f}', f'{report[label]["recall"]:.3f}',
             f'{report[label]["f1-score"]:.3f}', f'{report[label]["support"]:.0f}'] for label in ("died", "survived")]
    table(["Class", "Precision", "Recall", "F1", "Passengers"], rows)


def extras(log_path, script_path, extra=None):
    if extra:
        with st.expander(extra[0]):
            st.code(extra[1], language=None)
    with st.expander("Full console output from the script"):
        st.code(log_path.read_text(encoding="utf-8"), language=None)
    with st.expander("Source code"):
        st.code(script_path.read_text(encoding="utf-8"), language="python")


# ---------------- Predictor: input panel ----------------
def apply_preset(key, name):
    for field, value in PRESETS[name].items():
        st.session_state[f"{key}_{field}"] = value
    st.session_state[f"{key}_passenger"] = dict(PRESETS[name])
    st.session_state[f"{key}_name"] = name


def predictor(key, render_explanation, model, title, subtitle):
    for field, value in DEFAULTS.items():
        st.session_state.setdefault(f"{key}_{field}", value)

    with st.container(key=f"panel_{key}"):
        html(f'<p class="panel-title">{title}</p><p class="panel-sub">{subtitle}</p>')
        left, right = st.columns([5, 6], gap="medium")
        with left:
            html('<p class="preset-label">Or try a famous passenger from the film:</p>')
            cols = st.columns(len(PRESETS))
            for col, name in zip(cols, PRESETS):
                col.button(name, key=f"{key}_preset_{name}", on_click=apply_preset, args=(key, name), width="stretch")
            with st.form(f"{key}_form", border=False):
                a, b = st.columns(2)
                a.selectbox("Ticket class", list(CLASSES), key=f"{key}_pclass")
                b.selectbox("Gender", ["Female", "Male"], key=f"{key}_gender")
                a, b = st.columns(2)
                a.number_input("Age (years)", 0.0, 80.0, step=1.0, format="%.0f", key=f"{key}_age",
                               help="The passengers in the data were up to 80 years old.")
                b.number_input("Fare (1912 pounds)", 0.0, 512.33, step=1.0, format="%.2f", key=f"{key}_fare",
                               help="The fares in the data run from 0 to 512.33. The median is 14.45.")
                a, b = st.columns(2)
                a.number_input("Siblings or spouse aboard", 0, 8, step=1, key=f"{key}_sibsp")
                b.number_input("Parents or children aboard", 0, 6, step=1, key=f"{key}_parch")
                if st.form_submit_button("Predict my fate", width="stretch"):
                    st.session_state[f"{key}_passenger"] = {f: st.session_state[f"{key}_{f}"] for f in DEFAULTS}
                    st.session_state[f"{key}_name"] = "Custom passenger"
        with right:
            passenger = st.session_state.get(f"{key}_passenger")
            if passenger is None:
                html('<div class="pass"><div class="pass-head"><span>RMS TITANIC</span><span>BOARDING PASS</span></div>'
                     '<div class="pass-empty"><b>Your ticket is not issued yet</b>'
                     'Enter a passenger on the left, then press Predict.</div></div>')
            else:
                x = to_features(passenger, model)
                proba = float(model.predict_proba(x)[0, 1])
                survived = bool(model.predict(x)[0])
                html(boarding_pass(passenger, st.session_state[f"{key}_name"], proba, survived,
                                   render_explanation(x, proba, survived)))


def to_features(p, model):
    row = {"pclass": CLASSES[p["pclass"]], "gender": int(p["gender"] == "Female"), "age": p["age"],
           "sibsp": p["sibsp"], "parch": p["parch"], "fare": p["fare"]}
    return pd.DataFrame([row])[list(model.feature_names_in_)]


def boarding_pass(p, name, proba, survived, explanation):
    state = "survived" if survived else "died"
    stamp = "LIKELY SURVIVED" if survived else "LIKELY DID NOT SURVIVE"
    fields = [("CLASS", p["pclass"].split()[0]), ("GENDER", p["gender"]), ("AGE", f'{p["age"]:g}'),
              ("FARE", f'£{p["fare"]:.2f}'), ("SIB / SPOUSE", p["sibsp"]), ("PARENT / CHILD", p["parch"])]
    grid = "".join(f"<div><small>{k}</small><b>{v}</b></div>" for k, v in fields)
    return f"""
    <div class="pass">
      <div class="pass-head"><span>RMS TITANIC &middot; {name.upper()}</span><span>BOARDING PASS</span></div>
      <div class="pass-body">
        <div class="pass-grid">{grid}</div>
        <div class="tear"></div>
        <div class="verdict">
          <span class="stamp {state}">{stamp}</span>
          <span class="prob"><b>{pct(proba, 0)}</b><span>chance of survival</span></span>
        </div>
        <div class="meter"><i class="{state}" style="width:{proba * 100:.1f}%"></i><em></em></div>
        <div class="meter-scale"><span>0%</span><span>50%</span><span>100%</span></div>
        <div class="tear"></div>
        {explanation}
      </div>
    </div>"""


# ---------------- Predictor: how each model reached its answer ----------------
def describe_split(feature, threshold, went_left):
    if feature == "gender":
        return "Female" if not went_left else "Male"
    if feature == "pclass":
        if threshold < 2:
            return "1st class" if went_left else "2nd or 3rd class"
        return "1st or 2nd class" if went_left else "3rd class"
    label = {"age": "Age", "fare": "Fare", "sibsp": "Siblings or spouse aboard", "parch": "Parents or children aboard"}[feature]
    return f"{label} {'&le;' if went_left else '&gt;'} {threshold:.2f}".replace(".00", "")


def tree_route(tree):
    def render(x, proba, survived):
        t = tree.tree_
        nodes = tree.decision_path(x).indices
        names = list(tree.feature_names_in_)
        items = ""
        for node, child in zip(nodes[:-1], nodes[1:]):
            feature = names[t.feature[node]]
            text = describe_split(feature, t.threshold[node], child == t.children_left[node])
            share, count = t.value[child][0][1], int(t.n_node_samples[child])
            items += f"<li><b>{text}</b><small>{pct(share, 0)} of the {count} training passengers here survived</small></li>"
        leaf_n = int(t.n_node_samples[nodes[-1]])
        return (f'<p class="how-title">The route this passenger took through the tree</p><ol class="route">{items}</ol>'
                f'<p class="fine">The chance shown is the survival share among the {leaf_n} training passengers who ended '
                f'on the same branch. Every feature not on this route had no effect on the answer.</p>')
    return render


def forest_votes(forest, n_trees):
    def render(x, proba, survived):
        arr = x.to_numpy()   # the forest's trees were fitted on plain arrays
        tree_p = np.array([t.predict_proba(arr)[0, 1] for t in forest.estimators_])
        yes = int((tree_p > 0.5).sum())
        no = n_trees - yes
        order = np.argsort(-tree_p)
        cells = "".join(f'<div class="{"s" if tree_p[i] > 0.5 else "d"}" '
                        f'title="Tree {i + 1}: {pct(tree_p[i], 0)} chance of survival"></div>' for i in order)
        note = ""
        if (yes > no) != survived and yes != no:
            note = " Here the head count and the final answer differ, because some trees were much more confident than others."
        return (f'<p class="how-title">How the {n_trees} trees voted</p><div class="votes">{cells}</div>'
                f'<div class="legend"><span><i class="s"></i>Survived: <b>{yes}</b> trees</span>'
                f'<span><i class="d"></i>Did not survive: <b>{no}</b> trees</span></div>'
                f'<p class="fine">The trees\' own estimates range from {pct(tree_p.min(), 0)} to {pct(tree_p.max(), 0)}. '
                f'The final chance is their average, not a head count.{note} Hover over a square to see one tree.</p>')
    return render


# ---------------- Tabs ----------------
def decision_tree_tab(tree, r):
    f, u = r["final"], r["unlimited"]
    html('<p class="lede">One tree that asks a short series of yes/no questions. Every prediction can be traced '
         'from top to bottom, so you can see exactly why it gave its answer.</p>')
    kpis([("Test accuracy", pct(f["test_acc"]), f'{r["test_size"]} unseen passengers'),
          ("Cross-validated accuracy", pct(r["best_cv_acc"]), "5 folds on the training set"),
          ("Baseline", pct(r["baseline_acc"]), "always predicting “died”"),
          ("Tree depth", f["depth"], f'{f["leaves"]} leaves'),
          ("Min samples per leaf", r["best_params"]["min_samples_leaf"], "chosen by grid search")])
    predictor("dt", tree_route(tree), tree, "Would the tree let you survive?",
              "Enter any passenger. The saved decision tree answers instantly and shows its reasoning.")

    html('<div class="section-title">How it was built</div>')
    step(1, "Load and clean the data",
         f'{r["passengers"]} passengers, of whom {r["survived"]} survived. Six features were kept: class, gender, age, '
         'siblings or spouse, parents or children, and fare. Gender was encoded as 0 or 1, and missing ages were filled with the median.')
    step(2, "Split the data", f'{r["train_size"]} passengers for training and {r["test_size"]} held back for the final test, '
         'keeping the same survival ratio in both.')
    step(3, "Grow a tree with no limits",
         f'It reached depth {u["depth"]} with {u["leaves"]} leaves and scored {pct(u["train_acc"])} on training data '
         f'but only {pct(u["test_acc"])} on the test set. It had memorised passengers rather than learned a pattern.')
    step(4, "Tune it with cross-validation",
         f'A grid search over 40 settings picked max_depth = {r["best_params"]["max_depth"]} and '
         f'min_samples_leaf = {r["best_params"]["min_samples_leaf"]}. It used training data only, so the test set stayed unseen.')
    picture(DT_DIR / "titanic_tree_fig1_depth.png", "Accuracy on training and test data as the tree gets deeper")
    step(5, "Read the final tree",
         f'The tuned tree has only {f["leaves"]} leaves, so it fits on one page. Its first question is about gender, '
         'and the next ones are about age and class for men, and class and fare for women.')
    picture(DT_DIR / "titanic_tree_fig2_tree.png", "The final decision tree")
    step(6, "Evaluate on the test set",
         f'Test accuracy is {pct(f["test_acc"])} against a {pct(r["baseline_acc"])} baseline.')
    picture(DT_DIR / "titanic_tree_fig3_importance_confusion.png", "Feature importance and test-set confusion matrix")
    report_table(r["report"])
    if f["test_acc"] > r["best_cv_acc"]:
        html(f'<div class="callout"><b>Read this number with care.</b> The test accuracy ({pct(f["test_acc"])}) is higher than '
             f'the training accuracy ({pct(f["train_acc"])}) and the cross-validated accuracy ({pct(r["best_cv_acc"])}). '
             'That usually means this test set happens to be an easy one, so the cross-validated figure is the safer estimate. '
             'The Comparison tab tests this across 25 different test sets.</div>')
    extras(DT_DIR / "run_log.txt", DT_SCRIPT, ("The tree's rules as text", r["rules"]))


def random_forest_tab(forest, r):
    f, u = r["final"], r["unlimited"]
    n = r["n_trees"]
    html(f'<p class="lede">{n} decision trees that each see a random sample of passengers and a random subset of features, '
         'then average their answers. Each tree makes different mistakes, and averaging cancels out many of them.</p>')
    kpis([("Test accuracy", pct(f["test_acc"]), f'{r["test_size"]} unseen passengers'),
          ("Cross-validated accuracy", pct(r["best_cv_acc"]), "5 folds on the training set"),
          ("Baseline", pct(r["baseline_acc"]), "always predicting “died”"),
          ("Trees", n, f'{f["total_leaves"]:,} leaves in total'),
          ("Max depth", f["depth_limit"], f'min {r["best_params"]["min_samples_leaf"]} samples per leaf')])
    predictor("rf", forest_votes(forest, n), forest, f"Put your fate to a vote of {n} trees",
              "Enter any passenger. Every tree in the saved forest gives its own answer.")

    html('<div class="section-title">How it was built</div>')
    step(1, "Use the same data and split as the decision tree",
         f'The same {r["passengers"]} passengers, cleaning and random seed, giving the same {r["train_size"]} training and '
         f'{r["test_size"]} test passengers. The model is the only thing that changes.')
    step(2, "Grow a forest with no limits",
         f'With {n} unrestricted trees (average depth {u["avg_depth"]:.1f}), the forest scored {pct(u["train_acc"])} on training data, '
         f'{pct(u["oob_acc"])} out-of-bag and {pct(u["test_acc"])} on the test set. '
         'Out-of-bag accuracy tests each tree on the passengers it did not train on.')
    step(3, "Tune it with the same grid",
         f'The search picked max_depth = {r["best_params"]["max_depth"]} and min_samples_leaf = {r["best_params"]["min_samples_leaf"]}, '
         f'with a cross-validated accuracy of {pct(r["best_cv_acc"])}.')
    picture(RF_DIR / "titanic_rf_fig1_depth.png", "Accuracy on training and test data as the trees get deeper")
    counts = r["trees_vs_cv"]
    step(4, "Check how many trees are enough",
         f'Cross-validated accuracy goes from {pct(counts["1"])} with 1 tree to {pct(counts["50"])} with 50, {pct(counts["200"])} with 200 '
         f'and {pct(counts["500"])} with 500. More trees steady the vote, and accuracy levels off instead of overfitting.')
    picture(RF_DIR / "titanic_rf_fig4_n_trees.png", "Number of trees against cross-validated accuracy")
    step(5, "Look inside one tree",
         f'A forest cannot be drawn as a single tree, so this shows one of the {n}. The others differ from it.')
    picture(RF_DIR / "titanic_rf_fig2_one_tree.png", "One tree from the forest, top 3 levels")
    step(6, "Evaluate on the test set",
         f'Test accuracy is {pct(f["test_acc"])} against a {pct(r["baseline_acc"])} baseline.')
    picture(RF_DIR / "titanic_rf_fig3_importance_confusion.png", "Feature importance and test-set confusion matrix")
    report_table(r["report"])
    if f["test_acc"] > r["best_cv_acc"]:
        html(f'<div class="callout"><b>Read this number with care.</b> The test accuracy ({pct(f["test_acc"])}) is higher than '
             f'the cross-validated accuracy ({pct(r["best_cv_acc"])}), which suggests this test set happens to be an easy one. '
             'The Comparison tab tests this across 25 different test sets.</div>')
    extras(RF_DIR / "run_log.txt", RF_SCRIPT)


def comparison_tab(r):
    a, b, m, cost, gaps = r["shared_test"], r["nested_cv"], r["mcnemar"], r["cost"], r["accuracy_gaps"]
    dt, rf = "Decision tree", "Random forest"
    acc = b["Accuracy"]
    auc = b["ROC AUC"]
    html(VERDICT.format(rf=pct(acc["RF mean"]), dt=pct(acc["DT mean"]), gap=f'{acc["RF - DT"] * 100:.1f}',
                        spread=f'{acc["DT std"] * 100:.1f}', rf_auc=f'{auc["RF mean"]:.3f}', dt_auc=f'{auc["DT mean"]:.3f}',
                        auc_wins=f'{auc["RF wins (of 25)"]:.0f}'))
    kpis([("Forest accuracy wins", f'{gaps["rf_better"]} of 25', f'the tree won {gaps["dt_better"]}, tied {gaps["tied"]}'),
          ("Forest ROC AUC wins", f'{auc["RF wins (of 25)"]:.0f} of 25', f'the tree won {auc["DT wins (of 25)"]:.0f}'),
          ("Tree size", f'{cost[dt]["Total nodes"]:.0f} nodes', "readable by hand"),
          ("Forest size", f'{cost[rf]["Total nodes"]:,.0f} nodes', f'across {cost[rf]["Trees"]:.0f} trees')])

    html('<div class="section-title">The fair test: 25 different test sets</div>'
         '<p class="section-sub">Repeated nested cross-validation. Settings are re-tuned inside every fold, so test passengers never influence them.</p>')
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    rows, best = [], set()
    for i, name in enumerate(metrics):
        s = b[name]
        rows.append([name, f'{s["DT mean"]:.3f} ± {s["DT std"]:.3f}', f'{s["RF mean"]:.3f} ± {s["RF std"]:.3f}',
                     f'{s["RF - DT"]:+.3f}', f'{s["RF wins (of 25)"]:.0f}', f'{s["DT wins (of 25)"]:.0f}'])
        best.add((i, 2 if s["RF mean"] > s["DT mean"] else 1))
    table(["Metric", "Decision tree", "Random forest", "RF − DT", "RF wins", "DT wins"], rows, best)
    html('<p class="fine">Mean ± standard deviation. The better mean is shaded. The 25 folds reuse the same passengers, '
         'so they are not independent and the win counts are descriptive only.</p>')
    picture(CMP_DIR / "comparison_fig4_stability.png", "Accuracy on each of the 25 test sets, and the gap between the models")
    picture(CMP_DIR / "comparison_fig1_metrics.png", "All five metrics on the shared test set and across the 25 test sets")

    html(f'<div class="section-title">The single shared test set</div>'
         f'<p class="section-sub">The {r["test_size"]} passengers used by both model scripts.</p>')
    labels = [("Accuracy", True), ("Precision", True), ("Recall", True), ("F1", True), ("ROC AUC", True),
              ("Log loss (lower is better)", False), ("Train accuracy", None), ("Train - test gap", None)]
    rows, best = [], set()
    for i, (name, higher) in enumerate(labels):
        s = a[name]
        rows.append([name.replace("Train - test gap", "Train − test gap"), f"{s[dt]:.3f}", f"{s[rf]:.3f}", f'{s["RF - DT"]:+.3f}'])
        if higher is not None and s[dt] != s[rf]:
            best.add((i, 2 if (s[rf] > s[dt]) == higher else 1))
    table(["Metric", dt, rf, "RF − DT"], rows, best)
    significance = "not statistically significant" if m["p_value"] >= 0.05 else "statistically significant"
    html(f'<div class="callout"><b>Why one test set is not enough.</b> Here the tree has the higher accuracy ({pct(a["Accuracy"][dt])} against '
         f'{pct(a["Accuracy"][rf])}), the opposite of the 25-set average. The models disagreed on only {m["only_dt"] + m["only_rf"]} passengers: '
         f'only the tree was right on {m["only_dt"]}, and only the forest on {m["only_rf"]}. An exact McNemar test gives p = {m["p_value"]:.3f}, '
         f'which is {significance} at the 5% level.</div>')
    left, right = st.columns(2)
    with left:
        picture(CMP_DIR / "comparison_fig2_confusion.png", "Confusion matrices on the same color scale")
    with right:
        picture(CMP_DIR / "comparison_fig3_roc.png", "ROC curves")

    u_dt, u_rf = cost["Decision tree, no limit"], cost["Random forest, no limit"]
    html(f'<div class="section-title">Why the models behave differently</div>'
         f'<p class="section-sub">With no depth limit, both reach {pct(u_dt["Train acc"])} on training data, but the tree falls to '
         f'{pct(u_dt["Test acc"])} on the test set while the forest holds {pct(u_rf["Test acc"])}. Averaging many trees absorbs the overfitting that sinks a single one. '
         f'The forest also spreads importance more evenly, because feature sampling makes its trees use more than just the strongest features.</p>')
    left, right = st.columns(2)
    with left:
        picture(CMP_DIR / "comparison_fig6_depth.png", "As depth grows, the tree overfits more than the forest")
    with right:
        picture(CMP_DIR / "comparison_fig5_importance.png", "Feature importance in each model")

    html('<div class="section-title">Overfitting and cost</div>')
    rows = [[k, pct(v["Train acc"]), pct(v["Test acc"]), f'{v["Trees"]:.0f}', f'{v["Total nodes"]:,.0f}',
             f'{v["Fit (ms)"]:,.1f}', f'{v["Predict (ms)"]:.1f}'] for k, v in cost.items()]
    table(["Model", "Train acc", "Test acc", "Trees", "Total nodes", "Fit (ms)", "Predict (ms)"], rows)
    html('<p class="fine">Times are from one run on one CPU core, so treat them as rough sizes rather than exact values.</p>')
    extras(CMP_DIR / "run_log.txt", CMP_SCRIPT)


VERDICT = """<div class="banner"><small>THE VERDICT</small>
<h3>The forest wins, but only just</h3>
<p>Across 25 test sets the forest averaged {rf} accuracy and the tree {dt}. That {gap}-point gap is smaller than the
{spread}-point spread between test sets, so accuracy alone does not separate them clearly. The forest's clearer advantage is ranking
passengers by risk: ROC AUC {rf_auc} against {dt_auc}, better on {auc_wins} of 25 test sets. Choose the forest for the best
probabilities, and the tree when you need to explain every decision.</p></div>"""


# ---------------- Page ----------------
def main():
    st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")
    st.html(Path(__file__).with_name("style.css"))
    tree, forest = load_models()
    dt_r, rf_r, cmp_r = load_results()

    html(f"""
    <div class="hero">
      <div class="eyebrow">RMS TITANIC &middot; 15 APRIL 1912</div>
      <h1>Would you have survived?</h1>
      <p>Two machine learning models learned from the real passenger list. Give them a passenger and they will predict the outcome, then show how they got there.</p>
      <div class="hero-stats">
        <div><b>{dt_r["passengers"]}</b><span>passengers in the data</span></div>
        <div><b>{pct(dt_r["survived"] / dt_r["passengers"], 0)}</b><span>actually survived</span></div>
        <div><b>{pct(cmp_r["nested_cv"]["Accuracy"]["DT mean"])}</b><span>decision tree accuracy*</span></div>
        <div><b>{pct(cmp_r["nested_cv"]["Accuracy"]["RF mean"])}</b><span>random forest accuracy*</span></div>
      </div>
      <div class="waves"></div>
    </div>
    <p class="fine">*Mean across 25 test sets from repeated nested cross-validation.</p>""")

    t1, t2, t3 = st.tabs([":material/account_tree: Decision Tree", ":material/forest: Random Forest", ":material/balance: Comparison"])
    with t1:
        decision_tree_tab(tree, dt_r)
    with t2:
        random_forest_tab(forest, rf_r)
    with t3:
        comparison_tab(cmp_r)


main()
