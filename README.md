---
title: Titanic Survival Predictor
emoji: 🚢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: Decision tree vs random forest, with live predictions
---

# Would you have survived the Titanic?

An interactive Streamlit app for a decision tree and a random forest trained on the Titanic passenger data.
Enter a passenger's details and each model predicts whether they would have survived.

| Folder | Contents |
|---|---|
| `DT-Titanic/` | Decision tree script, charts, saved model, results |
| `RF-Titanic/` | Random forest script, charts, saved model, results |
| `Comparison-Titanic/` | Comparison script, charts, results |
| `Frontend/` | The Streamlit app (`app.py`, `style.css`) |

## Run locally

From this folder:

```
pip install -r requirements.txt
streamlit run Frontend/app.py
```

## Re-run the analysis

The app reads the saved models and `results.json` files, so it does not train anything when it starts.
To regenerate them, install `matplotlib` as well and run the three scripts. They need internet access to download the data, and the comparison takes about 10 minutes.

```
python DT-Titanic/set2_model3_decision_tree_titanic.py > DT-Titanic/run_log.txt
python RF-Titanic/set2_model4_random_forest_titanic.py > RF-Titanic/run_log.txt
python Comparison-Titanic/compare_dt_vs_rf_titanic.py > Comparison-Titanic/run_log.txt
```
