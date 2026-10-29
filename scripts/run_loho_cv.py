"""Run only the 4-fold LOHO-CV and append results to model_metrics.csv."""
import sys, matplotlib
matplotlib.use("Agg")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import time

from src import config as C
from src.splits import get_model_splits, ALL_ELIGIBLE_CHANNELS
from src.features import prepare_house_data
from src.models import train_lightgbm, postprocess_predictions, zero_clamp, optimize_on_threshold
from src.evaluate import compute_metrics

print("Loading existing primary metrics...")
existing = pd.read_csv(C.OUT_DIR / "model_metrics.csv")
print(existing[["house","model","sae","f1","mae_all_w"]].to_string(index=False))

splits     = get_model_splits()
loho_folds = splits["loho_folds"]

print("\nLoading house data for all eligible houses...")
t0 = time.time()
house_data = {}
for h, ch in ALL_ELIGIBLE_CHANNELS.items():
    X, y, df_raw = prepare_house_data(h, ch)
    house_data[h] = (X, y, df_raw)
    print(f"  H{h}: {len(X):,} rows")

loho_results = []
print("\nRunning 4-Fold LOHO-CV...")
for fold_name, fold_cfg in loho_folds.items():
    f_train = fold_cfg["train"]
    f_val   = fold_cfg["val"][0]
    f_test  = fold_cfg["test"][0]

    X_tr = pd.concat([house_data[h][0] for h in f_train])
    y_tr = pd.concat([house_data[h][1] for h in f_train])
    X_va, y_va, df_va = house_data[f_val]
    X_te, y_te, df_te = house_data[f_test]

    model = train_lightgbm(X_tr, y_tr, X_va, y_va)

    v_raw   = model.predict(X_va)
    v_preds = postprocess_predictions(v_raw, df_va.loc[X_va.index, "aggregate"])
    th, _   = optimize_on_threshold(y_va, v_preds)

    te_raw   = model.predict(X_te)
    te_preds = postprocess_predictions(te_raw, df_te.loc[X_te.index, "aggregate"])
    te_preds = zero_clamp(te_preds, th)
    s_te     = pd.Series(te_preds, index=y_te.index)

    res = compute_metrics(y_te, s_te, th,
                          model_name=f"LOHO LightGBM v2 (Fold {f_test})",
                          house=f_test)
    loho_results.append(res)
    print(f"  Fold H{f_test}: best_iter={model.best_iteration_}  "
          f"thresh={th:.0f}W  MAE={res['mae_all_w']}  F1={res['f1']}  SAE={res['sae']}")

loho_df = pd.DataFrame(loho_results)
num_cols = ["on_threshold_w","mae_all_w","rmse_w","mae_on_w","mae_off_w",
            "sae","precision","recall","f1","cycle_recall","cycle_precision","cycle_energy_mape"]
pooled = {c: round(float(loho_df[c].mean()), 4) for c in num_cols}
pooled.update({
    "house": "Pooled (All 4)",
    "model": "LOHO LightGBM v2 (Mean)",
    "n_true_cycles": int(loho_df["n_true_cycles"].sum()),
    "n_pred_cycles": int(loho_df["n_pred_cycles"].sum()),
})
loho_results.append(pooled)

# Rebuild full metrics file: keep primary rows, replace any old LOHO rows
primary_rows = existing[~existing["model"].str.contains("LOHO", na=False)]
new_rows     = pd.DataFrame(loho_results)
metrics_df   = pd.concat([primary_rows, new_rows], ignore_index=True)
metrics_path = C.OUT_DIR / "model_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print(f"\nSaved {metrics_path}")
print("\n--- FULL METRICS ---")
print(metrics_df[["house","model","sae","f1","mae_all_w","cycle_recall"]].to_string(index=False))
print(f"\nCompleted in {time.time()-t0:.1f}s")
