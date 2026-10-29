import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd

from src import config as C
from src.splits import get_model_splits, CHANNELS
from src.features import prepare_house_data
from src.models import (
    BaselineAlwaysOff,
    BaselineTimeOfDay,
    train_lightgbm,
    postprocess_predictions,
    optimize_on_threshold,
)
from src.evaluate import compute_metrics, generate_s3_figures


def run_pipeline():
    print("=" * 60)
    print("STAGE: Section 3 - NILM Disaggregation Modeling & Evaluation")
    print("=" * 60)
    t0 = time.time()

    # 1. Splits
    print("1. Generating model splits...")
    splits = get_model_splits()
    prim = splits["primary_split"]
    print(f"Primary Split: Train={prim['train']}, Val={prim['val']}, Test={prim['test']}")

    # 2. Load and prepare data for all 4 cohort houses
    print("\n2. Loading and preparing feature matrices...")
    house_data = {}
    for h in [2, 3, 5, 6]:
        t_h = time.time()
        ch = CHANNELS[h]
        X, y, df_raw = prepare_house_data(h, ch)
        house_data[h] = (X, y, df_raw)
        print(f"  House {h} ({ch}): {len(X):,} valid minutes ({time.time() - t_h:.1f}s)")

    # 3. Train Primary Model
    train_houses = prim["train"]
    val_house = prim["val"][0]
    test_house = prim["test"][0]

    X_train = pd.concat([house_data[h][0] for h in train_houses])
    y_train = pd.concat([house_data[h][1] for h in train_houses])

    X_val, y_val, df_val_raw = house_data[val_house]
    X_test, y_test, df_test_raw = house_data[test_house]

    print(f"\n3. Training LightGBM on {len(X_train):,} rows (Train={train_houses}, Val={val_house})...")
    t_train = time.time()
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
    print(f"  Training finished in {time.time() - t_train:.1f}s (Best iteration: {lgbm_model.best_iteration_})")

    # Fit baselines
    print("Fitting baselines...")
    b1 = BaselineAlwaysOff().fit(X_train, y_train)
    b2 = BaselineTimeOfDay().fit(X_train, y_train)

    # 4. Threshold optimization on validation house
    print("\n4. Optimizing ON-state decision threshold on validation house...")
    val_raw_preds = lgbm_model.predict(X_val)
    val_preds = postprocess_predictions(val_raw_preds, df_val_raw.loc[X_val.index, "aggregate"])
    best_thresh, best_val_f1 = optimize_on_threshold(y_val, val_preds)
    print(f"  Optimal ON Threshold: {best_thresh:.1f} W (Validation F1 = {best_val_f1:.4f})")

    # 5. Evaluate on Primary Test House
    print(f"\n5. Evaluating on Primary Test House {test_house}...")
    b1_preds = b1.predict(X_test)
    b2_preds = b2.predict(X_test)
    test_raw_preds = lgbm_model.predict(X_test)
    test_preds = postprocess_predictions(test_raw_preds, df_test_raw.loc[X_test.index, "aggregate"])

    s_y_true = y_test
    s_b1 = pd.Series(b1_preds, index=y_test.index)
    s_b2 = pd.Series(b2_preds, index=y_test.index)
    s_lgbm = pd.Series(test_preds, index=y_test.index)

    m_b1 = compute_metrics(s_y_true, s_b1, on_threshold=best_thresh, model_name="Baseline 1 (Always Off)", house=test_house)
    m_b2 = compute_metrics(s_y_true, s_b2, on_threshold=best_thresh, model_name="Baseline 2 (Time-of-Day)", house=test_house)
    m_lgbm = compute_metrics(s_y_true, s_lgbm, on_threshold=best_thresh, model_name="LightGBM Disaggregator", house=test_house)

    metrics_rows = [m_lgbm, m_b2, m_b1]

    # Save test predictions
    test_pred_df = pd.DataFrame({
        "timestamp": y_test.index,
        "aggregate_w": df_test_raw.loc[y_test.index, "aggregate"].values,
        "true_wm_w": y_test.values,
        "pred_lgbm_w": test_preds,
        "pred_b2_w": b2_preds,
        "pred_b1_w": b1_preds,
    }).set_index("timestamp")

    pred_out_path = C.OUT_DIR / "test_predictions.parquet"
    test_pred_df.to_parquet(pred_out_path, compression="snappy")
    print(f"Saved {pred_out_path} ({len(test_pred_df):,} rows)")

    # 6. Generate Section 3 Figures
    print("\n6. Generating Section 3 figures...")
    generate_s3_figures(
        test_house=test_house,
        df_raw=df_test_raw,
        y_true=s_y_true,
        y_pred_lgbm=s_lgbm,
        y_pred_b1=s_b1,
        y_pred_b2=s_b2,
        on_threshold=best_thresh,
    )

    # 7. 4-Fold Leave-One-House-Out Cross-Validation
    print("\n7. Executing 4-Fold Leave-One-House-Out Cross-Validation across cohort...")
    loho_folds = splits["loho_folds"]
    loho_results = []

    for fold_name, fold_cfg in loho_folds.items():
        f_train = fold_cfg["train"]
        f_val = fold_cfg["val"][0]
        f_test = fold_cfg["test"][0]

        X_tr_f = pd.concat([house_data[h][0] for h in f_train])
        y_tr_f = pd.concat([house_data[h][1] for h in f_train])
        X_va_f, y_va_f, df_va_f = house_data[f_val]
        X_te_f, y_te_f, df_te_f = house_data[f_test]

        m_f = train_lightgbm(X_tr_f, y_tr_f, X_va_f, y_va_f)
        v_preds = postprocess_predictions(m_f.predict(X_va_f), df_va_f.loc[X_va_f.index, "aggregate"])
        th_f, _ = optimize_on_threshold(y_va_f, v_preds)

        te_preds = postprocess_predictions(m_f.predict(X_te_f), df_te_f.loc[X_te_f.index, "aggregate"])
        s_te_p = pd.Series(te_preds, index=y_te_f.index)

        res = compute_metrics(y_te_f, s_te_p, on_threshold=th_f, model_name=f"LOHO LightGBM (Fold {f_test})", house=f_test)
        loho_results.append(res)
        metrics_rows.append(res)
        print(f"  Fold H{f_test}: MAE={res['mae_all_w']}W, RMSE={res['rmse_w']}W, F1={res['f1']}, SAE={res['sae']}, Cycle Recall={res['cycle_recall']}")

    # Add Pooled LOHO average row
    loho_df = pd.DataFrame(loho_results)
    pooled_row = {
        "house": "Pooled (All 4)",
        "model": "LOHO LightGBM (Mean)",
        "on_threshold_w": round(float(loho_df["on_threshold_w"].mean()), 1),
        "mae_all_w": round(float(loho_df["mae_all_w"].mean()), 2),
        "rmse_w": round(float(loho_df["rmse_w"].mean()), 2),
        "mae_on_w": round(float(loho_df["mae_on_w"].mean()), 2),
        "mae_off_w": round(float(loho_df["mae_off_w"].mean()), 2),
        "sae": round(float(loho_df["sae"].mean()), 4),
        "precision": round(float(loho_df["precision"].mean()), 4),
        "recall": round(float(loho_df["recall"].mean()), 4),
        "f1": round(float(loho_df["f1"].mean()), 4),
        "n_true_cycles": int(loho_df["n_true_cycles"].sum()),
        "n_pred_cycles": int(loho_df["n_pred_cycles"].sum()),
        "cycle_recall": round(float(loho_df["cycle_recall"].mean()), 4),
        "cycle_precision": round(float(loho_df["cycle_precision"].mean()), 4),
        "cycle_energy_mape": round(float(loho_df["cycle_energy_mape"].mean()), 2),
    }
    metrics_rows.append(pooled_row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = C.OUT_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")

    print("\n--- MODEL METRICS SUMMARY ---")
    print(metrics_df.to_string(index=False))

    print(f"\nSection 3 pipeline completed in {time.time() - t0:.1f}s")
    print("=" * 60)
    return metrics_df


if __name__ == "__main__":
    run_pipeline()
