"""Phase 6 v2 – NILM disaggregation pipeline with expanded training set.

Key changes vs v1
-----------------
1. Regular 1-min grid: features retain NaN where rolling windows touch bad
   minutes; LightGBM handles them natively (no silent window dropping).
2. Expanded training: all 17 eligible houses (all eligible - val H5 - test H6)
   instead of just H2 + H3.
3. Zero-clamping: predictions below the optimised ON threshold are hard-zeroed
   to 0.0 W, eliminating idle-minute phantom energy accumulation.
4. Physical rule baseline (Baseline 3): aggregate > 1500 W above 60-min
   rolling min for >= 10 consecutive minutes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from src import config as C
from src.splits import get_model_splits, ALL_ELIGIBLE_CHANNELS, CHANNELS
from src.features import prepare_house_data
from src.models import (
    BaselineAlwaysOff,
    BaselineTimeOfDay,
    BaselinePhysicalRule,
    train_lightgbm,
    postprocess_predictions,
    zero_clamp,
    optimize_on_threshold,
)
from src.evaluate import compute_metrics, generate_s3_figures


def run_pipeline():
    print("=" * 60)
    print("STAGE: Section 3 v2 – NILM Disaggregation (expanded training)")
    print("=" * 60)
    t0 = time.time()

    # ── 1. Splits ────────────────────────────────────────────────────
    print("1. Generating model splits...")
    splits = get_model_splits()
    prim = splits["primary_split"]
    print(f"   Primary Split: Train={prim['train']} ({len(prim['train'])} houses), "
          f"Val={prim['val']}, Test={prim['test']}")

    # ── 2. Load all eligible houses ──────────────────────────────────
    print("\n2. Loading and preparing feature matrices for ALL eligible houses...")
    all_houses = sorted(ALL_ELIGIBLE_CHANNELS)
    house_data = {}
    for h in all_houses:
        ch = ALL_ELIGIBLE_CHANNELS[h]
        t_h = time.time()
        X, y, df_raw = prepare_house_data(h, ch)
        house_data[h] = (X, y, df_raw)
        pct_nan = X.isna().any(axis=1).mean() * 100
        print(f"   H{h:>2} ({ch}): {len(X):>7,} valid rows  |  "
              f"{pct_nan:.1f}% rows have >=1 NaN feat  ({time.time()-t_h:.1f}s)")

    # ── 3. Build train / val / test matrices ─────────────────────────
    train_houses = prim["train"]
    val_house    = prim["val"][0]
    test_house   = prim["test"][0]

    X_train = pd.concat([house_data[h][0] for h in train_houses])
    y_train = pd.concat([house_data[h][1] for h in train_houses])
    X_val, y_val, df_val_raw   = house_data[val_house]
    X_test, y_test, df_test_raw = house_data[test_house]

    on_pct_train = (y_train > C.ON_W).mean() * 100
    print(f"\n   Train matrix: {len(X_train):,} rows | ON%={on_pct_train:.2f}%")
    print(f"   Val   matrix: {len(X_val):,} rows   | H{val_house}")
    print(f"   Test  matrix: {len(X_test):,} rows   | H{test_house}")

    # ── 4. Train primary LightGBM ────────────────────────────────────
    print(f"\n3. Training LightGBM on {len(X_train):,} rows "
          f"(Train={train_houses}, Val=H{val_house})...")
    t_train = time.time()
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
    print(f"   Finished in {time.time()-t_train:.1f}s  "
          f"(Best iteration: {lgbm_model.best_iteration_})")

    # ── 5. Fit baselines ─────────────────────────────────────────────
    print("\n4. Fitting baselines...")
    b1 = BaselineAlwaysOff().fit(X_train, y_train)
    b2 = BaselineTimeOfDay().fit(X_train, y_train)
    b3 = BaselinePhysicalRule()

    # ── 6. Threshold optimisation on validation house ────────────────
    print("\n5. Optimising ON-state threshold on validation house...")
    val_raw   = lgbm_model.predict(X_val)
    val_preds = postprocess_predictions(val_raw, df_val_raw.loc[X_val.index, "aggregate"])
    best_thresh, best_val_f1 = optimize_on_threshold(y_val, val_preds)
    print(f"   Optimal threshold: {best_thresh:.1f} W  (Val F1={best_val_f1:.4f})")

    # ── 7. Test-house predictions ────────────────────────────────────
    print(f"\n6. Evaluating on primary test house H{test_house}...")
    agg_test = df_test_raw.loc[X_test.index, "aggregate"]

    # LightGBM: clip then zero-clamp
    lgbm_raw   = lgbm_model.predict(X_test)
    lgbm_preds = postprocess_predictions(lgbm_raw, agg_test)
    lgbm_preds = zero_clamp(lgbm_preds, best_thresh)

    b1_preds = b1.predict(X_test)
    b2_preds = b2.predict(X_test)

    # Rule baseline operates on the full aggregate series for the test period
    agg_full_test = df_test_raw["aggregate"].reindex(y_test.index)
    b3_preds = b3.predict(agg_full_test)

    s_y_true = y_test
    s_b1     = pd.Series(b1_preds, index=y_test.index)
    s_b2     = pd.Series(b2_preds, index=y_test.index)
    s_lgbm   = pd.Series(lgbm_preds, index=y_test.index)
    s_b3     = pd.Series(b3_preds,   index=y_test.index)

    m_b1   = compute_metrics(s_y_true, s_b1,   best_thresh, "Baseline 1 (Always Off)",     test_house)
    m_b2   = compute_metrics(s_y_true, s_b2,   best_thresh, "Baseline 2 (Time-of-Day)",    test_house)
    m_b3   = compute_metrics(s_y_true, s_b3,   best_thresh, "Baseline 3 (Physical Rule)",  test_house)
    m_lgbm = compute_metrics(s_y_true, s_lgbm, best_thresh, "LightGBM Disaggregator v2",   test_house)

    metrics_rows = [m_lgbm, m_b3, m_b2, m_b1]

    # Phantom-energy audit
    true_wh  = s_y_true.sum() / 60.0
    pred_wh  = s_lgbm.sum()   / 60.0
    idle_wh  = s_lgbm[s_y_true <= C.ON_W].sum() / 60.0
    print(f"   True WM energy  : {true_wh:,.1f} Wh")
    print(f"   Pred WM energy  : {pred_wh:,.1f} Wh")
    print(f"   Idle leakage    : {idle_wh:,.1f} Wh  (phantom energy on OFF minutes)")
    print(f"   SAE             : {m_lgbm['sae']:.4f}")

    # ── 8. Save predictions ──────────────────────────────────────────
    test_pred_df = pd.DataFrame({
        "timestamp":    y_test.index,
        "aggregate_w":  df_test_raw.loc[y_test.index, "aggregate"].values,
        "true_wm_w":    y_test.values,
        "pred_lgbm_w":  lgbm_preds,
        "pred_rule_w":  b3_preds,
        "pred_b2_w":    b2_preds,
        "pred_b1_w":    b1_preds,
    }).set_index("timestamp")

    pred_path = C.OUT_DIR / "test_predictions.parquet"
    test_pred_df.to_parquet(pred_path, compression="snappy")
    print(f"\n   Saved {pred_path} ({len(test_pred_df):,} rows)")

    # ── 9. Section 3 figures ─────────────────────────────────────────
    print("\n7. Generating Section 3 figures (v2)...")
    generate_s3_figures(
        test_house=test_house,
        df_raw=df_test_raw,
        y_true=s_y_true,
        y_pred_lgbm=s_lgbm,
        y_pred_rule=s_b3,
        y_pred_b1=s_b1,
        y_pred_b2=s_b2,
        on_threshold=best_thresh,
    )

    # ── 10. 4-Fold LOHO-CV ───────────────────────────────────────────
    print("\n8. Executing 4-Fold LOHO-CV (expanded training per fold)...")
    loho_folds   = splits["loho_folds"]
    loho_results = []

    for fold_name, fold_cfg in loho_folds.items():
        f_train = fold_cfg["train"]
        f_val   = fold_cfg["val"][0]
        f_test  = fold_cfg["test"][0]

        # Load any house not already in house_data
        for h in list(f_train) + [f_val, f_test]:
            if h not in house_data:
                ch = ALL_ELIGIBLE_CHANNELS[h]
                house_data[h] = prepare_house_data(h, ch)

        X_tr_f = pd.concat([house_data[h][0] for h in f_train])
        y_tr_f = pd.concat([house_data[h][1] for h in f_train])
        X_va_f, y_va_f, df_va_f = house_data[f_val]
        X_te_f, y_te_f, df_te_f = house_data[f_test]

        m_f = train_lightgbm(X_tr_f, y_tr_f, X_va_f, y_va_f)

        v_raw   = m_f.predict(X_va_f)
        v_preds = postprocess_predictions(v_raw, df_va_f.loc[X_va_f.index, "aggregate"])
        th_f, _ = optimize_on_threshold(y_va_f, v_preds)

        te_raw   = m_f.predict(X_te_f)
        te_preds = postprocess_predictions(te_raw, df_te_f.loc[X_te_f.index, "aggregate"])
        te_preds = zero_clamp(te_preds, th_f)
        s_te_p   = pd.Series(te_preds, index=y_te_f.index)

        res = compute_metrics(
            y_te_f, s_te_p, th_f,
            model_name=f"LOHO LightGBM v2 (Fold {f_test})",
            house=f_test,
        )
        loho_results.append(res)
        metrics_rows.append(res)
        print(f"   Fold H{f_test}: MAE={res['mae_all_w']}W  F1={res['f1']}  "
              f"SAE={res['sae']}  CycleRecall={res['cycle_recall']}")

    # Pooled average
    loho_df = pd.DataFrame(loho_results)
    num_cols = ["on_threshold_w","mae_all_w","rmse_w","mae_on_w","mae_off_w",
                "sae","precision","recall","f1","cycle_recall","cycle_precision","cycle_energy_mape"]
    pooled_row = {c: round(float(loho_df[c].mean()), 4) for c in num_cols}
    pooled_row.update({
        "house":          "Pooled (All 4)",
        "model":          "LOHO LightGBM v2 (Mean)",
        "n_true_cycles":  int(loho_df["n_true_cycles"].sum()),
        "n_pred_cycles":  int(loho_df["n_pred_cycles"].sum()),
    })
    metrics_rows.append(pooled_row)

    # ── 11. Save metrics ─────────────────────────────────────────────
    metrics_df   = pd.DataFrame(metrics_rows)
    metrics_path = C.OUT_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n   Saved {metrics_path}")

    print("\n--- MODEL METRICS SUMMARY ---")
    print(metrics_df.to_string(index=False))

    print(f"\nSection 3 v2 pipeline completed in {time.time()-t0:.1f}s")
    print("=" * 60)
    return metrics_df


if __name__ == "__main__":
    run_pipeline()
