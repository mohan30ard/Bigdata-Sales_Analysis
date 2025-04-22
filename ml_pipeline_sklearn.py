# ml_pipeline_noleak.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection    import train_test_split, RandomizedSearchCV
from sklearn.preprocessing      import OneHotEncoder
from sklearn.compose            import ColumnTransformer
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import roc_auc_score, roc_curve
from lightgbm                   import LGBMClassifier
from neo4j                      import GraphDatabase

def main():
    # ── 1) Load & basic feature engineering ───────────────────
    df = pd.read_csv("orders_clean.csv", parse_dates=['order_date','ship_date'])
    df["returned_flag"]   = (df.returned_count > 0).astype(int)
    df["ship_delay_days"] = (df.ship_date - df.order_date).dt.days
    df["unit_price"]      = df.sales / df.quantity

    # ── 2) Train/Test split BEFORE group features ────────────
    X = df[['order_id','customer_id','product_id','ship_mode',
            'customer_segment','region','category','sub_category',
            'sales','quantity','discount','profit',
            'ship_delay_days','unit_price']]
    y = df["returned_flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 3) Compute group stats on TRAIN only ──────────────────
    train_df = X_train.copy()
    train_df["returned_flag"] = y_train.values
    cust_stats = (
        train_df.groupby("customer_id")["returned_flag"]
        .agg(cust_ret_rate="mean", cust_order_cnt="count")
        .reset_index()
    )
    prod_stats = (
        train_df.groupby("product_id")["returned_flag"]
        .mean()
        .rename("prod_ret_rate")
        .reset_index()
    )

    def add_group_feats(df_):
        df_ = df_.merge(cust_stats, on="customer_id", how="left")
        df_ = df_.merge(prod_stats, on="product_id", how="left")
        df_[["cust_ret_rate","cust_order_cnt","prod_ret_rate"]] = \
            df_[["cust_ret_rate","cust_order_cnt","prod_ret_rate"]].fillna(0)
        return df_

    train_df = add_group_feats(train_df)
    test_df  = add_group_feats(X_test.copy())

    # ── 4) Prepare X/y with final features ────────────────────
    cat_cols = ["ship_mode","customer_segment","region",
                "category","sub_category"]
    num_cols = ["sales","quantity","discount","profit",
                "ship_delay_days","unit_price",
                "cust_ret_rate","cust_order_cnt","prod_ret_rate"]
    features = num_cols + cat_cols

    X_train_fe = train_df[features]
    y_train_fe = train_df["returned_flag"]
    X_test_fe  = test_df[features]
    y_test_fe  = y_test

    # ── 5) Build preprocessing & model pipeline ─────────────
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="passthrough")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf",  LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
    ])

    # ── 6) Hyperparameter tuning ─────────────────────────────
    param_dist = {
        "clf__n_estimators":     [100,200,500],
        "clf__max_depth":        [5,10,15,None],
        "clf__learning_rate":    [0.01,0.05,0.1],
        "clf__subsample":        [0.6,0.8,1.0],
        "clf__colsample_bytree": [0.6,0.8,1.0]
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20,
        scoring="roc_auc", cv=3, n_jobs=-1, random_state=42
    )
    search.fit(X_train_fe, y_train_fe)
    best = search.best_estimator_
    print("Best LGBM params:", search.best_params_)

    # ── 7) Evaluate on TEST ───────────────────────────────────
    y_proba = best.predict_proba(X_test_fe)[:,1]
    auc     = roc_auc_score(y_test_fe, y_proba)
    print(f"Test ROC AUC (no leak) = {auc:.3f}")

    # ── 8A) ROC curve plot ────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test_fe, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Return Prediction (no leak)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve_noleak.png")
    plt.show()

    # ── 8B) Feature importances ───────────────────────────────
    clf        = best.named_steps["clf"]
    feat_names = best.named_steps["prep"].get_feature_names_out()
    importances= clf.feature_importances_
    fi = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).nlargest(10, "importance")

    plt.figure(figsize=(8,4))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importances (no leak)")
    plt.tight_layout()
    plt.savefig("feature_importances_noleak.png")
    plt.show()

    # ── 9) Export TEST predictions ─────────────────────────────
    test_df["predicted_return"] = best.predict(X_test_fe)
    test_df["predicted_proba"]  = best.predict_proba(X_test_fe)[:,1]
    test_df[["order_id","predicted_return","predicted_proba"]] \
        .to_csv("order_return_predictions_noleak.csv", index=False)

    # ── 10) Write TEST preds back to Neo4j ─────────────────────
    driver = GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j","password")
    )
    with driver.session() as session:
        for row in test_df.itertuples():
            session.run(
                """
                MATCH (o:Order {id:$order_id})
                SET o.predicted_return = $pred,
                    o.predicted_prob   = $prob
                """,
                order_id=row.order_id,
                pred=bool(row.predicted_return),
                prob=float(row.predicted_proba)
            )
    driver.close()
    print("Done. ML pipeline with no data leakage completed.")

if __name__ == "__main__":
    main()
