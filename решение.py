from __future__ import annotations
from argparse import*
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


RANDOM_STATE = 20261009


LOW_CARD_COLS = [
    "currency",
    "transaction_type",
    "merchant_category",
    "device_type",
    "country",
    "city",
    "is_foreign",
    "card_type",
]
HIGH_CARD_COLS = ["merchant_id", "card_id", "user_id"]


def rank01(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    return s.rank(method="average", pct=True).to_numpy(dtype=float)


def robust_group_deviation(df: pd.DataFrame, group_col: str, value_col: str, min_count: int = 8) -> pd.Series:
    grouped = df.groupby(group_col)[value_col]
    med = grouped.transform("median")
    q75 = grouped.transform(lambda s: s.quantile(0.75))
    q25 = grouped.transform(lambda s: s.quantile(0.25))
    count = grouped.transform("size")
    global_med = df[value_col].median()
    global_iqr = df[value_col].quantile(0.75) - df[value_col].quantile(0.25)
    global_iqr = max(float(global_iqr), 1e-6)
    med = med.where(count >= min_count, global_med)
    iqr = (q75 - q25).where(count >= min_count, global_iqr)
    iqr = iqr.clip(lower=global_iqr * 0.05)
    return ((df[value_col] - med).abs() / iqr).astype(float)


def add_frequency_features(features: pd.DataFrame, df: pd.DataFrame, cols: list[str]) -> None:
    n = len(df)
    for col in cols:
        counts = df[col].map(df[col].value_counts()).astype(float)
        features[f"{col}_freq"] = counts / n
        features[f"{col}_rarity"] = -np.log((counts + 1.0) / (n + 1.0))


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["user_registration_date"] = pd.to_datetime(df["user_registration_date"])
    df["card_issue_date"] = pd.to_datetime(df["card_issue_date"])
    amount_abs = df["amount"].abs()
    balance_before_abs = df["account_balance_before"].abs().clip(lower=1.0)
    balance_after_abs = df["account_balance_after"].abs()
    f = pd.DataFrame(index=df.index)
    f["amount"] = df["amount"]
    f["amount_abs"] = amount_abs
    f["log_amount_abs"] = np.log1p(amount_abs)
    f["amount_negative"] = (df["amount"] < 0).astype(int)
    f["balance_before"] = df["account_balance_before"]
    f["balance_after"] = df["account_balance_after"]
    f["balance_after_negative"] = (df["account_balance_after"] < 0).astype(int)
    f["log_balance_before"] = np.log1p(balance_before_abs)
    f["log_balance_after_abs"] = np.log1p(balance_after_abs)
    f["amount_to_balance"] = amount_abs / balance_before_abs
    f["balance_change_check"] = (df["account_balance_before"] - df["account_balance_after"] - df["amount"]).abs()
    f["hour"] = df["timestamp"].dt.hour
    f["dayofweek"] = df["timestamp"].dt.dayofweek
    f["day"] = df["timestamp"].dt.day
    f["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    f["is_night"] = df["timestamp"].dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    f["account_age_days"] = (df["timestamp"] - df["user_registration_date"]).dt.days
    f["card_age_days"] = (df["timestamp"] - df["card_issue_date"]).dt.days
    f["card_issued_after_tx"] = (f["card_age_days"] < 0).astype(int)
    f["card_age_abs"] = f["card_age_days"].abs()

    for col in LOW_CARD_COLS + HIGH_CARD_COLS:
        codes, _ = pd.factorize(df[col], sort=True)
        f[f"{col}_code"] = codes.astype(float)

    add_frequency_features(f, df, LOW_CARD_COLS + HIGH_CARD_COLS)

    for group in ["user_id", "card_id", "merchant_id", "merchant_category", "country", "city"]:
        f[f"log_amount_dev_by_{group}"] = robust_group_deviation(pd.concat([df[[group]], f[["log_amount_abs"]]], axis=1), group, "log_amount_abs", min_count=8 if group != "merchant_id" else 4)

    for group in ["user_id", "card_id"]:
        counts = df.groupby(group)["transaction_id"].transform("size").astype(float)
        for col in ["country", "city", "currency", "device_type", "merchant_category"]:
            pair_counts = df.groupby([group, col])["transaction_id"].transform("size").astype(float)
            f[f"{group}_{col}_share"] = pair_counts / counts
            f[f"{group}_{col}_rare"] = 1.0 - (pair_counts / counts)

    city_country = df.groupby(["country", "city"])["transaction_id"].transform("size").astype(float)
    country_count = df.groupby("country")["transaction_id"].transform("size").astype(float)
    f["city_given_country_share"] = city_country / country_count

    merchant_country = df.groupby(["merchant_category", "country"])["transaction_id"].transform("size").astype(float)
    merchant_count = df.groupby("merchant_category")["transaction_id"].transform("size").astype(float)
    f["country_given_mcc_share"] = merchant_country / merchant_count

    one_hot = pd.get_dummies(df[LOW_CARD_COLS].astype(str), prefix=LOW_CARD_COLS, dtype=float)
    f = pd.concat([f, one_hot], axis=1)
    return f.replace([np.inf, -np.inf], np.nan)


def synthetic_background(features: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    synth = pd.DataFrame({col: rng.choice(features[col].to_numpy(), size=n_samples, replace=True) for col in features.columns})
    numeric_colomnss = [
        "amount",
        "amount_abs",
        "log_amount_abs",
        "amount_to_balance",
        "balance_before",
        "balance_after",
        "account_age_days",
        "card_age_days",
    ]
    tail_n = n_samples // 3
    tail_idx = rng.choice(n_samples, size=tail_n, replace=False)
    for col in numeric_colomnss:
        if col not in synth:
            continue
        q_low, q_high = features[col].quantile([0.005, 0.995])
        spread = max(float(q_high - q_low), 1e-6)
        extreme = rng.choice([-1.0, 1.0], size=tail_n) * rng.gamma(2.0, spread / 3.0, size=tail_n)
        center = rng.choice([q_low, q_high], size=tail_n)
        synth.loc[tail_idx, col] = center + extreme
    return synth


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Build fraud probability predictions for transactions.csv.")
    parser.add_argument(
        "--input",
        default="transactions.csv",
        help="Path to the input transactions.csv file.",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Path where the output CSV will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.input)
    out_path = Path(args.output)
    df = pd.read_csv(data_path)
    features = build_features(df)
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(features),
        columns=features.columns,
        index=features.index,
    )
    scaled = RobustScaler(quantile_range=(5, 95)).fit_transform(imputed)
    iso_models = [
        IsolationForest(
            n_estimators=700,
            max_samples=min(12000, len(df)),
            contamination=c,
            max_features=mf,
            bootstrap=False,
            random_state=RANDOM_STATE + i,
            n_jobs=-1,
        )
        for i, (c, mf) in enumerate([(0.015, 0.7), (0.03, 0.9), (0.06, 1.0)])
    ]
    iso_scores = []
    for model in iso_models:
        model.fit(scaled)
        iso_scores.append(rank01(-model.score_samples(scaled)))
    iso_score = np.mean(iso_scores, axis=0)
    pca_scaler = RobustScaler(quantile_range=(5, 95))
    pca_input = pca_scaler.fit_transform(imputed)
    pca = PCA(n_components=0.93, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(pca_input)
    reconstructed = pca.inverse_transform(reduced)
    pca_error = ((pca_input - reconstructed) ** 2).mean(axis=1)
    pca_score = rank01(pca_error)
    rng = np.random.default_rng(RANDOM_STATE)
    synth = synthetic_background(imputed, len(imputed) * 2, rng)
    x_clf = pd.concat([imputed, synth], ignore_index=True)
    y_clf = np.r_[np.zeros(len(imputed), dtype=int), np.ones(len(synth), dtype=int)]
    clf = ExtraTreesClassifier(
        n_estimators=900,
        max_features="sqrt",
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(x_clf, y_clf)
    density_score = rank01(clf.predict_proba(imputed)[:, 1])
    heuristic = np.zeros(len(df), dtype=float)
    for col, weight in [
        ("log_amount_abs", 1.2),
        ("amount_to_balance", 1.0),
        ("amount_negative", 0.7),
        ("balance_after_negative", 0.6),
        ("is_foreign_code", 0.5),
        ("device_type_rarity", 0.35),
        ("currency_rarity", 0.25),
        ("merchant_category_rarity", 0.2),
        ("card_issued_after_tx", 0.4),
        ("user_id_country_rare", 0.5),
        ("user_id_city_rare", 0.4),
        ("user_id_currency_rare", 0.4),
        ("user_id_device_type_rare", 0.25),
        ("card_id_country_rare", 0.5),
        ("card_id_currency_rare", 0.4),
    ]:
        if col in imputed:
            heuristic += weight * rank01(imputed[col].to_numpy())
    for col in imputed.columns:
        if col.startswith("log_amount_dev_by_"):
            heuristic += 0.45 * rank01(imputed[col].to_numpy())
    heuristic_score = rank01(heuristic)

    blended = (
        0.38 * iso_score
        + 0.27 * density_score
        + 0.20 * heuristic_score
        + 0.15 * pca_score
    )
    r = rank01(blended)
    prediction = 1.0 / (1.0 + np.exp(-8.0 * (r - 0.88)))
    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
    submission = pd.DataFrame(
        {
            "transaction_id": df["transaction_id"].astype(int),
            "prediction": prediction,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(submission.shape)
    print(submission["prediction"].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]).to_string())



if __name__ == "__main__":
    main()
else:
    exit()
