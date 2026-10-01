from numpy import*
from pandas import*
from sys import*
from os import*



def func_rank_01(vals: Series) -> Series: 
  return vals.rank(method="average", pct=True)


def robust_group_amount_score(df: DataFrame, key: str) -> Series:
    log_at = log1p(df["amount"].clip(lower=0))
    gpoups = log_at.groupby(df[key])
    med_values = gpoups.transform("median")
    metric_q75 = gpoups.transform(lambda s: s.quantile(0.75))
    metric_q25 = gpoups.transform(lambda s: s.quantile(0.25))
    iqr = (metric_q75 - metric_q25).replace(0, nan).fillna((log_at.quantile(0.75) - log_at.quantile(0.25)))
    return ((log_at - med_values) / (iqr + 10**-6)).clip(lower=0, upper=8) / 8


def main() -> None:
    d2f = read_csv("transactions.csv")
    tx_time = to_datetime(d2f["timestamp"])
    issue_date = to_datetime(d2f["card_issue_date"])
    log_at = log1p(d2f["amount"].clip(lower=0))
    amount_global = func_rank_01(log_at)
    amount_user = robust_group_amount_score(d2f, "user_id")
    amount_card = robust_group_amount_score(d2f, "card_id")
    user_country = d2f.groupby("user_id")["country"].transform(lambda s: s.value_counts().idxmax())
    card_country = d2f.groupby("card_id")["country"].transform(lambda s: s.value_counts().idxmax())
    country_shift = (d2f["country"].ne(user_country) | d2f["country"].ne(card_country)).astype(float)
    merchant_freq = d2f["merchant_id"].map(d2f["merchant_id"].value_counts())
    rare_merchant = func_rank_01(-merchant_freq)
    balance = func_rank_01((-d2f["account_balance_after"]).clip(lower=0))
    negative_amount = d2f["amount"].lt(0).astype(float)
    issue_after_transaction = issue_date.gt(tx_time).astype(float)
    unknown_device = d2f["device_type"].eq("unknown").astype(float)
    prepaid_card = d2f["card_type"].eq("prepaid").astype(float)
    high_amount_foreign = amount_global * country_shift
    
    raw_score = (
        (2.80 * amount_global)
        + (2.20 * amount_user)
        + (1.80 * amount_card)
        + (2.60 * country_shift)
        + (2.40 * high_amount_foreign)
        + (1.70 * negative_amount)
        + (0.65 * unknown_device)
        + (0.50 * issue_after_transaction)
        + (0.35 * prepaid_card)
        + (0.35 * rare_merchant)
        + (0.30 * balance))
    
    zzz = (raw_score - raw_score.mean()) / (raw_score.std(ddof=0) + 1e-9)
    predict = 1 / (1 + exp(-zzz))
    predict = clip(predict, 10**-6, 1-10**-6)
    submission = DataFrame({"transaction_id": d2f["transaction_id"], "prediction": predict})
    submission.to_csv("submission.csv", index=False)



if __name__ == "__main__":
    main()
else:
    exit()

