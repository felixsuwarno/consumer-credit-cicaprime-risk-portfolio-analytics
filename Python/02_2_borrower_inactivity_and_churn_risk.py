import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pandas display settings
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 2000)


# Load data
# -----------------------------------------------------------

script_dir      = os.path.dirname(os.path.abspath(__file__))
data_dir        = os.path.join(script_dir, "..", "Data_Generated")
path_file       = os.path.normpath(os.path.join(data_dir, "02_2_borrower_inactivity_and_churn_risk.csv"))

df_customer     = pd.read_csv(path_file)


# Keep only observable customers (inactive_flag not null)
# -----------------------------------------------------------

df_customer_observable                      = df_customer.loc[df_customer["inactive_flag"].notna()].copy()


# Ensure inactive_flag is numeric
# -----------------------------------------------------------

df_customer_observable["inactive_flag"]     = pd.to_numeric(df_customer_observable["inactive_flag"],errors="coerce")


# -----------------------------------------------------------
# Compute overall inactivity (churn) rate
# -----------------------------------------------------------

n_customers_observable                      = len(df_customer_observable)
n_inactive_customers                        = (df_customer_observable["inactive_flag"] == 1).sum()

inactivity_rate                             = n_inactive_customers / n_customers_observable

print("Observable customers :", n_customers_observable)
print("Inactive customers   :", n_inactive_customers)
print("Inactivity rate      :", round(inactivity_rate, 4))



# Inactivity by segment (counts + rate)
# -----------------------------------------------------------

list_segment_cols = [
    "acquisition_channel",
    "risk_tier_at_signup",
    "income_band",
    "age_band",
    "region"
]

dict_segment_tables = {}

for col_segment in list_segment_cols:
    df_segment =    (
                        df_customer_observable
                            .groupby(col_segment, dropna = False)
                            .agg(
                                    n_customers     =("customer_id", "count"),
                                    inactivity_rate =("inactive_flag", "mean")
                                )
                            .reset_index()
                    )

    df_segment["inactivity_rate_pct"]   = df_segment["inactivity_rate"] * 100

    df_segment                          = df_segment.sort_values(by=["inactivity_rate", "n_customers"],ascending=[False, False])

    dict_segment_tables[col_segment]    = df_segment

    print("\n--- Inactivity by", col_segment, "---")
    print(df_segment)


# Rank each segment table by risk (high inactivity first)
# -----------------------------------------------------------

for col_segment, df_segment in dict_segment_tables.items():
    df_ranked   = df_segment.sort_values    (
                                                by          = ["inactivity_rate", "n_customers"],
                                                ascending   = [False, False]

                                            )   .reset_index(drop=True)

    dict_segment_tables[col_segment] = df_ranked

    print("\n--- Ranked inactivity by", col_segment, "---")
    print(df_ranked)

# -----------------------------------------------------------
# Bar charts: inactivity rate (%) by each segment column
# -----------------------------------------------------------

list_segment_cols = [
    "acquisition_channel",
    "risk_tier_at_signup",
    "income_band",
    "age_band",
    "region"
]

for col_segment in list_segment_cols:
    df_plot = dict_segment_tables[col_segment].copy()

    df_plot["segment_label"] = df_plot[col_segment].astype(str)
    df_plot["inactivity_rate_pct"] = df_plot["inactivity_rate"] * 100

    plt.figure(figsize=(12, 6))

    plt.bar(
        df_plot["segment_label"],
        df_plot["inactivity_rate_pct"]
    )

    plt.title(
        f"Inactivity rate (180-day) by {col_segment}",
        fontsize=15   # ~25% larger than default
    )
    plt.ylabel(
        "Inactivity rate (%)",
        fontsize=13
    )
    plt.xlabel(
        col_segment,
        fontsize=13
    )

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=11
    )
    plt.yticks(
        fontsize=11
    )

    plt.tight_layout()
    plt.show()

