import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Pandas display settings
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 2000)

# find the path to current active Python path, ground it here
script_dir                  = os.path.dirname(os.path.abspath(__file__))

data_dir                    = os.path.join(script_dir, "..", "Data_Generated")

# Attach the file name into the path
pd_path                     = os.path.join(data_dir, "03_3_loss_given_default.csv")
df_lgd                      = pd.read_csv(pd_path)

# Ensure date columns are parsed correctly
df_lgd["origination_month"] = pd.to_datetime(df_lgd["origination_month"])

# ----------------
# LGD by Risk Tier
# ----------------

df_lgd_by_tier = (df_lgd
                        .groupby("risk_tier_at_signup", as_index=False)
                        .agg(
                                {
                                    "principal_loss"                : "sum",
                                    "principal_unpaid_on_default"   : "sum",
                                    "loan_id"                       : "count"
                                }
                            )
                )

df_lgd_by_tier              = df_lgd_by_tier.rename(columns={"loan_id": "defaulted_loan_count"})

df_lgd_by_tier["lgd_rate"]  = (df_lgd_by_tier["principal_loss"] / df_lgd_by_tier["principal_unpaid_on_default"]).round(4)

df_lgd_by_tier              = df_lgd_by_tier.sort_values("risk_tier_at_signup")



# ----------------
# LGD by Vintage
# ----------------

df_lgd_by_vintage = (df_lgd
                            .groupby("origination_month", as_index=False)
                            .agg(
                                    {
                                        "principal_loss"                : "sum",
                                        "principal_unpaid_on_default"   : "sum",
                                        "loan_id"                       : "count"
                                    }
                                )
                    )

df_lgd_by_vintage              = df_lgd_by_vintage.rename(columns={"loan_id": "defaulted_loan_count"})

df_lgd_by_vintage["lgd_rate"]  = (df_lgd_by_vintage["principal_loss"] / df_lgd_by_vintage["principal_unpaid_on_default"]).round(4)

df_lgd_by_vintage              = df_lgd_by_vintage.sort_values("origination_month")

# Pretty month labels like 2023-01
df_lgd_by_vintage["origination_month_label"] = df_lgd_by_vintage["origination_month"].dt.strftime("%Y-%m")



# ----------------
# Chart 1: LGD by Risk Tier
# ----------------

plt.figure(figsize=(18, 10))

x_tier                      = df_lgd_by_tier["risk_tier_at_signup"]
y_lgd_tier                  = df_lgd_by_tier["lgd_rate"]  # ratio (0–1)

bars                        = plt.bar(x_tier, y_lgd_tier)

plt.title("Exposure-Weighted LGD by Risk Tier at Signup", fontsize=22)
plt.xlabel("Risk Tier", fontsize=18)
plt.ylabel("LGD Rate (%)", fontsize=18)

ax                          = plt.gca()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Legend-style note box
plt.text(
        0.02, 0.96,
        "Numbers above bars = Defaulted loan count (N)",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black")
    )

# N labels above bars
for bar, n in zip(bars, df_lgd_by_tier["defaulted_loan_count"]):
    bar_height              = bar.get_height()
    plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar_height + 0.01,
            f"{int(n)}",
            ha="center",
            va="bottom",
            fontsize=16
        )

plt.tight_layout()
plt.show()



# ----------------
# Chart: LGD by Origination Month
# ----------------

plt.figure(figsize=(22, 10))

x_vintage                   = df_lgd_by_vintage["origination_month_label"]
y_lgd_vintage               = df_lgd_by_vintage["lgd_rate"]

bars                        = plt.bar(x_vintage, y_lgd_vintage)

plt.title("Exposure-Weighted LGD by Origination Month", fontsize=22)
plt.xlabel("Origination Month", fontsize=18)
plt.ylabel("LGD Rate (%)", fontsize=18)

ax                          = plt.gca()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# Add headroom so labels and legend don't collide
plt.ylim(0, y_lgd_vintage.max() * 1.20)

plt.xticks(rotation=45, ha="right", fontsize=14)
plt.yticks(fontsize=16)

# Legend-style note box
plt.text(
        0.70, 0.98,
        "Numbers above bars = Defaulted loan count (N)",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black")
    )

# N labels above bars
for bar, n in zip(bars, df_lgd_by_vintage["defaulted_loan_count"]):
    bar_height              = bar.get_height()
    plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar_height + 0.015,
            f"{int(n)}",
            ha="center",
            va="bottom",
            fontsize=12
        )

plt.tight_layout()
plt.show()