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
script_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(script_dir, "..", "Data_Generated")

# Attach the file name into the path
pd_path                     = os.path.join(data_dir, "03_1_probability_of_default.csv")

# (Optional) normalize the path string
pd_path                     = os.path.normpath(pd_path)

# Load CSV files to dataframe
df_pd                       = pd.read_csv(pd_path)

df_pd_eligible              = df_pd.loc[df_pd["is_pd_eligible"] == 1].copy()

total_eligible_loans        = df_pd_eligible.shape[0]

total_defaults              = df_pd_eligible["is_default_12m"].sum()

overall_pd_12m              = total_defaults / total_eligible_loans


# -----------------------------------------------------------
# PD by risk tier at signup
# -----------------------------------------------------------

df_pd_eligible                  = df_pd.loc[df_pd["is_pd_eligible"] == 1].copy()

# Step 2: Aggregate by risk tier
df_pd_by_tier                   =   ( df_pd_eligible.groupby("risk_tier_at_signup")
                                        .agg(
                                                total_eligible_loans    = ("is_pd_eligible" , "size"),
                                                total_defaults_12m      = ("is_default_12m" , "sum" ),
                                            )
                                        .reset_index()
                                    )

df_pd_by_tier["pd_12m"]         = (df_pd_by_tier["total_defaults_12m"] / df_pd_by_tier["total_eligible_loans"]).round(2)

df_pd_by_tier["pd_12m_pct"]     = (df_pd_by_tier["pd_12m"] * 100).round(2)



# -----------------------------------------------------------
# PD by vintage (origination_month)
# -----------------------------------------------------------


df_pd_by_vintage                =   ( df_pd_eligible.groupby("origination_month")
                                        .agg(
                                                total_eligible_loans    = ("is_pd_eligible" , "size"),
                                                total_defaults_12m      = ("is_default_12m" , "sum" ),
                                            )
                                        .reset_index()
                                    )

df_pd_by_vintage["pd_12m"]      = (df_pd_by_vintage["total_defaults_12m"] / df_pd_by_vintage["total_eligible_loans"]).round(2)



print(df_pd_by_tier.head(100))
print()
print(df_pd_by_vintage.head(100))



# -----------------------------------------------------------
# Bar Chart: 12M PD by Risk Tier at Signup (fixed)
# -----------------------------------------------------------

df_plot_tier = df_pd_by_tier.copy()

# Make sure pd_12m is numeric (protects against dtype issues)
df_plot_tier["pd_12m"] = pd.to_numeric(df_plot_tier["pd_12m"], errors="coerce")
df_plot_tier = df_plot_tier.loc[df_plot_tier["pd_12m"].notna()].copy()

# Convert tier to string (protects against categorical/NaN issues)
df_plot_tier["risk_tier_at_signup"] = df_plot_tier["risk_tier_at_signup"].astype(str)

# Sort by PD for a clean visual
df_plot_tier = df_plot_tier.sort_values("pd_12m")

srs_pd = df_plot_tier["pd_12m"].astype(float)
srs_n = df_plot_tier["total_eligible_loans"].astype(int)
srs_tier = df_plot_tier["risk_tier_at_signup"]

# Dynamic headroom
max_pd = float(srs_pd.max())
y_top = max_pd * 1.30
if y_top < max_pd + 0.02:
    y_top = max_pd + 0.02

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(srs_tier, srs_pd)

ax.set_title("12M PD by Risk Tier at Signup", fontsize=16, fontweight="bold", pad=18)
ax.set_xlabel("Risk Tier", fontsize=13, fontweight="bold", labelpad=15)
ax.set_ylabel("12M PD", fontsize=13, fontweight="bold", labelpad=15)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.tick_params(axis="y", labelsize=11)
ax.tick_params(axis="x", labelsize=12)

ax.set_ylim(0, y_top)

label_offset = y_top * 0.02

# Add eligible loan count labels above bars
for idx in range(df_plot_tier.shape[0]):
    y_val = float(srs_pd.iloc[idx])
    n_val = int(srs_n.iloc[idx])

    ax.text(
        idx,
        y_val + label_offset,
        str(n_val),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=True
    )

# Legend explaining what the number means
ax.plot([], [], " ", label="Numbers above bars = Eligible loan count (N)")
ax.legend(loc="upper left", frameon=False, fontsize=11)

ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.show()



# -----------------------------------------------------------
# Bar Chart: 12M PD by Origination Month (with Eligible Loan Count labels)
# -----------------------------------------------------------

df_plot = df_pd_by_vintage.copy()

# Convert origination_month to datetime (your column is like "2023-01-01")
df_plot["origination_month_dt"] = pd.to_datetime(df_plot["origination_month"])

# Sort chronologically
df_plot = df_plot.sort_values("origination_month_dt")

# Month label for x-axis
df_plot["month_label"] = df_plot["origination_month_dt"].dt.strftime("%Y-%m")

# Values
srs_pd = df_plot["pd_12m"]
srs_n = df_plot["total_eligible_loans"].astype(int)

# Dynamic headroom so labels never go above chart area
max_pd = float(srs_pd.max())
y_top = max_pd * 1.25
if y_top < max_pd + 0.02:
    y_top = max_pd + 0.02

fig, ax = plt.subplots(figsize=(16, 6))

bars = ax.bar(df_plot["month_label"], srs_pd)

# Axis labels (bigger, bold, with padding)
ax.set_title("12M PD by Origination Month", fontsize=16, fontweight="bold", pad=18)

ax.set_xlabel("Origination Month", fontsize=13, fontweight="bold", labelpad=18)
ax.set_ylabel("12M PD", fontsize=13, fontweight="bold", labelpad=18)

# Format y-axis as percent and enlarge tick labels
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.tick_params(axis="y", labelsize=11)
ax.tick_params(axis="x", labelsize=10, pad=10)

# Make x labels all show and readable
plt.xticks(rotation=45, ha="right")

# Set y-limit with headroom for labels
ax.set_ylim(0, y_top)

# Put loan-count labels just above each bar (bigger, no overlap)
label_offset = y_top * 0.015
for idx in range(df_plot.shape[0]):
    y_val = float(srs_pd.iloc[idx])
    n_val = int(srs_n.iloc[idx])

    ax.text(
        idx,
        y_val + label_offset,
        str(n_val),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=True
    )

# Legend explaining label meaning
ax.plot([], [], " ", label="Numbers above bars = Eligible loan count (N)")
ax.legend(loc="upper right", frameon=False, fontsize=11)

ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.show()
