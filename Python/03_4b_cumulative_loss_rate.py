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
clr_filename                = "03_4b_cumulative_loss_rate.csv"
clr_path_local              = os.path.join(data_dir, clr_filename)

# Load dataset

df_clr12m                   = pd.read_csv(clr_path_local)

# ----------------------------
# CHART
# ----------------------------
df_clr12m                   = df_clr12m.copy()
df_clr12m["origination_month"] = pd.to_datetime(df_clr12m["origination_month"])
df_clr12m                   = df_clr12m.sort_values("origination_month")

srs_x                       = df_clr12m["origination_month"]
srs_n_loans                 = df_clr12m["n_loans_in_vintage"]
srs_total_loss_12m          = df_clr12m["total_loss_12m"]
srs_clr_12m                 = df_clr12m["clr_12m"]

fig                         = plt.figure(figsize=(14,6))
ax1                         = plt.gca()

bar_loans                   = ax1.bar(
    srs_x,
    srs_n_loans,
    width=20,
    label="Loans in Vintage (N)"
)

bar_loss                    = ax1.bar(
    srs_x,
    srs_total_loss_12m,
    width=10,
    label="12M Cumulative Loss (Total $)"
)

ax1.set_xlabel("Origination Month", fontsize=20)
ax1.set_ylabel("Loans / Total Loss", fontsize=20)

# Add headroom for bars
max_left                    = max(srs_n_loans.max(), srs_total_loss_12m.max())
ax1.set_ylim(0, max_left * 1.25)

ax2                         = ax1.twinx()

ax2.plot(
    srs_x,
    srs_clr_12m,
    label="12M Cumulative Loss Rate (CLR)",
    color="black",
    linewidth=3,
    marker="o",
    markersize=9,
    markeredgewidth=2.5
)

ax2.set_ylabel("12M Cumulative Loss Rate", fontsize=20)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

# Add headroom for line
max_right                   = srs_clr_12m.max()
ax2.set_ylim(0, max_right * 1.25)

plt.title(
    "12M Cumulative Loss Rate and Loan Counts by Vintage",
    fontsize=24,
    fontweight="bold"
)

# ----------------------------
# BAR LABELS
# ----------------------------
for rect in bar_loans:
    x_pos                   = rect.get_x() + rect.get_width() / 2
    y_val                   = rect.get_height()
    if pd.notna(y_val) and y_val != 0:
        ax1.text(
            x_pos,
            y_val,
            f"{int(y_val)}",
            ha="center",
            va="bottom",
            color=rect.get_facecolor(),
            fontsize=12,
            fontweight="bold"
        )

for rect in bar_loss:
    x_pos                   = rect.get_x() + rect.get_width() / 2
    y_val                   = rect.get_height()
    if pd.notna(y_val) and y_val != 0:
        ax1.text(
            x_pos,
            y_val,
            f"{y_val:,.0f}",
            ha="center",
            va="bottom",
            color=rect.get_facecolor(),
            fontsize=12,
            fontweight="bold"
        )

# ----------------------------
# LINE LABELS
# ----------------------------
y_max                       = float(np.nanmax(srs_clr_12m)) if len(srs_clr_12m) else 0.0
y_offset                    = max(0.15, y_max * 0.03)

for x_val, y_val in zip(srs_x, srs_clr_12m):
    if pd.notna(y_val):
        ax2.text(
            x_val,
            y_val + y_offset,
            f"{y_val:.2f}%",
            ha="center",
            va="bottom",
            color="black",
            fontsize=12,
            fontweight="bold"
        )

ax1.set_xticks(srs_x)
ax1.set_xticklabels(
    [d.strftime("%Y-%m") for d in srs_x],
    rotation=45,
    ha="right",
    fontsize=14
)

handles1, labels1           = ax1.get_legend_handles_labels()
handles2, labels2           = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left",
    fontsize=12
)

plt.tight_layout()
plt.show()