import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Load data
# -----------------------------------------------------------

script_dir              = os.path.dirname(os.path.abspath(__file__))
data_dir                = os.path.join(script_dir, "..", "Data_Generated")
cust_activation_timing  = os.path.normpath(os.path.join(data_dir, "02_1_customer_activation_timing.csv"))

df_cat                  = pd.read_csv(cust_activation_timing)
df_cat["year_month"]    = pd.to_datetime(df_cat["year_month"])
df_cat                  = df_cat.sort_values("year_month")

# -----------------------------------------------------------
# Charting (copy-paste)
# -----------------------------------------------------------

as_of_date  = df_cat["year_month"].max()
cutoff_date = as_of_date - pd.DateOffset(months=18)

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# ============================
# Top chart: Avg + Median Activation Days
# ============================

avg_line = ax1.plot(
    df_cat["year_month"],
    df_cat["avg_activation_days"],
    marker="o",
    markersize=8,
    markerfacecolor="black",
    markeredgecolor="black",
    label="Average Activation Days"
)

median_line = ax1.plot(
    df_cat["year_month"],
    df_cat["median_activation_days"],
    linewidth=2,
    label="Median Activation Days"
)

ax1.set_title(
    "Activation Timing by Signup Month (2023–2025)",
    fontsize=24,
    fontweight="bold"
)

ax1.set_ylabel("Activation Days", fontsize=18, fontweight="bold")

# Value labels above AVG dots
label_offset = 8
for x, y in zip(df_cat["year_month"], df_cat["avg_activation_days"]):
    ax1.text(
        x,
        y + label_offset,
        f"{int(round(y))}",
        ha="center",
        va="bottom",
        fontsize=9
    )

# Legend (large, upper-right)
ax1.legend(
    loc="upper right",
    fontsize=14,
    frameon=True
)

# ============================
# Bottom chart: Activated Customers
# ============================

bar_width = 25
bars = ax2.bar(
    df_cat["year_month"],
    df_cat["n_customers"],
    width=bar_width
)

ax2.set_ylabel("Activated Customers", fontsize=18, fontweight="bold")
ax2.set_xlabel("Year-Month")

# Bar labels
for bar in bars:
    h = bar.get_height()
    if h > 0:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=9
        )

# ============================
# Vertical guides + cutoff
# ============================

jan_mask = df_cat["year_month"].dt.month == 1
for x in df_cat.loc[jan_mask, "year_month"]:
    ax1.axvline(x=x, linestyle=":", linewidth=1, color="gray")
    ax2.axvline(x=x, linestyle=":", linewidth=1, color="gray")

ax1.axvline(x=cutoff_date, linestyle=":", linewidth=3, color="red")
ax2.axvline(x=cutoff_date, linestyle=":", linewidth=3, color="red")

# ============================
# Trend lines (AVG + Customers, pre-cutoff only)
# ============================

mask_pre = df_cat["year_month"] < cutoff_date
df_pre = df_cat.loc[mask_pre]

if len(df_pre) >= 2:
    x_pre = df_pre["year_month"].map(pd.Timestamp.toordinal).to_numpy()

    # Trend for avg activation days
    y_pre_avg = df_pre["avg_activation_days"].to_numpy()
    coef_avg = np.polyfit(x_pre, y_pre_avg, 1)
    trend_avg = np.polyval(coef_avg, x_pre)

    (trend_line,) = ax1.plot(
        df_pre["year_month"],
        trend_avg,
        linestyle="--",
        linewidth=2
    )
    trend_color = trend_line.get_color()

    # Trend for activated customers (same color)
    y_pre_cust = df_pre["n_customers"].to_numpy()
    coef_cust = np.polyfit(x_pre, y_pre_cust, 1)
    trend_cust = np.polyval(coef_cust, x_pre)

    ax2.plot(
        df_pre["year_month"],
        trend_cust,
        linestyle="--",
        linewidth=2,
        color=trend_color
    )

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
