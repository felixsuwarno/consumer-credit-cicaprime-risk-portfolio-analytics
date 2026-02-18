import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates


# Pandas display settings
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 2000)

# -----------------------------------------------------------
# Load data
# -----------------------------------------------------------

script_dir      = os.path.dirname(os.path.abspath(__file__))
data_dir        = os.path.join(script_dir, "..", "Data_Generated")
pareto_path     = os.path.normpath(os.path.join(data_dir, "02_4_value_concentration.csv"))
df_pareto       = pd.read_csv(pareto_path)

# -----------------------------------------------------------
# Pareto curve visualization (2.4)
# -----------------------------------------------------------

# Ensure numeric
df_pareto["pareto_x"]   = pd.to_numeric(df_pareto["pareto_x"], errors="coerce")
df_pareto["pareto_y"]   = pd.to_numeric(df_pareto["pareto_y"], errors="coerce")

# Customer count
n_customers             = df_pareto.shape[0]

# Font scaling (approx 2x normal)
title_fontsize          = 24
label_fontsize          = 20
tick_fontsize           = 18

# Output path
charts_dir              = os.path.join(script_dir, "..", "Charts")
os.makedirs(charts_dir, exist_ok=True)

chart_path              = os.path.normpath(
    os.path.join(charts_dir, "02_4_value_concentration_pareto_curve.png")
)

# Plot
fig, ax                 = plt.subplots(figsize=(8, 6))

ax.plot(
    df_pareto["pareto_x"],
    df_pareto["pareto_y"],
    linewidth=2
)

# 45-degree reference line
ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1)

ax.set_title(
    f"02.4 Value Concentration — Pareto Curve (LTV 180d)\nPortfolio Period: 2023–2025 | Total Customers: {n_customers}",
    fontsize=title_fontsize
)

ax.set_xlabel("Cumulative % of Customers", fontsize=label_fontsize)
ax.set_ylabel("Cumulative % of Net LTV (180d)", fontsize=label_fontsize)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

ax.xaxis.set_major_formatter(mtick.PercentFormatter(100))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))

ax.tick_params(axis="both", labelsize=tick_fontsize)

ax.grid(True)

plt.tight_layout()
plt.savefig(chart_path, dpi=200)
plt.show()

print("Saved chart to:", chart_path)
