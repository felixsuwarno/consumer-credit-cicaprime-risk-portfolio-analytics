import os
import pandas as pd

# -----------------------------------------------------------
# Load data
# -----------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "Data_Generated")

ltv_path = os.path.normpath(
    os.path.join(data_dir, "02_3a_customer_LTV_180d.csv")
)

summary_path = os.path.normpath(
    os.path.join(data_dir, "02_3b_customer_LTV_180d_summary.csv")
)

df_ltv = pd.read_csv(ltv_path)
df_summary = pd.read_csv(summary_path)

# -----------------------------------------------------------
# Question 1: Total collected
# -----------------------------------------------------------

total_collected_180d = df_ltv["total_payment_180d"].sum()

# -----------------------------------------------------------
# Question 2: Total loss
# -----------------------------------------------------------

total_loss_180d = df_ltv["total_loss_180d"].sum()

# -----------------------------------------------------------
# Question 3: Most profitable customer
# -----------------------------------------------------------

df_sorted = df_ltv.sort_values("net_ltv_180d", ascending=False)
top_customer = df_sorted.iloc[0]

print("\n===== 2.3 LTV Report =====")

print("\nTotal collected in 180 days:")
print(f"{total_collected_180d:,.2f}")

print("\nTotal loss in 180 days:")
print(f"{total_loss_180d:,.2f}")

print("\nMost profitable customer:")
print(f"Customer ID     : {int(top_customer['customer_id'])}")
print(f"Net LTV 180d    : {top_customer['net_ltv_180d']:,.2f}")
print(f"Payment 180d    : {top_customer['total_payment_180d']:,.2f}")
print(f"Loss 180d       : {top_customer['total_loss_180d']:,.2f}")
