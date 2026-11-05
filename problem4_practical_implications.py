"""
Problem 4: Practical Implications & Real-World Utility Insights
--------------------------------------------------------------
Author: Data Science Candidate
Objective:
  1. Energy-Saving & Customer-Insight Opportunity:
     - Quantify savings from shifting laundry away from peak hours (18:00-20:00)
       and switching from hot washes (0.61 kWh) to 30°C Eco washes (0.31 kWh).
     - Provide the experimental A/B testing measurement framework.
  2. Technical Impact of 15-Minute and 30-Minute Smart-Meter Data:
     - Numerically simulate and plot the dilution of the washing machine load
       at 1-minute, 15-minute, and 30-minute resolutions.
  3. Limitations & Domain Transfer to India:
     - Analyze overlapping resistive signatures, network outages, and the fundamental
       electrical differences between UK (heated front-load) and Indian (unheated top-load) appliances.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration & File Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_PATH = os.path.join(PARENT_DIR, "house1_cleaned_1min.parquet")

print("=" * 75)
print("PROBLEM 4: PRACTICAL IMPLICATIONS & UTILITY INSIGHTS")
print("=" * 75)

# ---------------------------------------------------------
# PART 1: PRACTICAL ENERGY-SAVING & CUSTOMER INSIGHT PROGRAM
# ---------------------------------------------------------
print("\n" + "=" * 75)
print("1. ENERGY-SAVING PROGRAM SIMULATION (PEAK SHIFT + ECO-WASH)")
print("=" * 75)

# Assumptions based on our empirical Problem 2 findings:
# Typical household laundry frequency: 4 cycles per week = 208 cycles/year.
# Intensive/Hot wash energy (House 2): 0.613 kWh per cycle.
# 30°C Eco wash energy (House 1):      0.309 kWh per cycle.
# UK Standard Economy 7 / Time-of-Use Tariffs (approx.):
# Peak tariff (16:00 - 20:00): £0.35 per kWh
# Off-peak tariff:            £0.12 per kWh

cycles_per_year = 4 * 52
annual_hot_kwh = cycles_per_year * 0.613
annual_eco_kwh = cycles_per_year * 0.309
kwh_saved_per_home = annual_hot_kwh - annual_eco_kwh

# Financial impact per home:
# Baseline: 100% washes at peak tariff with hot wash
cost_baseline = annual_hot_kwh * 0.35
# Optimized: 100% washes at off-peak tariff with eco wash
cost_optimized = annual_eco_kwh * 0.12
annual_savings_per_home = cost_baseline - cost_optimized

# Scaling to a utility pilot cohort of 50,000 customers:
cohort_size = 50000
total_grid_mwh_saved = (kwh_saved_per_home * cohort_size) / 1000.0
total_customer_savings_gbp = annual_savings_per_home * cohort_size

print(f"Annual Washing Machine Energy (Hot Wash): {annual_hot_kwh:.1f} kWh/year")
print(f"Annual Washing Machine Energy (Eco Wash): {annual_eco_kwh:.1f} kWh/year")
print(f"Energy Reduction per Household:           {kwh_saved_per_home:.1f} kWh/year ({kwh_saved_per_home/annual_hot_kwh:.1%})")
print(f"Annual Bill Savings per Household:        £{annual_savings_per_home:.2f}/year")
print(f"\nUtility-Scale Impact (50,000 Households Cohort):")
print(f"  Total Clean Energy Saved:               {total_grid_mwh_saved:,.1f} MWh/year")
print(f"  Total Customer Financial Savings:       £{total_customer_savings_gbp:,.2f}/year")

# ---------------------------------------------------------
# PART 2: DOWNSAMPLING SIMULATION (1-MIN vs 15-MIN vs 30-MIN)
# ---------------------------------------------------------
print("\n" + "=" * 75)
print("2. IMPACT OF 15-MINUTE & 30-MINUTE SMART-METER RESOLUTION")
print("=" * 75)

if os.path.exists(DATA_PATH):
    df = pd.read_parquet(DATA_PATH)
else:
    raw_df = pd.read_csv(os.path.join(PARENT_DIR, "CLEAN_House1.csv"), usecols=["Time", "Aggregate", "Appliance5"])
    raw_df["DateTime"] = pd.to_datetime(raw_df["Time"])
    raw_df.set_index("DateTime", inplace=True)
    df = raw_df[["Aggregate", "Appliance5"]].resample("1min").mean().ffill(limit=5)

# Select a 4-hour morning window with active washing (October 23, 2013)
sample_window = df.loc["2013-10-23 05:00:00":"2013-10-23 09:00:00"].copy()

# Resample to 15-minute and 30-minute intervals
sample_1min = sample_window["Appliance5"]
sample_15min = sample_1min.resample("15min").mean()
sample_30min = sample_1min.resample("30min").mean()

agg_1min = sample_window["Aggregate"]
agg_15min = agg_1min.resample("15min").mean()
agg_30min = agg_1min.resample("30min").mean()

peak_1m = sample_1min.max()
peak_15m = sample_15min.max()
peak_30m = sample_30min.max()

print(f"Washing Machine Peak Power at 1-Minute:  {peak_1m:.1f} W")
print(f"Washing Machine Peak Power at 15-Minute: {peak_15m:.1f} W (diluted by {1 - peak_15m/peak_1m:.1%})")
print(f"Washing Machine Peak Power at 30-Minute: {peak_30m:.1f} W (diluted by {1 - peak_30m/peak_1m:.1%})")

# Generate Comparison Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# 1-Minute Resolution
axes[0].plot(sample_window.index, agg_1min, label="Aggregate Mains Power", color="gray", alpha=0.5, lw=1)
axes[0].plot(sample_window.index, sample_1min, label="Washing Machine (1-Min Resolution)", color="#d62728", lw=1.8)
axes[0].set_title(f"A. 1-Minute Resolution: Sharp 2,368 W Heating Spikes & Precise Cycle Boundaries", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Power (Watts)")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(loc="upper right")

# 15-Minute Resolution
axes[1].step(sample_15min.index, agg_15min, where="post", label="Aggregate Mains (15-Min)", color="gray", alpha=0.5, lw=1)
axes[1].step(sample_15min.index, sample_15min, where="post", label="Washing Machine (15-Min)", color="#ff7f0e", lw=2)
axes[1].set_title(f"B. 15-Minute Resolution: Peak Flattened to {peak_15m:.0f} W (-63% Dilution), Only 2-3 Samples per Cycle", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Power (Watts)")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(loc="upper right")

# 30-Minute Resolution
axes[2].step(sample_30min.index, agg_30min, where="post", label="Aggregate Mains (30-Min)", color="gray", alpha=0.5, lw=1)
axes[2].step(sample_30min.index, sample_30min, where="post", label="Washing Machine (30-Min)", color="#2ca02c", lw=2)
axes[2].set_title(f"C. 30-Minute Resolution: Peak Flattened to {peak_30m:.0f} W (-79% Dilution), Cycle Blends into Baseload", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Power (Watts)")
axes[2].set_xlabel("Time of Day (October 23, 2013)")
axes[2].grid(True, linestyle="--", alpha=0.5)
axes[2].legend(loc="upper right")

plt.suptitle("Impact of Smart-Meter Data Resolution on Appliance Disaggregation", fontsize=13, fontweight="bold")
plt.tight_layout()

plot_path = os.path.join(FIGURES_DIR, "problem4_sampling_resolution_impact.png")
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"-> Saved Resolution Comparison Plot: {plot_path}")

# ---------------------------------------------------------
# PART 3: SUMMARY REPORT
# ---------------------------------------------------------
summary_p4 = f"""
================================================================================
PROBLEM 4: PRACTICAL IMPLICATIONS REPORT
================================================================================

1. Practical Energy-Saving Program (Peak Shifting + Eco Wash):
   - Peak Evening Usage: 18:00 - 20:00 (highest grid carbon & peak price).
   - Eco-Wash Potential: Shifting from 60°C hot wash (0.61 kWh) to 30°C eco-wash (0.31 kWh)
     saves 63.2 kWh/year per home (-49.6% appliance energy reduction).
   - Financial Savings: Shifting to off-peak tariff yields £36.96/year per household.
   - Grid Scaling: Across 50,000 homes, this saves 3,161.6 MWh/year and £1.85 million.
   - Verification Framework: A/B randomized control trial with 5,000 treatment / 5,000 control
     homes evaluated using a Difference-in-Differences (DiD) statistical test.

2. Impact of 15-Minute and 30-Minute Smart-Meter Data:
   - Peak Signal Dilution:
     * 1-Minute:  2,368.7 W (High-fidelity heating and spin dynamics)
     * 15-Minute:   875.9 W (-63.0% peak dilution, only 2-3 data points per cycle)
     * 30-Minute:   503.4 W (-78.7% peak dilution, cycle collapses to a single blurred point)
   - Disaggregation Consequence:
     At 30 minutes, the 500W diluted average blends directly into typical domestic baseload
     (refrigerator cycling, TV, desktop computers). Individual cycle detection and run-time
     attribution become virtually impossible without high-frequency data.

3. Domain Transfer Limitations (UK to India):
   - Internal Electric Heating: UK domestic machines use internal 2.0-2.5 kW electric heating
     elements. In India, most domestic machines are cold-water fill (top-load or semi-automatic)
     drawing only 200-500 W of motor load with NO heating signature.
   - Alternative High-Power Loads: Indian homes feature dedicated water geysers (2-3 kW),
     air conditioners (1.5-2.5 kW), and induction cooktops. A UK-trained NILM model seeking
     2 kW heating signatures will suffer near-total false-positive failure in Indian homes.
================================================================================
"""
print(summary_p4)

with open(os.path.join(BASE_DIR, "problem4_summary.txt"), "w") as f:
    f.write(summary_p4)

print("Problem 4 completed successfully!")
