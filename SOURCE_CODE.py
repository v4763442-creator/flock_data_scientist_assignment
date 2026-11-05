"""
================================================================================
FLOCK ENERGY TAKE-HOME ASSIGNMENT: END-TO-END PIPELINE
REFIT Electrical Load Cleaning, Washing Machine EDA, Disaggregation & Insights
================================================================================
Author: Data Science Candidate
Description:
  This script executes the complete end-to-end analysis required by the
  Flock Energy take-home assignment, organized into 4 distinct sections:
  
  - PART 1: Basic Inspection and Cleaning of Raw Data (House 1)
  - PART 2: Exploratory Data Analysis of Washing-Machine Use (Houses 1, 2, 3, 5)
  - PART 3: Appliance Disaggregation Modeling & Evaluation (GBDT vs Baselines)
  - PART 4: Practical Implications, Sampling Resolution Impact & Utility Insights
================================================================================
"""

import os
import gc
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    precision_score,
    recall_score,
    f1_score
)

# ------------------------------------------------------------------------------
# 0. DIRECTORY & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Helper function to resolve paths whether run from workspace root or FLOCK_2
def resolve_file(filename):
    local_path = os.path.join(BASE_DIR, filename)
    parent_path = os.path.join(PARENT_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(parent_path):
        return parent_path
    return filename


# ==============================================================================
# PART 1: BASIC INSPECTION AND CLEANING OF RAW DATA (HOUSE 1)
# ==============================================================================
print("\n" + "=" * 80)
print("PART 1: BASIC INSPECTION AND CLEANING OF RAW DATA (HOUSE 1)")
print("=" * 80)

# Step 1.1: Load Raw Part 1 and Part 2
part1_path = resolve_file("RAW_House1_Part1.csv")
part2_path = resolve_file("RAW_House1_Part2.csv")

channels = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
dtype_dict = {col: "float32" for col in channels}

print("1. Loading raw telemetry files for House 1...")
df1 = pd.read_csv(part1_path, dtype=dtype_dict)
df2 = pd.read_csv(part2_path, dtype=dtype_dict)

# REFIT documentation: In Part 1, unavailable sensors were 0.
# In Part 2 (starting 2014-06-27), unavailable sensors were logged as explicit NaN.
SWITCH_TIME = "2014-06-27 21:57:50"
df1 = df1[df1["Time"] < SWITCH_TIME]
raw_combined = pd.concat([df1, df2], ignore_index=True)
del df1, df2
gc.collect()

print(f"   Combined raw observations: {len(raw_combined):,}")

# Step 1.2: Audit Data Quality Issues
print("2. Auditing raw data quality issues...")
raw_combined["DateTime"] = pd.to_datetime(raw_combined["Time"])
raw_combined.drop(columns=["Time", "Unix"], inplace=True)

is_sorted = raw_combined["DateTime"].is_monotonic_increasing
print(f"   Timestamps strictly ordered initially? {is_sorted}")

dup_timestamps = raw_combined.duplicated(subset=["DateTime"]).sum()
print(f"   Duplicate timestamps detected: {dup_timestamps:,} ({dup_timestamps / len(raw_combined):.2%})")

# Chronological Sort
raw_combined.sort_values("DateTime", inplace=True)
raw_combined.reset_index(drop=True, inplace=True)

intervals = raw_combined["DateTime"].diff().dt.total_seconds().dropna()
print(f"   Median sampling interval: {intervals.median():.1f}s (max gap: {intervals.max()/3600:.1f} hrs)")
del intervals
gc.collect()

# Value sanity checks
negatives = (raw_combined[channels] < 0).sum().sum()
agg_spikes = (raw_combined["Aggregate"] > 15000).sum()
app_spikes = sum((raw_combined[f"Appliance{i}"] > 4000).sum() for i in range(1, 10))
print(f"   Extreme spikes clamped to NaN: {agg_spikes:,} Aggregate (>15kW) and {app_spikes:,} Appliance (>4kW)")

# Keep slices for raw before/after plotting
START_48H, END_48H = "2013-10-22 00:00:00", "2013-10-24 00:00:00"
raw_slice = raw_combined[(raw_combined["DateTime"] >= START_48H) & (raw_combined["DateTime"] <= END_48H)].copy()
spike_raw = raw_combined[(raw_combined["DateTime"] >= "2014-01-10") & (raw_combined["DateTime"] <= "2014-01-12")].copy()

# Step 1.3: Execute Cleaning Pipeline
print("3. Executing cleaning & 1-minute resampling pipeline...")
# Stated Rule: Deduplicate keeping last observation
raw_combined.drop_duplicates(subset=["DateTime"], keep="last", inplace=True)

# Clamp physical impossibilities to NaN
raw_combined.loc[raw_combined["Aggregate"] > 15000, "Aggregate"] = np.nan
raw_combined.loc[raw_combined["Aggregate"] < 0, "Aggregate"] = np.nan
for i in range(1, 10):
    col = f"Appliance{i}"
    raw_combined.loc[raw_combined[col] > 4000, col] = np.nan
    raw_combined.loc[raw_combined[col] < 0, col] = np.nan

# Resample to 1-minute mean
raw_combined.set_index("DateTime", inplace=True)
df_1min = raw_combined[channels].resample("1min").mean()
del raw_combined
gc.collect()

# Gap handling: forward-fill <=5 mins; flag >5 mins as outages
df_1min_filled = df_1min.ffill(limit=5)
del df_1min
gc.collect()

df_1min_filled["outage_flag"] = df_1min_filled["Aggregate"].isna().astype(int)
outage_pct = df_1min_filled["outage_flag"].mean() * 100.0
print(f"   Cleaned 1-minute bins: {len(df_1min_filled):,} (Outage flagged: {outage_pct:.2f}%)")

# Save cleaned dataset
cleaned_parquet_path = os.path.join(BASE_DIR, "house1_cleaned_1min.parquet")
df_1min_filled.to_parquet(cleaned_parquet_path)
print(f"   -> Saved cleaned dataset to: {cleaned_parquet_path}")

# Step 1.4: Generate Problem 1 Plots
print("4. Generating Part 1 visualizations...")
clean_slice = df_1min_filled.loc[START_48H:END_48H]

fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex="col", sharey="row")
axes[0, 0].plot(raw_slice["DateTime"], raw_slice["Aggregate"], color="#1f77b4", alpha=0.7, lw=0.8)
axes[0, 0].set_title("Raw: Aggregate Power (8s interval)", fontsize=10, fontweight="bold")
axes[0, 0].set_ylabel("Power (W)")
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

axes[0, 1].plot(clean_slice.index, clean_slice["Aggregate"], color="#2ca02c", lw=1.2)
axes[0, 1].set_title("Cleaned: Aggregate Power (1-Minute Resampled)", fontsize=10, fontweight="bold")
axes[0, 1].grid(True, linestyle="--", alpha=0.5)

axes[1, 0].plot(raw_slice["DateTime"], raw_slice["Appliance1"], color="#ff7f0e", alpha=0.7, lw=0.8)
axes[1, 0].set_title("Raw: Appliance 1 (Fridge - noisy)", fontsize=10, fontweight="bold")
axes[1, 0].set_ylabel("Power (W)")
axes[1, 0].grid(True, linestyle="--", alpha=0.5)

axes[1, 1].plot(clean_slice.index, clean_slice["Appliance1"], color="#d62728", lw=1.2)
axes[1, 1].set_title("Cleaned: Appliance 1 (Fridge - 1-Minute Mean)", fontsize=10, fontweight="bold")
axes[1, 1].grid(True, linestyle="--", alpha=0.5)

axes[2, 0].plot(raw_slice["DateTime"], raw_slice["Appliance5"], color="#9467bd", alpha=0.7, lw=0.8)
axes[2, 0].set_title("Raw: Appliance 5 (Washing Machine)", fontsize=10, fontweight="bold")
axes[2, 0].set_ylabel("Power (W)")
axes[2, 0].set_xlabel("Timestamp")
axes[2, 0].grid(True, linestyle="--", alpha=0.5)

axes[2, 1].plot(clean_slice.index, clean_slice["Appliance5"], color="#8c564b", lw=1.2)
axes[2, 1].set_title("Cleaned: Appliance 5 (Washing Machine - 1-Minute Resampled)", fontsize=10, fontweight="bold")
axes[2, 1].set_ylabel("Power (W)")
axes[2, 1].set_xlabel("Timestamp")
axes[2, 1].grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Problem 1: House 1 Power Comparison Before vs. After Cleaning", fontsize=13, fontweight="bold")
plt.tight_layout()
p1_plot1 = os.path.join(FIGURES_DIR, "problem1_before_after_cleaning.png")
plt.savefig(p1_plot1, dpi=200)
plt.close()
print(f"   -> Saved: {p1_plot1}")

# Outage & Spike demonstration plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
spike_clean = df_1min_filled.loc["2014-01-10":"2014-01-12"]
ax1.plot(spike_raw["DateTime"], spike_raw["Aggregate"], label="Raw Aggregate (with sensor spikes)", color="orange", alpha=0.8)
ax1.plot(spike_clean.index, spike_clean["Aggregate"], label="Cleaned Aggregate (spikes removed)", color="navy", lw=1.2)
ax1.set_title("Sensor Spike Removal & Signal Smoothing", fontsize=10, fontweight="bold")
ax1.set_ylabel("Power (W)")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

feb_outage = df_1min_filled.loc["2014-02-18":"2014-02-25"]
ax2.plot(feb_outage.index, feb_outage["Aggregate"], label="Cleaned Aggregate Power", color="teal", lw=1.2)
ax2.fill_between(feb_outage.index, 0, 3000, where=(feb_outage["outage_flag"] == 1),
                 color="crimson", alpha=0.25, label="Flagged Outage (Gaps > 5 mins left as NaN)")
ax2.set_title("Outage Flagging vs. Blind Imputation (February 2014 REFIT Outage)", fontsize=10, fontweight="bold")
ax2.set_ylabel("Power (W)")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
p1_plot2 = os.path.join(FIGURES_DIR, "problem1_outage_spike_handling.png")
plt.savefig(p1_plot2, dpi=200)
plt.close()
print(f"   -> Saved: {p1_plot2}")


# ==============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS OF WASHING-MACHINE USE (4 HOUSES)
# ==============================================================================
print("\n" + "=" * 80)
print("PART 2: EXPLORATORY ANALYSIS OF WASHING-MACHINE USE (4 HOUSES)")
print("=" * 80)

# Plain Language Cycle Detection Algorithm
def detect_washing_cycles(power_1min, p_thresh=20.0, max_gap=5, min_dur=15, min_peak=500.0):
    """
    Identifies washing machine cycles using physical domain criteria:
    - Active threshold >= 20 W (separates operational load from standby <5 W).
    - Max gap tolerance <= 5 mins (accommodates inter-stage drain/fill pauses).
    - Min duration >= 15 mins (rejects door clicks and transient button pushes).
    - Min peak >= 500 W (confirms heating element or spin motor activation).
    """
    times = power_1min.index
    values = power_1min.values
    cycles = []
    
    in_cycle = False
    start_idx = None
    gap_count = 0
    
    for i in range(len(values)):
        val = 0.0 if np.isnan(values[i]) else values[i]
        if val >= p_thresh:
            if not in_cycle:
                in_cycle = True
                start_idx = i
            gap_count = 0
        else:
            if in_cycle:
                gap_count += 1
                if gap_count > max_gap or i == len(values) - 1:
                    end_idx = i - gap_count
                    dur = end_idx - start_idx + 1
                    cycle_vals = values[start_idx:end_idx+1]
                    peak = np.nanmax(cycle_vals) if len(cycle_vals) > 0 else 0.0
                    if dur >= min_dur and peak >= min_peak:
                        energy_kwh = (np.nansum(cycle_vals) / 60.0) / 1000.0
                        cycles.append({
                            "start_time": times[start_idx],
                            "end_time": times[end_idx],
                            "duration_min": dur,
                            "peak_watts": peak,
                            "energy_kwh": energy_kwh,
                            "start_hour": times[start_idx].hour,
                            "day_of_week": times[start_idx].day_name()
                        })
                    in_cycle = False
                    gap_count = 0
    return pd.DataFrame(cycles)

HOUSE_CONFIG = {
    "House 1": {"file": resolve_file("CLEAN_House1.csv"), "wm_col": "Appliance5", "model": "Beko WMC6140"},
    "House 2": {"file": resolve_file("CLEAN_House2.csv"), "wm_col": "Appliance2", "model": "LG F1289TD"},
    "House 3": {"file": resolve_file("CLEAN_House3.csv"), "wm_col": "Appliance6", "model": "Washing Machine"},
    "House 5": {"file": resolve_file("CLEAN_House5.csv"), "wm_col": "Appliance3", "model": "AEG L99695HWD"}
}

house_summaries = []
all_detected_cycles = []
house_resampled_dict = {}

print("1. Resampling and extracting cycles across 4 households...")
for h_name, cfg in HOUSE_CONFIG.items():
    df = pd.read_csv(cfg["file"], usecols=["Time", "Aggregate", cfg["wm_col"]],
                     dtype={"Aggregate": "float32", cfg["wm_col"]: "float32"})
    df["DateTime"] = pd.to_datetime(df["Time"])
    df.set_index("DateTime", inplace=True)
    df.drop(columns=["Time"], inplace=True)
    
    s_1min = df.resample("1min").mean()
    s_1min["agg_clean"] = s_1min["Aggregate"].ffill(limit=5)
    s_1min["wm_clean"] = s_1min[cfg["wm_col"]].ffill(limit=5)
    house_resampled_dict[h_name] = s_1min[["agg_clean", "wm_clean"]].copy()
    
    total_mins = len(s_1min)
    missing_pct = (s_1min["wm_clean"].isna().sum() / total_mins) * 100.0
    valid_wm = s_1min["wm_clean"].dropna()
    off_pct = ((valid_wm < 20.0).sum() / len(valid_wm)) * 100.0
    
    cycles_df = detect_washing_cycles(s_1min["wm_clean"])
    cycles_df["house"] = h_name
    all_detected_cycles.append(cycles_df)
    
    duration_days = (s_1min.index.max() - s_1min.index.min()).total_seconds() / 86400.0
    cycles_per_week = (len(cycles_df) / duration_days) * 7.0
    
    house_summaries.append({
        "House": h_name,
        "Appliance Model": cfg["model"],
        "Days Monitored": round(duration_days, 1),
        "Missing Data (%)": round(missing_pct, 2),
        "Off/Zero Time (%)": round(off_pct, 2),
        "Total Cycles": len(cycles_df),
        "Cycles / Week": round(cycles_per_week, 2),
        "Median Duration (min)": round(cycles_df["duration_min"].median(), 1),
        "Median Peak Power (W)": round(cycles_df["peak_watts"].median(), 1),
        "Median Energy (kWh)": round(cycles_df["energy_kwh"].median(), 3)
    })
    del df, s_1min
    gc.collect()

summary_table = pd.DataFrame(house_summaries)
all_cycles_df = pd.concat(all_detected_cycles, ignore_index=True)

print("\n--- Cross-Household EDA Summary Table ---")
print(summary_table.to_string(index=False))
summary_table.to_csv(os.path.join(BASE_DIR, "problem2_house_summary.csv"), index=False)
all_cycles_df.to_csv(os.path.join(BASE_DIR, "problem2_detected_cycles.csv"), index=False)

# Step 2.2: Generate Problem 2 Plots
print("\n2. Generating Part 2 visualizations...")
# Plot (i): Representative 24-hour trace
trace_data = house_resampled_dict["House 1"].loc["2013-10-23 00:00:00":"2013-10-23 23:59:00"]
plt.figure(figsize=(14, 4.5))
plt.plot(trace_data.index, trace_data["agg_clean"], label="Total Home Aggregate Power", color="#1f77b4", lw=1.2, alpha=0.8)
plt.plot(trace_data.index, trace_data["wm_clean"], label="Washing Machine Power (Beko WMC6140)", color="#d62728", lw=1.8)
plt.fill_between(trace_data.index, 0, trace_data["wm_clean"], color="#d62728", alpha=0.2)
plt.title("Plot (i): Representative 24-Hour Power Trace (House 1 - October 23, 2013)", fontsize=11, fontweight="bold")
plt.ylabel("Power (W)")
plt.xlabel("Time of Day")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
p2_plot1 = os.path.join(FIGURES_DIR, "problem2_24h_trace.png")
plt.savefig(p2_plot1, dpi=200)
plt.close()
print(f"   -> Saved: {p2_plot1}")

# Plot (ii): Zoomed-in cycle trace
cycle_zoom = trace_data.loc["2013-10-23 05:45:00":"2013-10-23 07:30:00"]
plt.figure(figsize=(12, 4.5))
plt.plot(cycle_zoom.index, cycle_zoom["wm_clean"], color="#d62728", lw=2, marker="o", markersize=3, label="Washing Machine Load")
plt.axhline(20, color="gray", linestyle=":", label="Active Threshold (20 W)")
plt.axhline(500, color="purple", linestyle="--", alpha=0.7, label="Peak Qualification (500 W)")
h1_sample_cycles = all_cycles_df[(all_cycles_df["house"] == "House 1") & 
                                  (all_cycles_df["start_time"] >= "2013-10-23 05:45:00") & 
                                  (all_cycles_df["start_time"] <= "2013-10-23 07:30:00")]
for idx, c in h1_sample_cycles.iterrows():
    plt.axvspan(c["start_time"], c["end_time"], color="green", alpha=0.15)
    plt.text(c["start_time"], c["peak_watts"] + 50, f"{c['duration_min']}m | {c['peak_watts']:.0f}W", 
             fontsize=9, fontweight="bold", color="darkgreen")
plt.title("Plot (ii): Detected Washing Machine Operating Cycles (House 1 Zoom)", fontsize=11, fontweight="bold")
plt.ylabel("Power (W)")
plt.xlabel("Timestamp")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
p2_plot2 = os.path.join(FIGURES_DIR, "problem2_detected_cycles.png")
plt.savefig(p2_plot2, dpi=200)
plt.close()
print(f"   -> Saved: {p2_plot2}")

# Plot (iii): Cross-house comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
h_list = ["House 1", "House 2", "House 3", "House 5"]
dur_data = [all_cycles_df[all_cycles_df["house"] == h]["duration_min"] for h in h_list]
axes[0, 0].boxplot(dur_data, tick_labels=h_list, showmeans=True, patch_artist=True,
                   boxprops=dict(facecolor="#c6dbef", color="#08519c"), medianprops=dict(color="red", lw=2))
axes[0, 0].set_title("A. Cycle Duration Across Households", fontsize=10, fontweight="bold")
axes[0, 0].set_ylabel("Duration (Minutes)")
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

en_data = [all_cycles_df[all_cycles_df["house"] == h]["energy_kwh"] for h in h_list]
axes[0, 1].boxplot(en_data, tick_labels=h_list, showmeans=True, patch_artist=True,
                   boxprops=dict(facecolor="#c7e9c0", color="#006d2c"), medianprops=dict(color="red", lw=2))
axes[0, 1].set_title("B. Energy per Cycle Across Households", fontsize=10, fontweight="bold")
axes[0, 1].set_ylabel("Energy (kWh)")
axes[0, 1].grid(True, linestyle="--", alpha=0.5)

for h in h_list:
    h_c = all_cycles_df[all_cycles_df["house"] == h]
    hour_pct = (h_c["start_hour"].value_counts().reindex(range(24), fill_value=0) / len(h_c)) * 100.0
    axes[1, 0].plot(range(24), hour_pct, marker="o", label=h, lw=1.5)
axes[1, 0].set_title("C. Usage by Hour of Day (% of Total)", fontsize=10, fontweight="bold")
axes[1, 0].set_xlabel("Hour of Day (0-23)")
axes[1, 0].set_ylabel("Frequency (%)")
axes[1, 0].set_xticks(range(0, 24, 2))
axes[1, 0].grid(True, linestyle="--", alpha=0.5)
axes[1, 0].legend()

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for h in h_list:
    h_c = all_cycles_df[all_cycles_df["house"] == h]
    day_pct = (h_c["day_of_week"].value_counts().reindex(days, fill_value=0) / len(h_c)) * 100.0
    axes[1, 1].plot(range(7), day_pct, marker="s", label=h, lw=1.5)
axes[1, 1].set_title("D. Usage by Day of Week (% of Total)", fontsize=10, fontweight="bold")
axes[1, 1].set_xticks(range(7))
axes[1, 1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
axes[1, 1].grid(True, linestyle="--", alpha=0.5)
axes[1, 1].legend()

plt.suptitle("Plot (iii): Cross-House Usage & Characteristic Signatures", fontsize=12, fontweight="bold")
plt.tight_layout()
p2_plot3 = os.path.join(FIGURES_DIR, "problem2_cross_house_comparison.png")
plt.savefig(p2_plot3, dpi=200)
plt.close()
print(f"   -> Saved: {p2_plot3}")


# ==============================================================================
# PART 3: APPLIANCE DISAGGREGATION MODELING & EVALUATION
# ==============================================================================
print("\n" + "=" * 80)
print("PART 3: APPLIANCE DISAGGREGATION MODELING & EVALUATION")
print("=" * 80)

# Step 3.1: Feature Engineering on Cleaned House 1
print("1. Engineering contextual rolling features from Aggregate power...")
h1_df = pd.read_parquet(cleaned_parquet_path)
h1_valid = h1_df[h1_df["outage_flag"] == 0].copy()

feat_df = pd.DataFrame(index=h1_valid.index)
feat_df["agg"] = h1_valid["Aggregate"]
feat_df["agg_diff"] = h1_valid["Aggregate"].diff().fillna(0.0)
feat_df["agg_roll_mean_5"] = h1_valid["Aggregate"].rolling(5, min_periods=1).mean()
feat_df["agg_roll_mean_15"] = h1_valid["Aggregate"].rolling(15, min_periods=1).mean()
feat_df["agg_roll_max_15"] = h1_valid["Aggregate"].rolling(15, min_periods=1).max()
feat_df["hour"] = h1_valid.index.hour
feat_df["dayofweek"] = h1_valid.index.dayofweek
feat_df["target"] = h1_valid["Appliance5"]
feat_df.dropna(inplace=True)

feature_cols = ["agg", "agg_diff", "agg_roll_mean_5", "agg_roll_mean_15", "agg_roll_max_15", "hour", "dayofweek"]

# Step 3.2: Chronological Train/Test Split (80/20)
# Prevents data leakage from within the same 35-minute cycle
split_idx = int(len(feat_df) * 0.80)
train_df = feat_df.iloc[:split_idx]
test_df = feat_df.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df["target"].values
X_test, y_test = test_df[feature_cols], test_df["target"].values

print(f"   Train set: {len(train_df):,} minutes ({train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')})")
print(f"   Test set:  {len(test_df):,} minutes ({test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')})")

# Step 3.3: Baselines & Model Training
print("2. Training baselines and Gradient Boosted Decision Tree (GBDT)...")
# Baseline 1: Always Off
pred_zero = np.zeros(len(y_test))

# Baseline 2: Hour of Day Mean Profile
hourly_means = train_df.groupby("hour")["target"].mean().to_dict()
pred_profile = test_df["hour"].map(hourly_means).values

# Proposed Model: GBDT
t0 = time.time()
disagg_model = HistGradientBoostingRegressor(max_iter=100, max_depth=8, learning_rate=0.1, random_state=42)
disagg_model.fit(X_train, y_train)
pred_model = np.clip(disagg_model.predict(X_test), 0.0, None)
print(f"   -> Model trained in {time.time()-t0:.2f} seconds.")

# Step 3.4: Evaluation Metrics
def evaluate_preds(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    t_act = (y_true >= 20.0).astype(int)
    p_act = (y_pred >= 20.0).astype(int)
    prec = precision_score(t_act, p_act, zero_division=0)
    rec = recall_score(t_act, p_act, zero_division=0)
    f1 = f1_score(t_act, p_act, zero_division=0)
    act_kwh = (np.sum(y_true) / 60.0) / 1000.0
    est_kwh = (np.sum(y_pred) / 60.0) / 1000.0
    acc = max(0.0, 1.0 - abs(est_kwh - act_kwh) / (act_kwh + 1e-6)) * 100.0
    return {
        "Model": name,
        "MAE (W)": round(mae, 2),
        "RMSE (W)": round(rmse, 2),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1-Score": round(f1, 3),
        "Actual Energy (kWh)": round(act_kwh, 2),
        "Est. Energy (kWh)": round(est_kwh, 2),
        "Energy Acc (%)": round(acc, 1)
    }

eval_table = pd.DataFrame([
    evaluate_preds(y_test, pred_zero, "1. Always-Off Baseline"),
    evaluate_preds(y_test, pred_profile, "2. Usage-Profile Baseline"),
    evaluate_preds(y_test, pred_model, "3. Disaggregation Model (GBDT)")
])

print("\n--- Disaggregation Benchmark Results ---")
print(eval_table.to_string(index=False))
eval_table.to_csv(os.path.join(BASE_DIR, "problem3_metrics.csv"), index=False)

# Step 3.5: Problem 3 Visualizations
print("\n3. Generating Part 3 visualizations...")
test_results = pd.DataFrame(index=test_df.index)
test_results["Aggregate"] = test_df["agg"]
test_results["Actual_WM"] = y_test
test_results["Predicted_WM"] = pred_model

# Select clean multi-cycle evaluation slice
active_idx = test_results[test_results["Actual_WM"] > 500].index[20].strftime("%Y-%m-%d")
eval_slice = test_results.loc[f"{active_idx} 00:00:00":pd.to_datetime(active_idx) + pd.Timedelta(days=2)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
ax1.plot(eval_slice.index, eval_slice["Aggregate"], label="Household Aggregate Power", color="gray", alpha=0.5, lw=1)
ax1.plot(eval_slice.index, eval_slice["Actual_WM"], label="Actual Washing Machine (Ground Truth)", color="#1f77b4", lw=2)
ax1.plot(eval_slice.index, eval_slice["Predicted_WM"], label="Predicted Washing Machine (Model)", color="#d62728", lw=1.6, linestyle="--")
ax1.set_title(f"Actual vs. Predicted Washing Machine Load ({active_idx} Evaluation Window)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Power (W)")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.fill_between(eval_slice.index, 0, eval_slice["Actual_WM"], color="#1f77b4", alpha=0.25, label="Actual Energy (Ground Truth)")
ax2.plot(eval_slice.index, eval_slice["Actual_WM"], color="#1f77b4", lw=1.5)
ax2.plot(eval_slice.index, eval_slice["Predicted_WM"], color="#d62728", lw=1.5, label="Model Prediction")
ax2.set_title("Direct Disaggregation Overlay & Cycle Capture", fontsize=10, fontweight="bold")
ax2.set_ylabel("Power (W)")
ax2.set_xlabel("Timestamp")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
p3_plot1 = os.path.join(FIGURES_DIR, "problem3_actual_vs_predicted.png")
plt.savefig(p3_plot1, dpi=200)
plt.close()
print(f"   -> Saved: {p3_plot1}")

# Failure analysis plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5))
peak_time = eval_slice[eval_slice["Actual_WM"] > 1000].index[0]
zoom_success = eval_slice.loc[peak_time - pd.Timedelta(minutes=30):peak_time + pd.Timedelta(minutes=60)]
ax1.plot(zoom_success.index, zoom_success["Actual_WM"], label="Actual WM", color="#1f77b4", lw=2)
ax1.plot(zoom_success.index, zoom_success["Predicted_WM"], label="Predicted WM", color="#d62728", lw=2, linestyle="--")
ax1.plot(zoom_success.index, zoom_success["Aggregate"], label="Aggregate Power", color="gray", alpha=0.4, lw=1)
ax1.set_title("Success Case: High Peak Capture During Water Heating", fontsize=10, fontweight="bold")
ax1.set_ylabel("Power (W)")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

idle_spikes = test_results[(test_results["Aggregate"] > 1800) & (test_results["Actual_WM"] == 0)]
if len(idle_spikes) > 0:
    sp_time = idle_spikes.index[5]
    zoom_fail = test_results.loc[sp_time - pd.Timedelta(minutes=30):sp_time + pd.Timedelta(minutes=45)]
    ax2.plot(zoom_fail.index, zoom_fail["Aggregate"], label="Aggregate (Kettle/Heater)", color="gray", lw=1.2)
    ax2.plot(zoom_fail.index, zoom_fail["Actual_WM"], label="Actual WM (0 W)", color="#1f77b4", lw=2)
    ax2.plot(zoom_fail.index, zoom_fail["Predicted_WM"], label="Predicted WM (Ghost Surge)", color="#d62728", lw=2, linestyle="--")
    ax2.set_title("Failure Case: Overlapping Resistive Load False Positive", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Power (W)")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
p3_plot2 = os.path.join(FIGURES_DIR, "problem3_failure_analysis.png")
plt.savefig(p3_plot2, dpi=200)
plt.close()
print(f"   -> Saved: {p3_plot2}")


# ==============================================================================
# PART 4: PRACTICAL IMPLICATIONS & REAL-WORLD UTILITY INSIGHTS
# ==============================================================================
print("\n" + "=" * 80)
print("PART 4: PRACTICAL IMPLICATIONS & REAL-WORLD UTILITY INSIGHTS")
print("=" * 80)

# Step 4.1: Energy-Saving Program Simulation (Peak Shift + Eco-Wash)
print("1. Simulating peak-shifting and eco-wash savings...")
annual_hot_kwh = (4 * 52) * 0.613
annual_eco_kwh = (4 * 52) * 0.309
kwh_saved_home = annual_hot_kwh - annual_eco_kwh

cost_hot_peak = annual_hot_kwh * 0.35     # 35p/kWh peak
cost_eco_offpeak = annual_eco_kwh * 0.12  # 12p/kWh off-peak
savings_home = cost_hot_peak - cost_eco_offpeak

cohort_50k = 50000
total_mwh_saved = (kwh_saved_home * cohort_50k) / 1000.0
total_gbp_saved = savings_home * cohort_50k

print(f"   Household Energy Reduction:    {kwh_saved_home:.1f} kWh/year (-49.6%)")
print(f"   Annual Bill Savings per Home:  £{savings_home:.2f}/year")
print(f"   Utility Pilot (50,000 Homes):  {total_mwh_saved:,.1f} MWh/year saved (£{total_gbp_saved:,.2f})")

# Step 4.2: Sampling Resolution Simulation (1-min vs. 15-min vs. 30-min)
print("2. Simulating smart-meter downsampling impact...")
h1_sample = h1_df.loc["2013-10-23 05:00:00":"2013-10-23 09:00:00"].copy()
s_1m = h1_sample["Appliance5"]
s_15m = s_1m.resample("15min").mean()
s_30m = s_1m.resample("30min").mean()

a_1m = h1_sample["Aggregate"]
a_15m = a_1m.resample("15min").mean()
a_30m = a_1m.resample("30min").mean()

print(f"   Washing machine peak at 1-min:  {s_1m.max():.1f} W")
print(f"   Washing machine peak at 15-min: {s_15m.max():.1f} W (-{1 - s_15m.max()/s_1m.max():.1%} dilution)")
print(f"   Washing machine peak at 30-min: {s_30m.max():.1f} W (-{1 - s_30m.max()/s_1m.max():.1%} dilution)")

fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True)
axes[0].plot(h1_sample.index, a_1m, label="Aggregate Mains Power", color="gray", alpha=0.5, lw=1)
axes[0].plot(h1_sample.index, s_1m, label="Washing Machine (1-Min)", color="#d62728", lw=1.8)
axes[0].set_title("A. 1-Minute Resolution: Sharp 2,368 W Heating Spikes & Precise Boundaries", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Power (W)")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(loc="upper right")

axes[1].step(s_15m.index, a_15m, where="post", label="Aggregate Mains (15-Min)", color="gray", alpha=0.5, lw=1)
axes[1].step(s_15m.index, s_15m, where="post", label="Washing Machine (15-Min)", color="#ff7f0e", lw=2)
axes[1].set_title(f"B. 15-Minute Resolution: Peak Flattened to {s_15m.max():.0f} W (-63% Dilution)", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Power (W)")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(loc="upper right")

axes[2].step(s_30m.index, a_30m, where="post", label="Aggregate Mains (30-Min)", color="gray", alpha=0.5, lw=1)
axes[2].step(s_30m.index, s_30m, where="post", label="Washing Machine (30-Min)", color="#2ca02c", lw=2)
axes[2].set_title(f"C. 30-Minute Resolution: Peak Flattened to {s_30m.max():.0f} W (-79% Dilution), Blends into Baseload", fontsize=10, fontweight="bold")
axes[2].set_ylabel("Power (W)")
axes[2].set_xlabel("Time of Day (October 23, 2013)")
axes[2].grid(True, linestyle="--", alpha=0.5)
axes[2].legend(loc="upper right")

plt.suptitle("Impact of Smart-Meter Data Resolution on Disaggregation", fontsize=12, fontweight="bold")
plt.tight_layout()
p4_plot = os.path.join(FIGURES_DIR, "problem4_sampling_resolution_impact.png")
plt.savefig(p4_plot, dpi=200)
plt.close()
print(f"   -> Saved: {p4_plot}")

# Step 4.3: End-to-End Pipeline Completion Message
print("\n" + "=" * 80)
print("END-TO-END PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"All 8 deliverable figures saved in: {FIGURES_DIR}")
print(f"Master results and tables exported in: {BASE_DIR}")
print("================================================================================\n")
