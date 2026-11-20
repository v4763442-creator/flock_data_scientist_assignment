# Flock Energy Take-Home Assignment: REFIT Data Cleaning & Appliance Disaggregation

**Candidate Submission**  
**Dataset:** REFIT Electrical Load Measurements (University of Strathclyde, Loughborough, UEA)  
**Repository:** [flock_data_scientist_assignment](https://github.com/v4763442-creator/flock_data_scientist_assignment)

---

## Welcome & Project Overview

Hi! In this repository, I have built an end-to-end, reproducible data science pipeline to solve the challenge of **smart-meter data cleaning and non-intrusive appliance disaggregation (NILM)** using the open-source REFIT dataset.

The central problem utilities like Flock Energy face is that smart meters only record the **total active power** entering a house. We don't have dedicated sub-meters on every appliance in every home. My goal in this project was to:
1. **Clean raw, noisy, high-frequency smart-meter data** (House 1) and resample it to uniform 1-minute intervals without inventing fake data during power outages.
2. **Analyze washing machine behavior across multiple homes** (Houses 1, 2, 3, and 5) to understand how people actually do laundry (cycle duration, peak power, energy consumed, and times of day).
3. **Build an appliance disaggregation model** that estimates washing machine power consumption directly from whole-home aggregate power, and evaluate it fairly against realistic baselines.
4. **Translate my findings into practical utility recommendations**—including energy-saving programs, the real-world impact of 15/30-minute smart meters, and why UK-trained models cannot simply be copy-pasted to Indian households.

I intentionally wrote all the code in a clean, self-contained, and pragmatic style. Rather than creating convoluted abstractions or splitting simple functions across multiple confusing helper files, each part is fully readable, transparent, and easy to run.

---

## Repository Structure

Inside this repository, you will find the **master source code deliverables** (`SOURCE_CODE.ipynb` and `SOURCE_CODE.py`), detailed markdown reports, and all generated figures:

```
.
├── README.md                           # Master project guide, setup, validation design & trade-offs
├── DATA.md                             # Detailed raw data audit, quality findings & cleaning rules
├── RESULTS.md                          # Full analysis of EDA, model metrics, and failure modes
├── Recommendation.md                   # One-page utility-facing summary & Section 4 answers
│
├── SOURCE_CODE.ipynb                   # Master interactive notebook with all 4 problems & inline figures
├── SOURCE_CODE.py                      # Master end-to-end Python script (Runs all 4 parts in ~60s)
│
├── figures/                            # All 8 generated high-resolution plots
│   ├── problem1_before_after_cleaning.png
│   ├── problem1_outage_spike_handling.png
│   ├── problem2_24h_trace.png
│   ├── problem2_detected_cycles.png
│   ├── problem2_cross_house_comparison.png
│   ├── problem3_actual_vs_predicted.png
│   ├── problem3_failure_analysis.png
│   └── problem4_sampling_resolution_impact.png
│
├── house1_cleaned_1min.parquet         # Cleaned 1-minute House 1 dataset (ready-to-use)
└── .gitignore                          # Git ignore configuration
```

---

## How to Set Up and Run the Code

### 1. Requirements
I kept dependencies minimal and standard. You only need:
- Python 3.10+ (tested on Python 3.13)
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `pyarrow` (for fast data reading).

To install them:
```bash
pip install pandas numpy matplotlib scikit-learn pyarrow
```

### 2. Running Everything in One Command
If you just want to run the whole pipeline from beginning to end, run:
```bash
python SOURCE_CODE.py
```
This script executes all 4 parts, prints clean summary tables in your terminal, and saves all 8 figures into the `figures/` folder. It takes roughly **60 to 75 seconds** on a standard laptop.

You can also open [`SOURCE_CODE.ipynb`](SOURCE_CODE.ipynb) in JupyterLab or VS Code to step through each code cell and view the charts inline.

---

## Detailed Walkthrough of Preprocessing Decisions & Assumptions

Here is what I encountered in the data and why I made specific design choices:

### 1. Merging Part 1 and Part 2 (House 1)
When I opened the raw REFIT files for House 1, I noticed that `RAW_House1_Part1.csv` and `RAW_House1_Part2.csv` overlap between June and October 2014. According to the dataset documentation:
- In **Part 1**, whenever a sensor was disconnected or unavailable, it was recorded as `0`.
- In **Part 2** (starting June 27, 2014), the university updated its logger so that disconnected sensors were explicitly recorded as `NaN`.

**My decision:** I took Part 1 up to `2014-06-27 21:57:50`, and from that point onward, I used Part 2. This gave me a seamless continuous timeline of **7,260,898 raw records** and allowed me to properly distinguish between an appliance being turned off (0 Watts) versus an appliance sensor being disconnected (`NaN`).

### 2. Chronological Sorting & Deduplication
- Because sensors were polled wirelessly over a Zigbee network every 6–8 seconds, packets sometimes arrived slightly out of chronological order. I sorted the data by timestamp first.
- I found **348,686 duplicate timestamps (4.8%)**.
- **My stated rule:** When multiple readings had the exact same second, I kept the `last` reading (`keep='last'`). The last reading reflects the most recent state transmission from the sensor.

### 3. Removing Impossible Sensor Spikes
- In the UK, standard domestic wall sockets are single-phase 230V rated at 13 Amperes. That means the theoretical maximum continuous power through a wall plug is $230 \text{ V} \times 13 \text{ A} \approx 3,000\text{ W}$ (or ~3,120 W peak).
- In the raw data, I discovered **2,889 appliance readings exceeding 4,000 W** (some reaching tens of thousands of Watts due to wireless bit-flip errors).
- On total household power, I found **168 readings exceeding 15,000 W** (15 kW is an unrealistic continuous load for a standard UK domestic supply).
- **My decision:** I clamped all appliance readings $> 4,000$ W and aggregate readings $> 15,000$ W to `NaN`. I did not find any negative readings (all were $\ge 0$).

### 4. Handling Short Gaps vs. Long Outages
- I resampled the data to uniform **1-minute intervals** using the interval mean.
- **Short gaps ($\le 5$ minutes):** In a wireless sensor network, dropping 1 or 2 polling packets is very common. An appliance does not fundamentally change its state in 60 seconds, so I forward-filled gaps up to 5 minutes (`limit=5`).
- **Long outages (> 5 minutes):** The REFIT data has major network crashes where loggers went down for days (the longest gap was ~998 hours, notably in February 2014).
- **My decision:** I **refused to forward-fill or interpolate long outages**. Doing so would invent hours or days of fake, constant power consumption that never happened! Instead, I kept them as `NaN` and created an explicit `outage_flag = 1` column. In House 1, **12.70% of the timeline** represented true outages and was safely excluded from model training.

---

## Plain-Language Rule for Detecting Washing Machine Cycles

Washing machines are episodic appliances: they sit idle for days, run for an hour, and shut off. A domestic wash cycle involves filling with water, tumbling the drum, heating water with an internal electric element (1,800–2,500 W), rinsing, draining, and spinning.

To detect active cycles automatically across Houses 1, 2, 3, and 5, I defined a simple, physical 4-step rule:
1. **Active Threshold ($P \ge 20$ W):** Distinguishes operational motor/electronics from low standby power ($< 5$ W).
2. **Pause Tolerance ($\le 5$ minutes):** Between wash, rinse, and spin phases, washing machines pause for 1 to 3 minutes while draining or reversing drum direction. If power briefly dips below 20 W for $\le 5$ minutes, my algorithm treats it as part of the *same* continuous cycle.
3. **Minimum Duration ($\ge 15$ minutes):** Filters out accidental door openings, light button presses, or brief test runs.
4. **Minimum Peak Power ($\ge 500$ W):** Confirms that either the water heating element or high-speed spin motor was actively engaged.

### What the Exploratory Analysis Revealed Across 4 Homes:
- **Extreme Inactivity:** Washing machines are turned off **92% to 98% of the time**.
- **House 1 (Beko):** Prefers quick/eco wash cycles (median duration **33 minutes**, median energy **0.31 kWh** per wash).
- **House 2 (LG):** Prefers long, intensive hot washes (median duration **97 minutes**, median energy **0.61 kWh** per wash—almost double the energy!).
- **House 3:** Highest laundry frequency (**7.6 cycles per week**, more than 1 wash every day).
- **Timing:** Washing across all homes peaks twice a day: a primary morning peak (**09:00–12:00**) and an evening post-work peak (**18:00–20:00**). Weekends (Saturday & Sunday) account for ~38% of all laundry.

---

## Validation Design & Preventing Data Leakage

When evaluating machine learning on time-series data, **how you split your data is everything**.

- **Why I avoided random train/test splits:** If you randomly shuffle 1-minute time slices, minutes from the *same* 35-minute washing machine cycle will end up in both the training set and the test set. The model would simply memorize the neighboring minutes of a known cycle, causing severe **data leakage** and unrealistically optimistic scores.
- **My validation choice:** I used a strict **chronological 80/20 train/test split**:
  - **Training Set (first 80%):** October 9, 2013 to February 21, 2015 (~638,000 minutes).
  - **Evaluation Set (last 20%):** February 21, 2015 to July 10, 2015 (~160,000 minutes).
- This strictly evaluates how well the model predicts **future unseen months**, replicating how a utility would deploy the algorithm in the real world.

---

## Model Selection, Baselines & The "NILM Evaluation Paradox"

### Why Gradient Boosted Decision Trees (HistGBM)?
Rather than building an over-engineered deep neural network that takes hours to train and requires specialized GPUs, I selected **Gradient Boosted Decision Trees (`HistGradientBoostingRegressor`)**. 
- It trains in **under 4 seconds**.
- It is robust to outliers and requires zero feature scaling.
- Combined with 7 intuitive domain features (current aggregate power, rate-of-change `diff`, 5-minute rolling mean, 15-minute rolling mean, 15-minute rolling max, hour of day, and day of week), it easily captures the multi-stage signature of a washing machine.

### Benchmarks Against Simple Baselines

| Model | MAE (Watts) | RMSE (Watts) | Active Precision | Active Recall | Active F1 | Actual Energy | Predicted Energy | Energy Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Always-Off Baseline** | **10.62 W** | 135.82 W | 0.000 | 0.000 | 0.000 | 28.25 kWh | 0.00 kWh | 0.0% |
| **2. Hourly Profile Baseline** | 20.44 W | 134.79 W | 0.047 | 0.729 | 0.089 | 28.25 kWh | 28.73 kWh | 98.3% |
| **3. My Disaggregation Model (GBDT)** | **12.32 W** | **99.89 W** | **0.249** | **0.748** | **0.374** | 28.25 kWh | 27.39 kWh | **97.0%** |

### The "NILM Evaluation Paradox" (Why MAE is Misleading)
Notice that the **Always-Off Baseline** has an MAE of only 10.62 Watts! 
- *Does this mean predicting zero is a great model?* **No!** Because the washing machine is off >98% of the time, predicting zero gives you zero error for 98% of the minutes. But that model detects **zero cycles** (Recall = 0.0).
- **Why RMSE is necessary:** RMSE squares the error, heavily punishing misses during the 2,000 W water heating phase. My GBDT model reduces RMSE from 135.82 W down to 99.89 W (**a 26.4% reduction**).
- **High Recall:** The model detects **74.8% of all active washing minutes** in unseen future data.
- **Energy Accuracy:** Over 4 months of testing, the true laundry energy was **28.25 kWh**; my model predicted **27.39 kWh** (**97.0% total energy estimation accuracy**).

---

## Known Failure Modes & Practical Limitations

No model is perfect, and I made sure to clearly document where the model struggles:

1. **Overlapping Resistive Heaters (False Positives):**
   In UK homes, electric kettles, toasters, and portable space heaters also draw 2,000–3,000 Watts in sudden square pulses. In active power alone, a 3-minute kettle boil looks identical to a brief water-heating surge, causing occasional false positives (Precision = ~25%).
2. **Quiet Agitation & Drain Phases (Under-estimation):**
   During the low-power tumble and draining stages (50–150 W), the washing machine signal is buried inside the home's background noise (refrigerators, entertainment centers). The model tends to under-predict during these quiet inter-stage intervals.
3. **Why 15-Minute or 30-Minute Smart Meters Make Disaggregation Harder:**
   In [Problem 4](Problem4_Practical_Implications.ipynb), I downsampled House 1 data:
   - At **1-minute**, the heating spike reaches a sharp **2,368 W**.
   - At **15-minute**, the peak is diluted down to **876 W** (-63% reduction).
   - At **30-minute**, the peak is flattened to **503 W** (-79% reduction).
   At 30 minutes, an entire 33-minute wash cycle collapses into a single blurry data point that blends directly into everyday refrigerator and computer baseload.
4. **Why UK Models Do Not Transfer Directly to India:**
   - **No Internal Heaters:** UK machines are front-load units with 2,000–2,500 W internal water heaters. In India, most domestic machines are cold-water fill top-load units drawing only **200–500 Watts** of pure motor load.
   - A model trained on UK 2 kW heating signatures will completely fail in typical Indian households.

---

---

## Possible Improvements if I Had More Time

1. **Reactive Power (VAR) Integration:** If smart meters record reactive power, we can measure phase shifts ($\cos \phi$) to instantly separate inductive electric motors from purely resistive kettles.
2. **Sequence-to-Point (Seq2Point) 1D-CNN:** Training a 1D convolutional neural network with a 60-minute sliding window would better capture the multi-stage narrative of wash $\rightarrow$ rinse $\rightarrow$ spin.
3. **Cross-Household Training:** Training on a combined multi-house dataset to evaluate how well the model generalizes to homes it has never seen before.
