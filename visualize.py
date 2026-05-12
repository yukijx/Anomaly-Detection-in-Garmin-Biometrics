"""
script: visualize.py 

Description: Generates graphs from datasets and OC-SVM results

Note: This script expects the following datasets and trained models to be available 
      from previous scripts to output figures (time-series plots, scatter plots, 
      correlation heatmaps, and OC-SVM visuals). 

      Expects:
        ./datasets/dataset_A_cardiovascular.csv
        ./datasets/dataset_B_stress_recovery.csv
        ./datasets/dataset_C_sleep.csv

        ./models/ocsvm_dataset_A_trained.joblib  
        ./models/ocsvm_dataset_B_trained.joblib  
        ./models/ocsvm_dataset_C_trained.joblib  

        ./models/meta_dataset_A_trained.json
        ./models/meta_dataset_B_trained.json
        ./models/meta_dataset_C_trained.json
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
warnings.filterwarnings("ignore")
import matplotlib.dates as mdates

DATASET_DIR = "./datasets"
MODEL_DIR   = "./models"
OUT_DIR     = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

""" COLORS """
BG          = "#FFFFFF"
CARD        = "#FFFFFF"
BLUE        = "#2563EB"
RED         = "#DC2626"
GRAY        = "#6B7280"
DARK        = "#111827"
LIGHT       = "#111827"
LIGHT_GRAY  = "#6B7280"
BORDER      = "#E5E7EB"
TEAL        = "#2563EB"   
MAROON      = "#800000"   


# MATPLOTLIB STYLE SETTINGS FOR CONSISTENT PLOT STYLE
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": BORDER, "axes.labelcolor": DARK,
    "xtick.color": GRAY, "ytick.color": GRAY,
    "text.color": DARK, "grid.color": BORDER,
    "grid.linewidth": 0.8, "font.family": "sans-serif", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})


"""
SAVE FIGURE TO OUTPUT_DIR

parameters: fig object to save, name of the file to save as
""" 
def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"SAVED: {name}")


"""
LOAD DATASET CSV INTO DATAFRAME

parameters: key (dataset key "A", "B", or "C")
returns: dataframe with date as index, features as columns, sorted by date
"""
def load(key):
    names = {
        "A": "dataset_A_cardiovascular.csv",
        "B": "dataset_B_stress_recovery.csv",
        "C": "dataset_C_sleep.csv",
    }
    
    path = os.path.join(DATASET_DIR, names[key])
    if not os.path.exists(path):
        print(f"SKIP: {path} NOT FOUND.")
        return None
    return pd.read_csv(path, index_col="date", parse_dates=True)

A = load("A")
B = load("B")
C = load("C")


"""
TIME-SERIES PLOT PER METRIC (with mean ±1 standard deviation ribbon) 

parameters: series to plot, title, y-axis label, line color, output filename
NOTE: only plots if there are at least 3 data points to show trend
"""
def plot_timeseries(series, title, ylabel, color, fname):
    
    s = series.dropna().sort_index()
    if len(s) < 3:
        return
    mu, sigma = s.mean(), s.std()


    fig, ax = plt.subplots(figsize=(11, 4))

    # DATA LINE
    ax.plot(s.index, s.values, color=color, lw=2, zorder=3)

    # OUTLIER SHADING (beyond 1.5 standard deviations)
    ax.fill_between(s.index, s.values, mu, alpha=0.12, color=color)

    # MEAN LINE
    ax.axhline(mu, color=LIGHT_GRAY, lw=1.2, linestyle="-",
               label=f"Mean: {mu:.1f}")
    
    # RIBBON FOR ±1 STANDARD DEVIATION
    ax.fill_between(s.index, mu - sigma, mu + sigma,
                    color=color, alpha=0.10,
                    label=f"±1 SD ({mu-sigma:.1f} - {mu+sigma:.1f})")

    # OUTLIERS BEYOND 1.5 STANDARD DEVIATIONS 
    outliers = s[np.abs(s - mu) > 1.5 * sigma]

    # PLOT OUTLIERS AS SCATTER POINTS
    if not outliers.empty:
        ax.scatter(outliers.index, outliers.values, color=MAROON,
                   zorder=5, s=60, label="Outliers (>1.5 SD)")

    # X-AXIS; one tick per week, {Month Day} format 
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(s.index.min() - pd.Timedelta(days=1),
                s.index.max() + pd.Timedelta(days=1))

    # TOTAL DAYS SUBTITLE 
    n_days = (s.index.max() - s.index.min()).days + 1
    ax.set_xlabel(f"{n_days} days  ({s.index.min().strftime('%b %d')} – {s.index.max().strftime('%b %d, %Y')})",
                  fontsize=9, color=LIGHT_GRAY)

    # AXIS LABELS, TITLE, GRID, LEGEND
    ax.set_title(title, color=LIGHT, fontsize=13, fontweight="bold", pad=10) # title w. padding
    ax.set_ylabel(ylabel, fontsize=10) # y-axis label w. smaller font
    ax.tick_params(axis="x", rotation=30) # rotate x-axis labels for readability
    ax.grid(True, axis="y", alpha=0.5) # horizontal grid lines with lighter color
    ax.legend(framealpha=0, labelcolor=LIGHT, fontsize=9, loc="upper right") # legend with transparent background and smaller font
    fig.tight_layout() # adjust layout to prevent clipping labels and title
    save(fig, fname) 

print("\n")
print("=" * 55)
print("TIME SERIES PLOTS FOR METRICS ")
print("=" * 55)

# DATASET A 
if A is not None:
    plot_timeseries(A["respiration_avg"], "Avg. Respiration Rate (During Sleep)",
                    "Breaths / min",    TEAL,                   "1_respiration.png")

# DATASET B
if B is not None:
    plot_timeseries(B["hrv_ms"],            "HRV 50-Day Baseline",
                    "HRV ms",               TEAL,              "2_hrv.png")
    plot_timeseries(B["avg_stress"],        "Average Daily Stress Score",
                    "Stress (0-100)",       MAROON,            "3_stress.png")
    plot_timeseries(B["body_battery_eod"],  "End-of-Day Body Battery",
                    "Body Battery",         "#38BDF8",       "4_body_battery.png")
    plot_timeseries(B["steps"],             "Daily Steps",
                    "Steps",                TEAL,              "5_steps.png")
    plot_timeseries(B["sleep_hours"],       "Sleep Duration",
                    "Hours",                "#818CF8",       "6_sleep_hours.png")

# DATASET C
if C is not None:
    plot_timeseries(C["sleep_hours"],       "Sleep Duration",
                    "Hours",                "#818CF8",       "7_sleep_hours.png")
    plot_timeseries(C["deep_pct"],          "Deep Sleep %",
                    "% of total sleep",     TEAL,              "8_deep_pct.png")
    plot_timeseries(C["rem_pct"],           "REM Sleep %",
                    "% of total sleep",     "#7C3AED",       "9_rem_pct.png")


"""
SLEEP STAGE BREAKDOWN STACKED BAR CHART % of deep/REM/light/awake/sleep per night

NOTE: not sure if I really need this plot
"""
print("\n")
print("=" * 55)
print("SLEEP STAGES ")
print("=" * 55)

if C is not None:
    cols   = ["deep_pct", "rem_pct", "light_pct", "awake_pct"]
    labels = ["Deep", "REM", "Light", "Awake"]
    colors = [BLUE, "#7C3AED", LIGHT_GRAY, MAROON]
    
    # only plot if at least 2 of the sleep stage percentages are present for night 
    df = C[cols].dropna(thresh=2).sort_index() 

    # if there are fewer than 3 nights with sleep stage data, skip plot 
    if not df.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        bottom = np.zeros(len(df))
        for col, label, color in zip(cols, labels, colors):
            vals = df[col].fillna(0).values
            ax.bar(df.index, vals, bottom=bottom, label=label,
                   color=color, width=0.8, alpha=0.9)
            bottom += vals


        # X-AXIS
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.set_xlim(df.index.min() - pd.Timedelta(days=1),
                    df.index.max() + pd.Timedelta(days=1))
        ax.set_ylim(0, 125)

        # TOTAL DAYS SUBTITLE
        n_days = (df.index.max() - df.index.min()).days + 1
        ax.set_xlabel(f"{n_days} days  ({df.index.min().strftime('%b %d')} – {df.index.max().strftime('%b %d, %Y')})",
                      fontsize=9, color=LIGHT_GRAY)
        

        ax.set_title("Sleep Stage Composition per Night",
                     color=LIGHT, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Ratio of total sleep", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(framealpha=0, labelcolor=LIGHT, fontsize=9,
                  loc="upper right", ncol=4)
        ax.set_ylim(0, 115)
        fig.tight_layout()
        save(fig, "10_sleep_stages.png")



"""
SCATTER PLOTS: 
- stress vs body battery 
- steps vs sleep duration
- stress vs sleep duration
"""
print("\n")
print("=" * 55)
print("SCATTER PLOTS ")
print("=" * 55)

def scatter_with_trend(df, xcol, ycol, xlabel, ylabel, title, fname):
    d = df[[xcol, ycol]].dropna()
    if len(d) < 5:
        return
    corr = d[xcol].corr(d[ycol])
    z = np.polyfit(d[xcol], d[ycol], 1)
    xs = np.linspace(d[xcol].min(), d[xcol].max(), 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d[xcol], d[ycol], color=TEAL, s=70, alpha=0.85,
               edgecolors=BORDER, linewidths=0.5, zorder=3)
    ax.plot(xs, np.poly1d(z)(xs), color=LIGHT_GRAY, lw=1.5,
            linestyle="--", alpha=0.7, label=f"r = {corr:.2f}")
    
    ax.set_title(title, color=LIGHT, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.legend(framealpha=0, labelcolor=LIGHT, fontsize=10)
    fig.tight_layout()
    save(fig, fname)

if B is not None:
    scatter_with_trend(B, "avg_stress", "body_battery_eod",
                       "Avg Daily Stress", "Body Battery (EOD)",
                       "Stress vs Body Battery", "11_stress_vs_battery.png")
    scatter_with_trend(B, "steps", "sleep_hours",
                       "Daily Steps", "Sleep Hours",
                       "Steps vs Sleep Duration", "12_steps_vs_sleep.png")
    scatter_with_trend(B, "avg_stress", "sleep_hours",
                       "Avg Daily Stress", "Sleep Hours",
                       "Stress vs Sleep Duration", "13_stress_vs_sleep.png")



"""
FEATURE CORRELATION FIGURE FOR DATASET B (STRESS & RECOVERY)
"""
print("\n")
print("=" * 55)
print("SCATTER PLOTS ")
print("=" * 55)

# only plot if there are at least 4 features with some data to correlate 
if B is not None:
    corr = B.dropna(thresh=4).corr()
    n = len(corr)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    nice = [c.replace("_", "\n") for c in corr.columns]
    ax.set_xticks(range(n)); ax.set_xticklabels(nice, fontsize=8, color=LIGHT)
    ax.set_yticks(range(n)); ax.set_yticklabels(nice, fontsize=8, color=LIGHT)
    ax.tick_params(length=0)

    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.5 else LIGHT)

    ax.set_title("Feature Correlation: Dataset B (Stress & Recovery)",
                 color=LIGHT, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, "14_correlation.png")


"""
OC-SVM OUTPUT
"""
print("\n")
print("=" * 55)
print("OC-SVM OUTPUT")
print("=" * 55)

# FOR EACH DATASET, LOAD TRAINED OC-SVM MODEL AND META, 
# ALIGN FEATURES & PREDICT ANOMALIES, 
# PLOT DECISION SCORES OVER TIME 
for key, df in [("A", A), ("B", B), ("C", C)]:
    model_path = os.path.join(MODEL_DIR, f"ocsvm_dataset_{key}_trained.joblib")
    meta_path  = os.path.join(MODEL_DIR, f"meta_dataset_{key}_trained.json")

    if not os.path.exists(model_path):
        print(f"No trained model for dataset {key} found. Run predict.py first.")
        continue
    if df is None:
        continue

    pipeline = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    X = df[[c for c in feature_cols if c in df.columns]].fillna(df.median())


    # decision_function() gives distance from boundary (negative = more anomalous) 
    # predict() gives +1 normal, -1 anomaly
    scores = pipeline.decision_function(X)
    preds  = pipeline.predict(X)          

    normal_mask  = preds ==  1
    anomaly_mask = preds == -1

    # DESCISION SCORE OVER TIME
    fig, ax = plt.subplots(figsize=(11, 4))

    ax.scatter(X.index[normal_mask], scores[normal_mask],
               color=TEAL, s=55, zorder=3,
               label=f"Normal ({normal_mask.sum()} days)")
    ax.scatter(X.index[anomaly_mask], scores[anomaly_mask],
               color=MAROON, s=80, marker="X", zorder=4,
               label=f"Flagged ({anomaly_mask.sum()} days, "
                     f"{anomaly_mask.mean()*100:.0f}%)")
    ax.fill_between(X.index, scores.min() - 0.02, 0,
                    color=MAROON, alpha=0.06)

    # X-AXIS 
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(X.index.min() - pd.Timedelta(days=1),
                X.index.max() + pd.Timedelta(days=1))

    n_days = (X.index.max() - X.index.min()).days + 1
    ax.set_xlabel(f"{n_days} days  ({X.index.min().strftime('%b %d')} - {X.index.max().strftime('%b %d, %Y')})",
                  fontsize=9, color=LIGHT_GRAY)

    ax.set_title(f"OC-SVM Decision Scores Over Time: Dataset {key} ({meta['dataset_name']})",
                 color=LIGHT, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Decision Score\n(negative = anomalous)", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(framealpha=0, labelcolor=LIGHT, fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, f"15_ocsvm_scores_dataset{key}.png")



    # IMPERSONATOR VS BASELINE SCORE COMPARISON SCATTER PLOT
    IMPERSONATOR_DATES = {"2026-05-05", "2026-05-06", "2026-05-07"}

    date_strs = X.index.strftime("%Y-%m-%d")
    imp_mask  = date_strs.isin(IMPERSONATOR_DATES)
    base_mask = ~imp_mask

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # BASELINE SCORES AS A BOX/STRIP WITH JITTER
    ax.scatter(
        scores[base_mask],
        np.random.uniform(0.6, 1.4, base_mask.sum()),  # jitter so points don't stack
        color=TEAL, s=50, alpha=0.7, zorder=3, label="My days (baseline)"
    )

    # IMPERSONATE SCORES
    imp_labels = {
        "2026-05-05": "Impersonator A",
        "2026-05-06": "Both (overlap)",
        "2026-05-07": "Impersonator B",
    }
    imp_colors = {
        "2026-05-05": MAROON,
        "2026-05-06": "#F59E0B",
        "2026-05-07": "#7C3AED",
    }
    plotted = set()
    for date_str, row_score in zip(date_strs[imp_mask], scores[imp_mask]):
        lbl = imp_labels.get(date_str, date_str)
        col = imp_colors.get(date_str, MAROON)
        ax.scatter(row_score, 1.0, color=col, s=150, zorder=5, marker="D",
                   label=lbl if lbl not in plotted else "_nolegend_")
        ax.annotate(lbl, (row_score, 1.0),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color=col)
        plotted.add(lbl)

    ax.axvline(0, color=DARK, lw=1.5, linestyle="-", label="Decision boundary (0)")
    ax.set_xlabel("Decision Score  (negative = anomalous)", fontsize=10)
    ax.set_title(f"Baseline vs Impersonator Scores: Dataset {key} ({meta['dataset_name']})",
                 color=DARK, fontsize=13, fontweight="bold", pad=10)
    ax.set_yticks([])
    ax.set_ylim(0, 2)
    ax.legend(framealpha=0, fontsize=9)
    ax.grid(True, axis="x", alpha=0.4)
    fig.tight_layout()
    save(fig, f"16_impersonator_scores_dataset{key}.png")


print(f"\nDONE. ALL FIGURES SAVED TO {OUT_DIR}/")