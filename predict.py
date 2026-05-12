"""
script: predict.py 

Desciption: Splits datasets A, B, and C into train/test sets, 
            trains OC-SVM on training days only, then scores the test set
            with the impersonator days labeled separately. 

Note. 
As of 05/09/2026, there are about 62 days of data total, though each dataset has missing/different number of days.
First 50 days of baseline data are used for training, and the remaining days & impersonator A and B's data are used for testing 

Impersonation dates:
        Person A      : 2026-05-05 & 2026-05-06 (person A wore watch May 5 day, returned May 6 after overnight wear)
        Person B      : 2026-05-06 & 2026-05-07 (person B wore watch May 6 day, returned May 7 after overnight wear)
        Me (baseline) : All other recorded days 
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import date
from sklearn import pipeline
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
warnings.filterwarnings("ignore")

"""
Global constants and parameters; Adjustable impersonator dates and training days
Dataset keys; directory paths; OC-SVM parameters (nu, kernel, gamma)
"""
DATASET_DIR = "./datasets"
MODEL_DIR   = "./models"
RESULTS_DIR = "./results"
TRAIN_DAYS  = 50  

IMPERSONATOR_A = {"2026-05-05", "2026-05-06"}
IMPERSONATOR_B = {"2026-05-06", "2026-05-07"}
ALL_IMPERSONATOR_DATES = IMPERSONATOR_A | IMPERSONATOR_B

DATASETS = {
    "A": "dataset_A_cardiovascular.csv",
    "B": "dataset_B_stress_recovery.csv",
    "C": "dataset_C_sleep.csv",
}

DATASET_NAMES = {
    "A": "Cardiovascular",
    "B": "Stress & Recovery",
    "C": "Sleep",
}

OCSVM_PARAMS = {
    "nu":     0.05, # 95% of own data as normal and up to 5% as anomalies
    "kernel": "rbf", 
    "gamma":  "scale",
}

os.makedirs(MODEL_DIR,   exist_ok=True) 
os.makedirs(RESULTS_DIR, exist_ok=True)


"""
Helper functions: 
* Load dataset 
* Align features to match training columns, add missing columns as NaN
* Replace missing values with median
* Label rows as "baseline", "impersonator_A", "impersonator_B", or "impersonator_both" based on date
"""

"""
paramters: key (dataset key "A", "B", or "C")
returns: dataframe with date as index, features as columns, sorted by date
"""
def load(key):
    path = os.path.join(DATASET_DIR, DATASETS[key])
    
    if not os.path.exists(path):
        print(f"  [SKIP] {path} NOT FOUND.")
        return None
    
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df.sort_index()
    return df

"""
parameter: dataframe
returns: dataframe with missing values filled with median of each column
"""
def handle_median(df):
    return df.fillna(df.median())

"""
parameter: date string in "YYYY-MM-DD" format
returns: label for that date, "baseline", "impersonator_A", "impersonator_B", or "impersonator_both" 
"""
def label_row(date_str):

    # TODO: should change this instead of hardcoding dates, should work for now
    # probably not best practice, but I'm overlapping days because I don't remember when exactly I had the switch-off...
    if date_str in IMPERSONATOR_A and date_str in IMPERSONATOR_B:
        return "impersonator_both" # overlap day (2026-05-06) when both impersonators wore the watch, this part is pretty much hardcoded 
    
    elif date_str in IMPERSONATOR_A:
        return "impersonator_A"
    
    elif date_str in IMPERSONATOR_B:
        return "impersonator_B"
    
    else:
        return "baseline"

"""
Core functionality (for each dataset A, B, C):
- Load dataset, split into training set (first 50 days of baseline data) and test set
- Train OC-SVM on training set only
- Score test set, label impersonator days separately
- Print summary of results and save full test set with scores to ./results/
"""
"""
parameter: dataset key "A", "B", or "C"
returns: dataframe with test set results, including anomaly scores and labels
"""
# TODO: should probably split this into smaller functions... if I have time 
def run(key):

    print(f"\n{'='*55}")
    print(f"DATASET {key}: {DATASET_NAMES[key]}")
    print(f"{'='*55}")

    df = load(key)
    if df is None:
        return

    """
    Splitting dataset to training (baseline) and testing data
    First 50 days (TRAIN_DATS) will be used for baseline data 
    Impersonator data excluded from baseline 
    """

    # BASELINE
    df_baseline = df[~df.index.strftime("%Y-%m-%d").isin(ALL_IMPERSONATOR_DATES)]
    df_train = df_baseline.iloc[:TRAIN_DAYS]
    df_test_mine = df_baseline.iloc[TRAIN_DAYS:]

    # IMPERSONATORS
    imp_dates = [d for d in ALL_IMPERSONATOR_DATES if d in df.index.strftime("%Y-%m-%d")]
    df_imp = df[df.index.strftime("%Y-%m-%d").isin(imp_dates)]

    # TEST SET = "NORMAL" DATA + IMPERSONATOR DATA
    df_test = pd.concat([df_test_mine, df_imp]).sort_index()

    """
    PRINT SUMMARY OF RESULTS IN TERMINAL
    """
    # TOTAL DAYS IN DATASET 
    print(f"Total days in dataset: {len(df)}")

    # TOTAL DAYS USED IN TRAINING SET (BASELINE) -- CHECK 50 DAYS 
    print(f"Training days (baseline): {len(df_train)}  " f"({df_train.index.min().date()} to {df_train.index.max().date()})")
    
    # TOTAL DAYS IN TEST SET (NORMAL + IMPERSONATORS)
    print(f"Test days (normal): {len(df_test_mine)}")
    print(f"Impersonator days: {len(df_imp)}  {imp_dates}")

    
    """
    TRAIN OC-SVM on training set only 
    """
    print(f"\n BEGIN TRAINING...")
    feature_cols = df.columns.tolist()
    x_train = handle_median(df_train[feature_cols]) #replace missing values with median

    # BUILD PIPELINE WITH STANDARD SCALER AND OC-SVM
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm",  OneClassSVM(**OCSVM_PARAMS)),
    ])
    pipeline.fit(x_train)

    # CHECK HOW MANY DAYS ARE FLAGGED AS ANOMALIES IN TRAINING SET 
    # note: should be around 5% based on nu parameter :( 
    train_preds   = pipeline.predict(x_train) # +1 = normal, -1 = anomaly
    train_flagged = (train_preds == -1).sum()

    print(f"\n TRAINING COMPLETED.")
    print(f"Flagged on own training data: {train_flagged}/{len(x_train)} "
          f"({train_flagged/len(x_train)*100:.1f}%)") # ratio and percentage 

    # SAVE MODEL 
    model_path = os.path.join(MODEL_DIR, f"ocsvm_dataset_{key}_trained.joblib")
    meta_path  = os.path.join(MODEL_DIR, f"meta_dataset_{key}_trained.json")
    joblib.dump(pipeline, model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "dataset_key":       key,
            "dataset_name":      DATASET_NAMES[key],
            "feature_cols":      feature_cols,
            "ocsvm_params":      OCSVM_PARAMS,
            "trained_on_days":   len(x_train),
            "train_date_range":  [str(df_train.index.min().date()),
                                  str(df_train.index.max().date())],
            "train_date":         str(date.today()),
            "impersonator_dates": list(ALL_IMPERSONATOR_DATES),
        }, f, indent=2)



    """
    TEST ON TEST SET, LABELING IMPERSONATORS A & B SEPARATELY
    """
    # SCORE TEST SET WITH DECISION FUNCTION()
    x_test = handle_median(df_test[feature_cols])
    scores = pipeline.decision_function(x_test)  # decision_function() gives distance from boundary (negative = more anomalous)
    preds  = pipeline.predict(x_test) # +1 = normal, -1 = anomaly

    # ADD SCORES AND LABELS TO TEST SET DATAFRAME
    results = df_test.copy() 
    results["anomaly_score"] = scores 
    results["is_anomaly"] = (preds == -1).astype(int) # 1 if flagged as anomaly, 0 if normal
    
    # EACH ROW LABELED AS: "baseline", "impersonator_A", "impersonator_B", or "impersonator_both" based on date
    results["who"] = [label_row(d) for d in results.index.strftime("%Y-%m-%d")] 
    
    
    
    """
    PRINT SUMMARY OF RESULTS IN TERMINAL
    """

    print(f"\n Test set results:")
    print("Note: Average anomaly score is the distance from boundary. The higher the score, the more normal." \
          "The lower the score, the more anomalous.")

    # SUMMARY  
    for who, group in results.groupby("who"):

        # COUNT FLAGGED DAYS
        n_flagged = group["is_anomaly"].sum()

        # AVERAGE DISTANCE FROM BOUNDARY
        avg_score = group["anomaly_score"].mean() # average distance from boundary 
        
        print(f"{who:25s}: {len(group):2d} days, "
              f"{n_flagged} flagged ({n_flagged/len(group)*100:.0f}%), "
              f"avg score: {avg_score:.4f}")

    # IMSPERSONATOR DETAILS 
    imp_results = results[results["who"].str.startswith("impersonator")]
    if not imp_results.empty:
        print(f"\n Impersonator days detailed:")
        print(imp_results[["who", "anomaly_score", "is_anomaly"]].to_string())

    # SAVE RESULTS 
    test_results_path = os.path.join(RESULTS_DIR, f"test_results_dataset{key}.csv")
    results.to_csv(test_results_path)
    print(f"\n Full results saved to {test_results_path}")

    # DATAFRAME
    return results


def main():

    print("=" * 55)
    print("TRAIN AND PREDICT (GARMIN OC-SVM Anomaly Detection)")
    print("=" * 55)

    print(f"\n Impersonator A dates: {sorted(IMPERSONATOR_A)}")
    print(f"Impersonator B dates: {sorted(IMPERSONATOR_B)}")
    print(f"Training on first {TRAIN_DAYS} of my own data for the baseline")

    # RUN FOR EACH DATASET A, B, AND C
    # SAVE RESULTS IN DICTIONARY
    all_results = {}
    for key in DATASETS:
        r = run(key)

        if r is not None:
            all_results[key] = r

    print("\n" + "=" * 55)
    print("TASK COMPLETED.")
    print("\n RESULTS SAVED TO ./results/")
    print("=" * 55)


main()