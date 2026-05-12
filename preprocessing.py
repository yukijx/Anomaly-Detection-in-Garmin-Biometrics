"""
script: preprocessing.py 

Description: This script takes the raw JSON files fetched from Garmin APIs,
             parses out relevant wellness metrics and builds three cleaned datasets for analysis. 
             Each dataset is composed of its own unique feature combination and focuses on a different aspect of health:

Dataset A: Cardiovascular
    RHR, HRV, SpO2, respiration rate, avg. HR during activities

Dataset B: Stress & Recovery 
    RHR, HRV, stress score, body battery, steps, sleep duration

Dataset C: Sleep 
    RHR, HRV, SpO2, respiration, sleep duration, deep/REM/light %

Outputs (in ./datasets/):
    dataset_A_cardiovascular.csv
    dataset_B_stress_recovery.csv
    dataset_C_sleep.csv
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

RAW_DIR   = "./raw_data"
OUT_DIR   = "./datasets"


# takes raw JSON data, returns a cleaned DataFrame with 'date' index and relevant columns
def load_json(name):
    path = os.path.join(RAW_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f" {name}.json not found, skipping")
        return []
    with open(path) as f:
        return json.load(f)


def to_date(s):
    """Normalize Garmin date string formats to YYYY-MM-DD"""
    if not s:
        return None
    
    # trying multiple patterns b/c Garmin APIs are inconsistent with date formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(s)[:19], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # return first 10 chars as fallback, haven't actually tested this tho
    return str(s)[:10]


# parsing, handling Garmin's inconsistent nesting and naming conventions
def parse_rhr(raw):
    rows = []
    for entry in raw:
        d = entry.get("data", {})
        date_str = to_date(entry.get("date") or d.get("calendarDate"))
        val = (d.get("restingHeartRate")
               or d.get("value")
               or d.get("rhr"))
        if date_str and val:
            rows.append({"date": date_str, "rhr": float(val)})
    
    # return empty DataFrame if no valid rows
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def parse_hrv(raw):
    rows = []
    for entry in raw:
        d = entry.get("data", {})
        date_str = to_date(entry.get("date") or d.get("calendarDate") or d.get("startTimestampLocal"))

        # HRV summary nests under hrvSummary
        summary = d.get("hrvSummary", d)

        val = (summary.get("lastNight")
               or summary.get("weeklyAvg")
               or summary.get("rmssd")
               or d.get("lastNight"))
        
        if date_str and val:
            rows.append({"date": date_str, "hrv_ms": float(val)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def parse_sleep(raw):
    rows = []
    for entry in raw:
        
        d = entry.get("data", {}) #getting data dict

        summary = d.get("dailySleepDTO", d)
        date_str = to_date(entry.get("date")
                    or summary.get("calendarDate")
                    or d.get("calendarDate"))

        duration_s  = summary.get("sleepTimeSeconds") or summary.get("totalSleepSeconds")
        deep_s      = summary.get("deepSleepSeconds")
        rem_s       = summary.get("remSleepSeconds")
        light_s     = summary.get("lightSleepSeconds")
        awake_s     = summary.get("awakeSleepSeconds")

        # only include rows with a valid date and total sleep duration
        if date_str and duration_s:
            total = duration_s
            row = {
                "date":              date_str,
                "sleep_hours":       round(total / 3600, 2), # convert seconds to hours
                "deep_pct":          round(deep_s  / total * 100, 1) if deep_s  else np.nan,
                "rem_pct":           round(rem_s   / total * 100, 1) if rem_s   else np.nan,
                "light_pct":         round(light_s / total * 100, 1) if light_s else np.nan,
                "awake_pct":         round(awake_s / total * 100, 1) if awake_s else np.nan,
            } # converting to percentages of total sleep time, rounding to 1 decimal
            rows.append(row)
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def parse_stress(raw):
    rows = []
    for entry in raw:
        d = entry.get("data", {})
        date_str = to_date(entry.get("date") or d.get("calendarDate"))

        # stress values come as a list of [timestamp, value] pairs
        stress_vals = d.get("stressValuesArray") or d.get("stressValues") or []
        if stress_vals:
            # Filter out -1 (no reading) and -2 (activity)
            # -1 and -2 are Garmin's codes for "no reading" and "activity time," which i want to exclude from stress calculations
            vals = [v[1] for v in stress_vals
                    if isinstance(v, (list, tuple)) and len(v) >= 2 and v[1] >= 0]
            
            # only days with valid date and at least one stress reading
            if vals:
                rows.append({
                    "date":       date_str,
                    "avg_stress": round(np.mean(vals), 1),
                    "max_stress": float(np.max(vals)),
                })
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def parse_body_battery(raw):
    """Body battery comes as a flat list of {date, charged, drained, ...}"""
    rows = []

    for entry in raw:
        if isinstance(entry, dict):
            date_str = to_date(entry.get("date") or entry.get("calendarDate"))
            
            val = (entry.get("endOfDayBodyBattery")
                   or entry.get("charged")
                   or entry.get("bodyBatteryStatList", [{}])[0].get("bodyBatteryEnd"))
            
            if date_str and val is not None:
                rows.append({"date": date_str, "body_battery_eod": float(val)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def parse_spo2(raw):
    rows = []

    for entry in raw:
        d = entry.get("data", {})
        date_str = to_date(entry.get("date") or d.get("calendarDate"))
        reading_list = d.get("continuousReadingDTOList") or []
        avg = (d.get("averageSpO2") or d.get("avg")
               or (reading_list[0].get("averageSpO2") if reading_list else None))
        
        if date_str and avg:
            rows.append({"date": date_str, "spo2_avg": float(avg)})

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def parse_respiration(raw):
    rows = []
    for entry in raw:
        d = entry.get("data", {})
        date_str = to_date(entry.get("date") or d.get("calendarDate") or d.get("startTimestampLocal"))
        avg = d.get("avgWakingRespirationValue") or d.get("avg") or d.get("averageRespirationValue")
        
        if date_str and avg:
            rows.append({"date": date_str, "respiration_avg": float(avg)})

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def parse_steps(raw):
    rows = []
    for entry in raw:
        d = entry.get("data", [])
        date_str = to_date(entry.get("date"))
        
        if not date_str:
            continue
        
        # data is a list of 15-min epoch intervals, sum steps across all of them
        if isinstance(d, list):
            total = sum(interval.get("steps", 0) for interval in d)
            if total > 0:
                rows.append({"date": date_str, "steps": total})
        
        elif isinstance(d, dict):
            steps = d.get("totalSteps") or d.get("steps")
            if steps:
                rows.append({"date": date_str, "steps": int(steps)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


"""extract daily rollups from activities: avg HR, active calories, activity count"""
def parse_activities(raw):
    rows = []
    for act in raw:
        date_str = to_date(act.get("startTimeLocal") or act.get("startTimeGMT"))
        if not date_str:
            continue

        act_type = str(act.get("activityType", {}).get("typeKey", "unknown")).lower()
        avg_hr   = act.get("averageHR")
        calories = act.get("calories")
        duration = act.get("duration")  # seconds

        rows.append({
            "date":         date_str,
            "activity_type": act_type,
            "avg_hr":        float(avg_hr)   if avg_hr   else np.nan,
            "active_cals":   float(calories) if calories else np.nan,
            "duration_min":  round(float(duration) / 60, 1) if duration else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    # aggregate to daily: avg HR, total calories, total duration, count
    daily = df.groupby("date").agg(
        avg_activity_hr = ("avg_hr",       "mean"),
        total_cals      = ("active_cals",  "sum"),
        total_active_min= ("duration_min", "sum"),
        activity_count  = ("date",         "count"),
    ).round(1)

    return daily


"""build datasets A, B, C with different feature combinations, save to .CSV files"""
def build_all():

    print("\nLoading raw data files...")

    rhr         = parse_rhr(load_json("rhr"))
    hrv         = parse_hrv(load_json("hrv"))
    sleep       = parse_sleep(load_json("sleep"))
    stress      = parse_stress(load_json("stress"))
    body_batt   = parse_body_battery(load_json("body_battery"))
    spo2        = parse_spo2(load_json("spo2"))
    respiration = parse_respiration(load_json("respiration"))
    steps       = parse_steps(load_json("steps"))
    activities  = parse_activities(load_json("activities"))

    os.makedirs(OUT_DIR, exist_ok=True)

    # Dataset A: Cardiovascular 
    print("\nBuilding Dataset A: Cardiovascular...")
 
    dfs_a = [rhr, hrv, spo2, respiration, activities[["avg_activity_hr", "total_cals"]]]
    dfs_a = [d for d in dfs_a if not d.empty]
    ds_a = pd.concat(dfs_a, axis=1, join="outer")
    ds_a = ds_a.dropna(thresh=3).sort_index()  # keep rows with at least 3 features
    ds_a.index.name = "date"
    path_a = os.path.join(OUT_DIR, "dataset_A_cardiovascular.csv")
    ds_a.to_csv(path_a)

    # print summary of each dataset and features included
    print(f" {path_a}  — {len(ds_a)} rows, {ds_a.shape[1]} features")
    print(f" Features: {list(ds_a.columns)}")

    # Dataset B: Stress & Recovery
    print("\nBuilding Dataset B: Stress & Recovery...")

    sleep_b = sleep[["sleep_hours"]] if not sleep.empty else pd.DataFrame()
    dfs_b = [rhr, hrv, stress, body_batt, steps, sleep_b]
    dfs_b = [d for d in dfs_b if not d.empty]
    ds_b = pd.concat(dfs_b, axis=1, join="outer")
    ds_b = ds_b.dropna(thresh=3).sort_index()
    ds_b.index.name = "date"
    path_b = os.path.join(OUT_DIR, "dataset_B_stress_recovery.csv")
    ds_b.to_csv(path_b)

    # print summary of each dataset and features included
    print(f" {path_b}  — {len(ds_b)} rows, {ds_b.shape[1]} features")
    print(f" Features: {list(ds_b.columns)}")

    # Dataset C: Sleep
    print("\nBuilding Dataset C: Sleep...")
    dfs_c = [rhr, hrv, spo2, respiration, sleep]
    dfs_c = [d for d in dfs_c if not d.empty]
    ds_c = pd.concat(dfs_c, axis=1, join="outer")
    ds_c = ds_c.dropna(thresh=3).sort_index()
    ds_c.index.name = "date"
    path_c = os.path.join(OUT_DIR, "dataset_C_sleep.csv")
    ds_c.to_csv(path_c)

    print(f" {path_c}  — {len(ds_c)} rows, {ds_c.shape[1]} features")
    print(f" Features: {list(ds_c.columns)}")

    print("\n== Summary =================================================")
    print(f"  Date range in data: {ds_a.index.min()} → {ds_a.index.max()}")

    # print number of rows and features in each dataset
    print(f"  Dataset A: {ds_a.shape[0]} days x {ds_a.shape[1]} features")
    print(f"  Dataset B: {ds_b.shape[0]} days x {ds_b.shape[1]} features")
    print(f"  Dataset C: {ds_c.shape[0]} days x {ds_c.shape[1]} features") 



build_all()
