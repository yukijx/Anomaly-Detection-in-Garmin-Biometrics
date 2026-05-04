import os
import json
import getpass
import traceback
from datetime import date, timedelta
from garminconnect import Garmin, GarminConnectAuthenticationError


DAYS_RECORDED = 43
OUTPUT_DIR = "./raw_data"

"""save data to json file in output directory, creating directory if it doesn't exist"""
def save(name, data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {name}.json ({len(data) if isinstance(data, list) else '1 record'})")

"""generate range of date strings from today --> DAYS_RECORDED"""
def date_range(DAYS_RECORDED):
    end = date.today()
    start = end - timedelta(days=DAYS_RECORDED)
    current = start
    while current <= end:
        yield current.isoformat()
        current += timedelta(days=1)

"""fetches daily data, skip empty/error days, returns list of {date, data} dicts"""
def fetch_daily(api, method_name, label):

    results = []
    method = getattr(api, method_name)
    dates = list(date_range(DAYS_RECORDED))
    print(f"Fetching {label} for {len(dates)} days...", end="", flush=True) 

    for i, d in enumerate(dates):
        try:
            result = method(d)
            if result:
                results.append({"date": d, "data": result})
        except Exception:
            pass  # skip days with no data

        # progress dot every 30 days
        if (i + 1) % 30 == 0:
            print(".", end="", flush=True)

    print(f"done ({len(results)} days with data)")
    return results


""" prompt user for Garmin credentials """
def login():
    print("\n Garmin Connect Login")
    
    email = input("Email: ").strip()
    password = getpass.getpass("Password: ")

    api = Garmin(email=email, password=password)

    try:
        api.login()
        api.garth.dump("~/.garth") # cache session to avoid re-login
    except GarminConnectAuthenticationError:

        # if MFA required, would error, but I don't need MFA LOL 
        print("\nUnexpected authentication error.")

    print(f"\nLogged in: {api.get_full_name()}")
    return api


""" fetch activities and save all activities to activities to json"""
def fetch_activities(api):
    print("\n Activities")
    all_activities = []
    batch_size = 100
    offset = 0

    while True:
        batch = api.get_activities(offset, batch_size)
        if not batch:
            break
        all_activities.extend(batch)
        print(f"Fetched {len(all_activities)} activities so far...", end="\r")
        if len(batch) < batch_size:
            break
        offset += batch_size

    print(f"Total activities fetched: {len(all_activities)}        ")
    save("activities", all_activities)


""" fetch wellness metrics for last DAYS_RECORDED days, save each metric to json"""
def fetch_wellness(api):
    print(f"\n Wellness Data (last {DAYS_RECORDED} days)")

    metrics = [
        ("get_sleep_data",       "sleep"),
        ("get_hrv_data",         "hrv"),
        ("get_rhr_day",          "rhr"),
        ("get_stress_data",      "stress"),
        ("get_spo2_data",        "spo2"),
        ("get_respiration_data", "respiration"),
        ("get_steps_data",       "steps"),
        ("get_heart_rates",      "heart_rates"),
    ]

    for method_name, label in metrics:
        try:
            data = fetch_daily(api, method_name, label)
            save(label, data)
        except Exception as e:
            print(f"\n Could not fetch {label}: {e}")

    # Body battery uses a date range endpoint but has a max window of ~30 days
    # so chunk the full range into 30-day batches
    print("Fetching body_battery in 30-day batches...", end="", flush=True)
    try:
        all_bb = []
        chunk_days = 30
        end = date.today()
        start = end - timedelta(days=DAYS_RECORDED)
        chunk_start = start

        """ 
        loop through date range in 30-day chunks, fetch body battery for each chunk, 
        skip empty/error chunks, and aggregate results
        """
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
            try:
                chunk = api.get_body_battery(chunk_start.isoformat(), chunk_end.isoformat())
                if chunk:
                    all_bb.extend(chunk)
            except Exception:
                pass  # skip chunks with no data
            print(".", end="", flush=True)
            chunk_start = chunk_end + timedelta(days=1)

        save("body_battery", all_bb)
    except Exception as e:
        print(f"\n Could not fetch body_battery: {e}")


def main():
    print("=" * 55)
    print(" Fetching Garmin Data")
    print("=" * 55)

    try:
        api = login()
        fetch_activities(api)
        fetch_wellness(api)

        print("\n" + "=" * 55)
        print("Raw data saved to ./raw_data/")
        print("=" * 55 + "\n")

    except GarminConnectAuthenticationError as e:
        print(f"\nAuthentication failed: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()


main()