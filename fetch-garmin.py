"""
script: fetch-garmin.py 

Desciption: Fetches Garmin biometric data using garminconnect API
            Data includes activities and wellness metrics: 
                - sleep 
                - heart rate variability (hrv) 
                - resting heart rate (rhr)
                - stress 
                - spo2
                - respiration 
                - steps
                - heart rates 
                - body battery
            for the last 62 days. 
            Saves raw data to JSON files in the ./raw_data/ directory.

Note: May cache authentication tokens to ./tokens/ for reuse on subsequent runs? Garmin limits the number of login attempt/IP addresses
and can't quite figure out how to fix this issue. Will implement this if I have te time. But for now, running this as infrequently 
as possible should be fine.
"""
import os
import json
import getpass
import traceback
from datetime import date, timedelta
from garminconnect import Garmin, GarminConnectAuthenticationError

DAYS_RECORDED = 65    # giving myself a buffer  
OUTPUT_DIR = "./raw_data"
TOKEN_DIR  = "./tokens"  


"""
SAVE DATA TO JSON FILE 

parameters: name of the file and data (dict or list) to be saved
"""
def save(name, data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {name}.json ({len(data) if isinstance(data, list) else '1 record'})")

"""
GENERATE DATE RANGE FOR DAYS RECORDED 

parameters: number of days to generate (DAYS_RECORDED)
returns: generator yielding date strings in YYYY-MM-DD format
"""
def date_range(days):
    end = date.today()
    start = end - timedelta(days=days)
    current = start
    while current <= end:
        yield current.isoformat()
        current += timedelta(days=1)

"""
FETCH DAILY DATA FOR A SPECIFIC METRIC, SKIP EMPTY DAYS

parameters: Garmin API instance, method name to call, label for printing
returns: list of dicts with date, and data for days that have data
"""
def fetch_daily(api, method_name, label):

    results = []
    method = getattr(api, method_name)
    dates = list(date_range(DAYS_RECORDED))
    print(f"Fetching {label} for {len(dates)} days...", end="", flush=True)

    # LOOP THRU DAYS, CALL API METHOD FOR EACH DAY, SKIP DAYS WITH NO DATA
    for idx, date in enumerate(dates):
        try:
            result = method(date)
            if result:
                results.append({"date": date, "data": result})
        except Exception:
            pass  # skip days with no data

        # PRINTING PROGRESS PER 30 DAYS 
        if (idx + 1) % 30 == 0:
            print(".", end="", flush=True)

    print(f"Done ({len(results)} days with data)")
    return results


"""
LOGIN USING FOR GARMIN CREDENTIALS 

parameters: none 
returns: authenticated Garmin API instance
"""
def login():
    print("\n Garmin Connect Login")

    # os.makedirs(TOKEN_DIR, exist_ok=True)

    email    = input("Email: ").strip()
    password = getpass.getpass("Password: ")

    # NOTE: work on the token caching if I have time?
    # tokenstore = tells the new garminconnect where to cache OAuth tokens
    # so I only need to login once; subsequent runs reuse the cached token
    api = Garmin(email=email, password=password)

    try:
        api.login()
        print(f"\n LOGGED IN: {api.get_full_name()}")

    except GarminConnectAuthenticationError:
        print("\n AUTHENTICATION ERROR. CHECK CREDENTIALS.")
        raise

    return api


"""
FETCH ACTIVITIES, SAVE ALL TO JSON

paremeters: authenticated Garmin API instance
"""
def fetch_activities(api):

    print("\n Activities")
    all_activities = []
    
    # NOTE: garminconnect API returns activities in batches (default 20, max 100)??
    batch_size = 100 
    offset = 0

    # LOOP TO FETCH BATCHES UNTIL NO MORE ACTIVITIES LEFT, APPEND TO ALL_ACTIVITIES
    while True:
        batch = api.get_activities(offset, batch_size)
        
        if not batch:
            break
        
        all_activities.extend(batch)

        print(f"Fetched {len(all_activities)} activities so far.", end="\r") 
        
        # if batch < than the batch size, reached the end of the activities
        if len(batch) < batch_size:
            break
        offset += batch_size

    print(f"Total activities fetched: {len(all_activities)}")
    save("activities", all_activities)


"""
FETCH WELLNESS METRICS FOR DAYS_RECORDED 
SAVE EACH METRIC TO A SEPARATE JSON FILE IN OUTPUT_DIR

parameters: authenticated Garmin API instance
"""
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
            print(f"\n  Could not fetch {label}: {e}")

    # NOTE: body battery uses a date range endpoint with a ~30-day max window 
    print("Fetching body_battery in 30-day batches...", end="", flush=True)
    
    # LOOP TO FETCH BODY BATTERY IN 30-DAY BATCHES, APPEND TO ALL_BB, THEN SAVE TO JSON
    try:
        all_bb     = []
        chunk_days = 30
        end        = date.today()
        start      = end - timedelta(days=DAYS_RECORDED)
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
            try:
                chunk = api.get_body_battery(chunk_start.isoformat(), chunk_end.isoformat())
                if chunk:
                    all_bb.extend(chunk)
            except Exception:
                pass
            print(".", end="", flush=True)
            chunk_start = chunk_end + timedelta(days=1)

        save("body_battery", all_bb)
    except Exception as e:
        print(f"\n  Could not fetch body_battery: {e}")


def main():
    print("=" * 55)
    print("FETCHING GARMIN DATA.")
    print("=" * 55)

    try:
        api = login()
        fetch_activities(api)
        fetch_wellness(api)

        print("\n" + "=" * 55)
        print("Raw data saved to ./raw_data/")
        # print("Token cached to ./tokens/ (reused on next run)")
        print("=" * 55 + "\n")

    except GarminConnectAuthenticationError as e:
        print(f"\n Authentication failed: {e}")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        traceback.print_exc()


main()