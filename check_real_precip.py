import requests
from datetime import datetime, timedelta

# === Fill these with your location values ===
lat = 51.50853
lon = -0.12574
horizon_min = 120  # horizon in minutes

# === Compute the target time ===
t0 = datetime.utcnow()  # Use the current UTC time
target_time = t0 + timedelta(minutes=horizon_min)
target_date = target_time.strftime("%Y-%m-%d")

# === Fetch observed precipitation for the target date ===
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": "precipitation",
    "timezone": "UTC"
}
resp = requests.get(url, params=params)
resp.raise_for_status()
data = resp.json()

# === Find the precipitation at the target hour ===
times = data["hourly"]["time"]
precips = data["hourly"]["precipitation"]

# Find the closest hour to target_time
target_hour_str = target_time.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
if target_hour_str in times:
    idx = times.index(target_hour_str)
    observed_precip = precips[idx]
    print(f"Observed precipitation at {target_hour_str}: {observed_precip} mm")
else:
    print(f"No observation found for {target_hour_str}. Available times: {times}")