
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 1. SETTINGS
# ==========================================

INPUT = Path("sc4024_cleaned.csv")
OUT = Path("sc4024_v2_outputs")
OUT.mkdir(exist_ok=True)

EVENT_DATES = pd.to_datetime([
    "2026-01-25",
    "2026-01-26"
])

TOP_N_AIRLINES = 8

COLUMNS = [
    "FL_DATE", "TAIL_NUM", "CRS_DEP_TIME",
    "OP_UNIQUE_CARRIER", "ORIGIN", "ORIGIN_CITY_NAME",
    "DEST", "DEST_CITY_NAME", "DISTANCE_GROUP",
    "DEP_DELAY", "ARR_DELAY", "Flight_Leg",
    "Cause_of_Delay", "Delay_Minutes", "Airline_Name"
]

FLIGHT_FILE = OUT / "01_flights.csv"
CAUSE_FILE = OUT / "02_positive_causes.csv"

for path in [FLIGHT_FILE, CAUSE_FILE]:
    if path.exists():
        path.unlink()

# ==========================================
# 2. EXTRACT TWO SEPARATE DATASETS
# ==========================================

total_input_rows = 0

for chunk in pd.read_csv(
    INPUT,
    usecols=COLUMNS,
    chunksize=200_000,
    low_memory=False
):
    total_input_rows += len(chunk)

    chunk["Delay_Minutes"] = pd.to_numeric(
        chunk["Delay_Minutes"], errors="coerce"
    )

    # One representative record per flight.
    # CARRIER_DELAY is the repeated-record selector,
    # not evidence of a carrier-caused delay.
    flights_chunk = chunk.loc[
        chunk["Cause_of_Delay"] == "CARRIER_DELAY"
    ].copy()

    flights_chunk = flights_chunk.rename(columns={
        "Delay_Minutes": "Carrier_Delay_Minutes"
    })

    # Only positive attributed delay minutes.
    causes_chunk = chunk.loc[
        chunk["Delay_Minutes"] > 0
    ].copy()

    flights_chunk.to_csv(
        FLIGHT_FILE, mode="a", index=False,
        header=not FLIGHT_FILE.exists()
    )

    causes_chunk.to_csv(
        CAUSE_FILE, mode="a", index=False,
        header=not CAUSE_FILE.exists()
    )

flights = pd.read_csv(FLIGHT_FILE)
causes = pd.read_csv(CAUSE_FILE)

if total_input_rows != 5 * len(flights):
    warnings.warn(
        "The input does not have exactly five rows per "
        "selected flight. Investigate before continuing."
    )

# ==========================================
# 3. CREATE FLIGHT-LEVEL VARIABLES
# ==========================================

def parse_dates(series):
    date_string = series.astype(str).str.split().str[0]
    return pd.to_datetime(
        date_string,
        format="%m/%d/%Y",
        errors="coerce"
    )

flights["DATE"] = parse_dates(flights["FL_DATE"])
flights = flights.loc[flights["DATE"].notna()].copy()

for col in ["DEP_DELAY", "ARR_DELAY",
            "CRS_DEP_TIME", "Flight_Leg"]:
    flights[col] = pd.to_numeric(
        flights[col], errors="coerce"
    )

flights["HOUR"] = (
    flights["CRS_DEP_TIME"] // 100
).astype("Int64")

flights["WEEKDAY"] = flights["DATE"].dt.day_name()

# NaN means arrival-delay information is missing.
flights["LATE15"] = np.where(
    flights["ARR_DELAY"].notna(),
    (flights["ARR_DELAY"] >= 15).astype(int),
    np.nan
)

flights["MISSING_ARRIVAL"] = (
    flights["ARR_DELAY"].isna()
)

# Identify event dates and a weekday-matched control.
event_mask = flights["DATE"].isin(EVENT_DATES)

matched_weekdays = set(EVENT_DATES.dayofweek)

control_mask = (
    ~event_mask
    & flights["DATE"].dt.dayofweek.isin(matched_weekdays)
)

flights["COHORT"] = np.select(
    [event_mask, control_mask],
    ["event", "weekday_control"],
    default="other_days"
)

flights["PERIOD"] = np.where(
    event_mask, "Jan25_26", "Other_January"
)

# ==========================================
# 4. COMMON AGGREGATION FUNCTION
# ==========================================

def summarise(grouped):
    result = grouped.agg(
        flights=("ARR_DELAY", "size"),
        valid_arrivals=("ARR_DELAY", "count"),
        delayed15=("LATE15", "sum"),
        missing_arrivals=("MISSING_ARRIVAL", "sum")
    ).reset_index()

    result["late_rate"] = (
        result["delayed15"]
        / result["valid_arrivals"].replace(0, np.nan)
    )

    result["missing_rate"] = (
        result["missing_arrivals"] / result["flights"]
    )

    return result

# ==========================================
# 5. PLOT 1 — CALENDAR
# ==========================================

calendar = summarise(
    flights.groupby("DATE")
)

calendar["missing_warning"] = (
    calendar["missing_rate"] >= 0.20
)

calendar["highlight_event"] = (
    calendar["DATE"].isin(EVENT_DATES)
)

calendar.to_csv(
    OUT / "03_calendar.csv", index=False
)

# ==========================================
# 6. PLOT 2 — TEMPORAL FINGERPRINT
# ==========================================

hourly = flights.loc[
    flights["HOUR"].between(5, 23)
].copy()

day_hour = summarise(
    hourly.groupby(["DATE", "HOUR"])
)

# Typical hourly rate excluding the chosen event.
baseline_hour = summarise(
    hourly.loc[hourly["COHORT"] != "event"]
    .groupby("HOUR")
)

baseline_hour = baseline_hour[
    ["HOUR", "late_rate"]
].rename(columns={
    "late_rate": "typical_hour_rate"
})

day_hour = day_hour.merge(
    baseline_hour, on="HOUR", how="left"
)

# Positive = worse than typical for that hour.
day_hour["excess_delay_pp"] = 100 * (
    day_hour["late_rate"]
    - day_hour["typical_hour_rate"]
)

day_hour["sufficient_data"] = (
    day_hour["valid_arrivals"] >= 35
)

day_hour.to_csv(
    OUT / "04_temporal_fingerprint.csv",
    index=False
)

# ==========================================
# 7. PLOT 3 — GEOSPATIAL CHANGE MAP
# ==========================================

airport_data = flights.loc[
    flights["COHORT"].isin([
        "event", "weekday_control"
    ])
]

airport_summary = summarise(
    airport_data.groupby([
        "ORIGIN",
        "ORIGIN_CITY_NAME",
        "COHORT"
    ])
)

airport_wide = airport_summary.pivot(
    index=["ORIGIN", "ORIGIN_CITY_NAME"],
    columns="COHORT",
    values=[
        "flights", "valid_arrivals",
        "late_rate", "missing_rate"
    ]
)

airport_wide.columns = [
    f"{measure}_{period}"
    for measure, period in airport_wide.columns
]

airport_wide = airport_wide.reset_index()

airport_wide["delay_change_pp"] = 100 * (
    airport_wide["late_rate_event"]
    - airport_wide["late_rate_weekday_control"]
)

# Minimum sample sizes for the main comparison.
airport_wide["sufficient_data"] = (
    (airport_wide["valid_arrivals_event"] >= 20)
    & (
        airport_wide[
            "valid_arrivals_weekday_control"
        ] >= 100
    )
)

airport_wide.to_csv(
    OUT / "05_airport_change.csv",
    index=False
)

# Routes for a separate map drill-down.
routes = summarise(
    flights.groupby(["ORIGIN", "DEST"])
)

routes = routes.loc[
    routes["flights"] >= 50
]

routes.to_csv(
    OUT / "06_routes.csv", index=False
)

# ==========================================
# 8. PLOT 4 — DEPARTURE/ARRIVAL SANKEY
# ==========================================

valid = flights.dropna(
    subset=["DEP_DELAY", "ARR_DELAY"]
).copy()

labels = ["Early", "0-14", "15-59", "60+"]
bins = [-np.inf, 0, 15, 60, np.inf]

valid["DEP_BUCKET"] = pd.cut(
    valid["DEP_DELAY"],
    bins=bins,
    labels=labels,
    right=False
)

valid["ARR_BUCKET"] = pd.cut(
    valid["ARR_DELAY"],
    bins=bins,
    labels=labels,
    right=False
)

transitions = (
    valid.groupby(
        ["DEP_BUCKET", "ARR_BUCKET"],
        observed=True
    )
    .size()
    .reset_index(name="flights")
)

transitions["departure_total"] = (
    transitions.groupby("DEP_BUCKET", observed=True)[
        "flights"
    ].transform("sum")
)

transitions["share_within_departure"] = (
    transitions["flights"]
    / transitions["departure_total"]
)

transitions.to_csv(
    OUT / "07_delay_transitions.csv",
    index=False
)

# ==========================================
# 9. PLOT 5 — AIRLINE FINGERPRINTS
# ==========================================

top_airlines = (
    flights["Airline_Name"]
    .value_counts()
    .head(TOP_N_AIRLINES)
    .index
)

airline_data = hourly.loc[
    hourly["Airline_Name"].isin(top_airlines)
]

airline_hour = summarise(
    airline_data.groupby(
        ["Airline_Name", "HOUR"]
    )
)

# Network-wide hourly comparison.
network_hour = summarise(
    hourly.groupby("HOUR")
)

network_hour = network_hour[
    ["HOUR", "late_rate"]
].rename(columns={
    "late_rate": "network_hour_rate"
})

airline_hour = airline_hour.merge(
    network_hour, on="HOUR", how="left"
)

airline_hour["relative_delay_pp"] = 100 * (
    airline_hour["late_rate"]
    - airline_hour["network_hour_rate"]
)

airline_hour["sufficient_data"] = (
    airline_hour["valid_arrivals"] >= 80
)

airline_hour.to_csv(
    OUT / "08_airline_fingerprints.csv",
    index=False
)

# ==========================================
# 10. PLOT 6 — DELAY-CAUSE SHIFT
# ==========================================

causes["DATE"] = parse_dates(causes["FL_DATE"])

causes["PERIOD"] = np.where(
    causes["DATE"].isin(EVENT_DATES),
    "Jan25_26",
    "Other_January"
)

cause_summary = (
    causes.groupby([
        "PERIOD", "Cause_of_Delay"
    ])["Delay_Minutes"]
    .sum()
    .reset_index()
)

cause_summary["period_total"] = (
    cause_summary.groupby("PERIOD")[
        "Delay_Minutes"
    ].transform("sum")
)

cause_summary["cause_share"] = (
    cause_summary["Delay_Minutes"]
    / cause_summary["period_total"]
)

cause_summary.to_csv(
    OUT / "09_cause_shift.csv",
    index=False
)

# ==========================================
# 11. EXPORT CLEANED DATA + QUALITY CHECK
# ==========================================

flights.to_csv(
    FLIGHT_FILE, index=False
)

causes.to_csv(
    CAUSE_FILE, index=False
)

print("Preprocessing completed.")
print("Original rows:", total_input_rows)
print("Flight records:", len(flights))
print("Positive-cause records:", len(causes))
print("Airlines:", flights["Airline_Name"].nunique())
print("Files saved to:", OUT.resolve())
