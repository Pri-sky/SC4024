import pandas as pd
from sympy import div
pd.set_option('display.max_columns', None)

df = pd.read_csv("T_ONTIME_MARKETING.csv")
# print(df.info()) #602953 x 118 cols



core_columns = [
    'FL_DATE', 'TAIL_NUM', 'CRS_DEP_TIME', 'OP_UNIQUE_CARRIER', 
    'ORIGIN', 'ORIGIN_CITY_NAME', 'DEST', 'DEST_CITY_NAME', 
    'DISTANCE_GROUP', 'DEP_DELAY', 'ARR_DELAY',
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
]

existing_cols = [c for c in core_columns if c in df.columns]
df_clean = df[existing_cols].copy()

delay_cats = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
existing_delay_cats = [c for c in delay_cats if c in df_clean.columns]
df_clean[existing_delay_cats] = df_clean[existing_delay_cats].fillna(0)

# 3. Calculate the "Flight Leg Sequence" for Graph 5
# Sort chronologically, then count the daily flights for each metal airplane
df_clean = df_clean.sort_values(by=['TAIL_NUM', 'FL_DATE', 'CRS_DEP_TIME'])
df_clean['Flight_Leg'] = df_clean.groupby(['TAIL_NUM', 'FL_DATE']).cumcount() + 1

# 4. Melt the delay categories for Graph 2
id_vars = [c for c in df_clean.columns if c not in existing_delay_cats]

df_final = pd.melt(
    df_clean,
    id_vars=id_vars,
    value_vars=existing_delay_cats,
    var_name='Cause_of_Delay',
    value_name='Delay_Minutes'
)

airline_names = {
    # Major Airlines
    'AA': 'American Airlines',
    'DL': 'Delta Air Lines',
    'UA': 'United Airlines',
    'WN': 'Southwest Airlines',
    'B6': 'JetBlue Airways',
    'AS': 'Alaska Airlines',
    'NK': 'Spirit Airlines',
    'F9': 'Frontier Airlines',
    'G4': 'Allegiant Air',
    
    # Regional/Contract Airlines
    'OO': 'SkyWest Airlines',
    'MQ': 'Envoy Air',
    'YX': 'Republic Airways',
    '9E': 'Endeavor Air',
    'OH': 'PSA Airlines',
    'YV': 'Mesa Airlines',
    'QX': 'Horizon Air',
    'PT': 'Piedmont Airlines',
    'G7': 'GoJet Airlines',
    'C5': 'CommuteAir'
}

# Create a new, readable column in your dataset
df_final['Airline_Name'] = df_final['OP_UNIQUE_CARRIER'].map(airline_names)

# Check the results
print(df_final[['OP_UNIQUE_CARRIER', 'Airline_Name']].drop_duplicates().head(5))
# Preview your perfectly formatted, presentation-ready dataset
print(df_final.info())
print(df_final.tail(20))

df_final.to_csv("sc4024_cleaned.csv", index=False)