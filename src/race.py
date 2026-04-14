import pandas as pd
from pathlib import Path

runs = pd.read_csv('data/processed/runs.csv', parse_dates=['date'])

# Look for races — Strava names them explicitly sometimes
race_keywords = ['race', 'Race', '5k', '5K', '10k', '10K', 'half', 'Half',
                 'marathon', 'Marathon', 'parkrun', 'Parkrun', 'Park Run',
                 'tempo', 'time trial', 'Time Trial']

mask = runs['name'].str.contains('|'.join(race_keywords), na=False, case=False)
races = runs[mask].copy()

print(f"Activities matching race keywords: {len(races)}")
print()
print(races[['date', 'name', 'distance_km', 'moving_time_min', 
             'avg_pace_min_km', 'hr_mean']].to_string(index=False))


# Also look at your fastest runs by pace — races often show up here
# even if not named explicitly
print("\nYour 15 fastest runs by average pace:")
fastest = (runs[runs['distance_km'] >= 3]  # filter out short warmups
           .nsmallest(15, 'avg_pace_min_km')
           [['date', 'name', 'distance_km', 'moving_time_min', 'avg_pace_min_km', 'hr_mean']])
print(fastest.to_string(index=False))