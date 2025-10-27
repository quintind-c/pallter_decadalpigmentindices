# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
from scipy.io import loadmat
import math
import os
from pathlib import Path
from datetime import datetime, timedelta
import gsw
# =============================================================================

# =============================================================================
# Functions
# =============================================================================
def extract_real_if_complex(value):
    if np.iscomplexobj(value):
        return value.real
    else:
        return value

def process_mat_file(file_path):
    # Load the MATLAB file
    mat_data = loadmat(file_path, squeeze_me=True, struct_as_record=False)

    # Initialize a DataFrame to hold all the aggregated data
    all_cruises_df = pd.DataFrame()

    # Filter out internal MATLAB keys (which start with '__') to get the cruise names
    cruise_names = [key for key in mat_data.keys() if not key.startswith('__')]

    # Iterate over each cruise in the loaded data
    for cruise_name in cruise_names:
        cruise_data = mat_data[cruise_name]

        # Initialize a dictionary to hold the aggregated data from all events for this cruise
        aggregated_data = {}

        # Iterate over all event fields in the cruise
        for event_name in cruise_data._fieldnames:
            event_data = getattr(cruise_data, event_name)

            # Determine the maximum length of arrays in this event to pad accordingly
            event_max_length = 0

            # Iterate over each field in the event to find the max length
            for field in event_data._fieldnames:
                field_data = getattr(event_data, field)
                if isinstance(field_data, np.ndarray):
                    event_max_length = max(event_max_length, field_data.size)

            # Pad data as necessary and aggregate
            for field in event_data._fieldnames:
                field_data = getattr(event_data, field)
                if isinstance(field_data, np.ndarray):
                    data_array = field_data.flatten()
                    if np.issubdtype(data_array.dtype, np.integer):
                        data_array = data_array.astype(float)
                    data_array = extract_real_if_complex(data_array)
                    padded_data = np.pad(data_array, (0, event_max_length - data_array.size), constant_values=np.nan)
                else:
                    if np.issubdtype(type(field_data), np.integer):
                        field_data = float(field_data)
                    field_data = extract_real_if_complex(field_data)
                    padded_data = np.full(event_max_length, field_data)
                if field not in aggregated_data:
                    aggregated_data[field] = padded_data.tolist()
                else:
                    aggregated_data[field].extend(padded_data.tolist())

        # Convert the aggregated data into a pandas DataFrame for this cruise
        cruise_df = pd.DataFrame(aggregated_data)
        cruise_df['Cruise'] = cruise_name
        all_cruises_df = pd.concat([all_cruises_df, cruise_df], ignore_index=True)

    return all_cruises_df

def find_nearest(value, sorted_values):
    """
    Find the nearest value in a sorted list. If equidistant, return None.
    """
    # Find the index of the smallest value larger than the input
    idx = next((i for i, v in enumerate(sorted_values) if v > value), len(sorted_values))
    
    # If the value is the smallest, return the first value in sorted_values
    if idx == 0:
        return sorted_values[0]
    # If the value is the largest, return the last value in sorted_values
    elif idx == len(sorted_values):
        return sorted_values[-1]
    # If equidistant, return None
    elif (sorted_values[idx] - value) == (value - sorted_values[idx - 1]):
        return None
    # Return the nearest value
    elif (sorted_values[idx] - value) < (value - sorted_values[idx - 1]):
        return sorted_values[idx]
    else:
        return sorted_values[idx - 1]

def round_to_nearest(grid_line, grid_station, rounded_grid_lines, rounded_grid_stations):
    """
    Round the given GridLine and GridStation to the nearest values from the lists.
    """
    return find_nearest(grid_line, rounded_grid_lines), find_nearest(grid_station, rounded_grid_stations)

def pressure_to_depth(pressure, latitude):
    """
    Equations from https://www.seabird.com/asset-get.download.jsa?id=54627861710
    """
    # Calculate x using the sine of latitude in radians
    x = math.sin(math.radians(latitude))**2

    # Calculate gravity (g) in m/s^2
    g = 9.780318 * (1.0 + ((5.2788 * 10**-3) + (2.36 * 10**-5) * x) * x) + (1.092 * 10**-6) * pressure

    # Calculate depth in meters
    depth = ((((-1.82 * 10**-15) * pressure + (2.279 * 10**-10)) * pressure - (2.2512 * 10**-5)) * pressure + 9.72659) * pressure / g

    return depth
# =============================================================================


# =============================================================================
# Load and Aggregate MATLAB file to DataFrame
# =============================================================================
"""
MUST RUN 'convertCruiseCTDmasterDatetimes' MATLAB CODE PRIOR TO IMPORT OR DATES
WILL BE INCOHERENT IN FINAL DATAFRAME
"""
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("CruiseCTD_MasterData_fromRUBoardwalk_DateAdjusted.mat")
loadpath = str(current_directory / absolute_path / filename)

df = process_mat_file(loadpath)
# =============================================================================


# =============================================================================
# Calculate Rounded Lines/Stations
# =============================================================================
#define min/max rounded grid line numbers based on true numbers
min_grid_line = (df['GridLine'].min() // 100) * 100
max_grid_line = ((df['GridLine'].max() // 100) + 1) * 100

#define min/max rounded grid station numbers based on true numbers
min_grid_station = (df['GridStation'].min() // 20) * 20
max_grid_station = ((df['GridStation'].max() // 20) + 1) * 20

#generate rounded line/station range lists based on min/max numbers
rounded_grid_lines = list(range(int(min_grid_line), int(max_grid_line) + 1, 100))
rounded_grid_stations = list(range(int(min_grid_station), int(max_grid_station) + 1, 20))

#apply the rounding function to each row
df['RoundedGridLine'], df['RoundedGridStation'] = zip(*df.apply(
    lambda row: round_to_nearest(row['GridLine'], row['GridStation'], rounded_grid_lines, rounded_grid_stations), axis=1))
# =============================================================================


# =============================================================================
# Assign Grid Regions Based on Rounded Line/Station Numbers (using EDI dataframe as reference)
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EDICruiseDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
df2 = pd.read_csv(loadpath)

region_dict = {}

#identify rounded line/stations belonging to regions - add to mapping dictionary
for index, row in df2.iterrows():
    key = (row['RoundedGridLine'], row['RoundedGridStation'])
    value = {'Region': row['Region'],'IORegion': row['IORegion'],'NSRegion': row['NSRegion']}
    region_dict[key] = value

#use mapping dictionary to assing regions in df
df['Region'] = df.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('Region', None), axis=1)
df['IORegion'] = df.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('IORegion', None), axis=1)
df['NSRegion'] = df.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('NSRegion', None), axis=1)

# #replace nan numbers with nan instances
# df['Region'] = df['Region'].fillna(-9999).astype('int64').replace(-9999, np.nan)
# df['IORegion'] = df['IORegion'].fillna(-9999).astype('int64').replace(-9999, np.nan)
# df['NSRegion'] = df['NSRegion'].fillna(-9999).astype('int64').replace(-9999, np.nan)

#replace None instances with nan instances
df['Region'] = df['Region'].replace([None], np.nan)
df['IORegion'] = df['IORegion'].replace([None], np.nan)
df['NSRegion'] = df['NSRegion'].replace([None], np.nan)
# =============================================================================


# =============================================================================
# Convert DateTime; Extract Temporal Info
# =============================================================================
df['Date'] = pd.to_datetime(df['DateTime'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Date'].dt.hour
# =============================================================================


# =============================================================================
# Calculate Depth from Pressure
# =============================================================================
#some depths are missing; use seabird equations and pressure to calculate missing depths 
df.rename(columns={'Depth': 'Orig_Depth'}, inplace=True)
df['Depth'] = df.apply(lambda row: round(pressure_to_depth(row['Pressure'], row['Latitude'])), axis=1)

#check values when overlapping to ensure calculations match
depthcheck = df[['Year','Event','Orig_Depth','Depth']].dropna()
# =============================================================================


# =============================================================================
# Misc. Data Handling
# =============================================================================
#reorder columns
columns_to_move = ['Cruise', 'Date', 'Year', 'Month', 'Day', 'Hour', 'Event', 
                   'GridLine', 'GridStation', 'RoundedGridLine', 'RoundedGridStation', 
                   'Region', 'NSRegion', 'IORegion', 'Latitude', 'Longitude', 'Depth']
df = df[columns_to_move + [col for col in df.columns if col not in columns_to_move]]

#fill SigmaTheta nan values with values from mislabeled column
df['SigmaTheta'] = df['SigmaTheta'].fillna(df['SigmatTheta'])

#fill missing density values with sigmatheta values
df['Density'] = df['Density'].fillna(df['SigmaTheta']+1000)

#fix negative QI values
df.loc[df['QI'] < 0,'QI'] = 0

#drop excess columns
droprefs = ['Station',
            'Cast',
            'Flag',
            'Orig_Depth',
            'SigmatTheta']
df = df.drop(columns=droprefs)
# =============================================================================


# =============================================================================
# Save DataFrame to CSV file
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_RutgersCruiseCTDMasterDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

df.to_csv(savepath, index=False)
# =============================================================================
























































