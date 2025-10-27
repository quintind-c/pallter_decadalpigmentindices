# =============================================================================
# Imports 
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import trapz
import warnings
warnings.filterwarnings('ignore')
# =============================================================================


# =============================================================================
# Functions
# =============================================================================
def fill_missing_years(df):    
    all_years = np.arange(1991, 2021)
    existing_years = df['Year'].unique()
    missing_years = np.setdiff1d(all_years, existing_years)
    missing_data = pd.DataFrame({'Year': missing_years})
    missing_data = missing_data.assign(**{column: np.nan for column in df.columns if column != 'Year'})
    df = pd.concat([df, missing_data], ignore_index=True)
    df = df.sort_values('Year')
    return df
# =============================================================================


# =============================================================================
# Load Bio Dataframes
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")

#load surface-averaged dataframe
filename = Path("PALLTER_EventLevel_SurfaceAvgBioDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
dfbiosrf1 = pd.read_csv(loadpath)

#load depth-averaged dataframe
filename = Path("PALLTER_EventLevel_DepthAvgBioDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
dfbiodep1 = pd.read_csv(loadpath)
# =============================================================================


# =============================================================================
# Load Hydro Dataframes
# =============================================================================
#load surface-averaged dataframe
filename = Path("PALLTER_EventLevel_SurfaceAvgHydroDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
dfhydsrf1 = pd.read_csv(loadpath)

#load depth-averaged dataframe
filename = Path("PALLTER_EventLevel_DepthAvgHydroDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
dfhyddep1 = pd.read_csv(loadpath)
# =============================================================================


# =============================================================================
# Load Sea Ice Dataframe
# =============================================================================
filename = Path("PALLTER_EDISeaIceDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
dfsi = pd.read_csv(loadpath)
# =============================================================================


# =============================================================================
# Load Region Reference Dataframe
# =============================================================================
filename = Path("PALLTER_EDICruiseDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
refdf = pd.read_csv(loadpath)
# =============================================================================


# =============================================================================
# Build Event-Level Surface-Averaged Core Dataframe
# =============================================================================
#reduce base dataframes to event-level
dfbiosrf = dfbiosrf1.groupby(['Year','RoundedGridLine','RoundedGridStation','Event']).median().reset_index()
dfhydsrf = dfhydsrf1.groupby(['Year','RoundedGridLine','RoundedGridStation','Event']).median().reset_index()

#merge surface base dataframes
left = dfbiosrf
right = dfhydsrf
merge_on = ['Year','RoundedGridLine','RoundedGridStation','Event']

srf_merged_EM = left.merge(right, on=merge_on, how='outer', suffixes=('_bio', '_hydro'))

#fill missing values from hydro with bio values
srf_merged_EM['Temperature'] = srf_merged_EM['Temperature_hydro'].fillna(srf_merged_EM['Temperature_bio'])
srf_merged_EM['Salinity'] = srf_merged_EM['Salinity_hydro'].fillna(srf_merged_EM['Salinity_bio'])
srf_merged_EM['Density'] = srf_merged_EM['Density_hydro'].fillna(srf_merged_EM['Density_bio'])

#fill missing values from bio with hydro values
srf_merged_EM['Latitude'] = srf_merged_EM['Latitude_bio'].fillna(srf_merged_EM['Latitude_hydro'])
srf_merged_EM['Longitude'] = srf_merged_EM['Longitude_bio'].fillna(srf_merged_EM['Longitude_hydro'])
srf_merged_EM['Month'] = srf_merged_EM['Month_bio'].fillna(srf_merged_EM['Month_hydro'])
srf_merged_EM['Day'] = srf_merged_EM['Day_bio'].fillna(srf_merged_EM['Day_hydro'])
srf_merged_EM['Hour'] = srf_merged_EM['Hour_bio'].fillna(srf_merged_EM['Hour_hydro'])

srf_merged_EM = srf_merged_EM.drop(['Temperature_bio',
                                    'Salinity_bio',
                                    'Density_bio',
                                    'Temperature_hydro',
                                    'Salinity_hydro',
                                    'Density_hydro',
                                    'Latitude_bio',
                                    'Latitude_hydro',
                                    'Longitude_bio',
                                    'Longitude_hydro',
                                    'Month_hydro',
                                    'Day_hydro',
                                    'Hour_hydro',
                                    'Month_bio',
                                    'Day_bio',
                                    'Hour_bio'], axis=1)

srf_merged_EM = fill_missing_years(srf_merged_EM)
# -----------------------------------------------------------------------------
# Add EDI sea ice data
# -----------------------------------------------------------------------------
srf_merged_EM = srf_merged_EM.merge(dfsi, on=['Year'], how='left')
# -----------------------------------------------------------------------------
# Assign grid regions based on roundedgridline/station numbers (using EDI dataframe as reference)
# -----------------------------------------------------------------------------
tempdf = srf_merged_EM
region_dict = {}

#identify rounded line/stations belonging to regions - add to mapping dictionary
for index, row in refdf.iterrows():
    key = (row['RoundedGridLine'], row['RoundedGridStation'])
    value = {'Region': row['Region'],'IORegion': row['IORegion'],'NSRegion': row['NSRegion']}
    region_dict[key] = value

#use mapping dictionary to assing regions in df
tempdf['Region'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('Region', None), axis=1)
tempdf['IORegion'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('IORegion', None), axis=1)
tempdf['NSRegion'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('NSRegion', None), axis=1)

#replace None instances with nan instances
tempdf['Region'] = tempdf['Region'].replace([None], np.nan)
tempdf['IORegion'] = tempdf['IORegion'].replace([None], np.nan)
tempdf['NSRegion'] = tempdf['NSRegion'].replace([None], np.nan)
# -----------------------------------------------------------------------------
# Change region info from string to float
# -----------------------------------------------------------------------------
NSRegion = tempdf['NSRegion']
IORegion = tempdf['IORegion']
Region = tempdf['Region']

NSRegion[NSRegion == 'N'] = 1
NSRegion[NSRegion == 'S'] = 2
NSRegion[NSRegion == 'FS'] = 3

IORegion[IORegion == 'C'] = 6
IORegion[IORegion == 'Sh'] = 5
IORegion[IORegion == 'Sl'] = 4

Region[Region == 'NC'] = 3
Region[Region == 'NSh'] = 2
Region[Region == 'NSl'] = 1
Region[Region == 'SC'] = 6
Region[Region == 'SSh'] = 5
Region[Region == 'SSl'] = 4
Region[Region == 'FSC'] = 9
Region[Region == 'FSSh'] = 8
Region[Region == 'FSSl'] = 7

NSRegion = tempdf.NSRegion.astype(float)
IORegion = tempdf.IORegion.astype(float)
Region = tempdf.Region.astype(float)

#   4       5       6    
#   -       -       - 
#   Sl      Sh      C   
# 
# |______|_______|_______|__
# |      |       |       | 
# |  1   |   2   |   3   |     N  - 1
# |______|_______|_______|__
# |      |       |       |
# |  4   |   5   |   6   |     S  - 2
# |______|_______|_______|__
# |      |       |       |
# |  7   |   8   |   9   |     FS - 3
# |______|_______|_______|__
#   ^Numeric representation of grid regions

tempdf['Region'] = tempdf['Region'].astype(float)
tempdf['NSRegion'] = tempdf['NSRegion'].astype(float)
tempdf['IORegion'] = tempdf['IORegion'].astype(float)

srf_merged_EM = tempdf
# -----------------------------------------------------------------------------
# Save surface averaged bio data
# -----------------------------------------------------------------------------
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EventLevel_SurfaceAvgCoreDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

srf_merged_EM.to_csv(savepath, index=False)
# =============================================================================


# =============================================================================
# Build Station-Level Surface-Averaged Core Dataframe
# =============================================================================
#reduce base dataframes to station-level
dfbiosrf = dfbiosrf1.groupby(['Year','RoundedGridLine','RoundedGridStation','Event']).median().reset_index()
dfbiosrf = dfbiosrf.groupby(['Year','RoundedGridLine','RoundedGridStation']).median().reset_index()

dfhydsrf = dfhydsrf1.groupby(['Year','RoundedGridLine','RoundedGridStation','Event']).median().reset_index()
dfhydsrf = dfhydsrf.groupby(['Year','RoundedGridLine','RoundedGridStation']).median().reset_index()

#merge surface base dataframes
left = dfbiosrf
right = dfhydsrf
merge_on = ['Year','RoundedGridLine','RoundedGridStation']

srf_merged_SM = left.merge(right, on=merge_on, how='outer', suffixes=('_bio', '_hydro'))

#fill missing values from hydro with bio values
srf_merged_SM['Temperature'] = srf_merged_SM['Temperature_hydro'].fillna(srf_merged_SM['Temperature_bio'])
srf_merged_SM['Salinity'] = srf_merged_SM['Salinity_hydro'].fillna(srf_merged_SM['Salinity_bio'])
srf_merged_SM['Density'] = srf_merged_SM['Density_hydro'].fillna(srf_merged_SM['Density_bio'])

#fill missing values from bio with hydro values
srf_merged_SM['Latitude'] = srf_merged_SM['Latitude_bio'].fillna(srf_merged_SM['Latitude_hydro'])
srf_merged_SM['Longitude'] = srf_merged_SM['Longitude_bio'].fillna(srf_merged_SM['Longitude_hydro'])
srf_merged_SM['Month'] = srf_merged_SM['Month_bio'].fillna(srf_merged_SM['Month_hydro'])
srf_merged_SM['Day'] = srf_merged_SM['Day_bio'].fillna(srf_merged_SM['Day_hydro'])
srf_merged_SM['Hour'] = srf_merged_SM['Hour_bio'].fillna(srf_merged_SM['Hour_hydro'])

srf_merged_SM = srf_merged_SM.drop(['Temperature_bio',
                                    'Salinity_bio',
                                    'Density_bio',
                                    'Temperature_hydro',
                                    'Salinity_hydro',
                                    'Density_hydro',
                                    'Latitude_bio',
                                    'Latitude_hydro',
                                    'Longitude_bio',
                                    'Longitude_hydro',
                                    'Month_hydro',
                                    'Day_hydro',
                                    'Hour_hydro',
                                    'Month_bio',
                                    'Day_bio',
                                    'Hour_bio',
                                    'Event_hydro',
                                    'Event_bio',], axis=1)

srf_merged_SM = fill_missing_years(srf_merged_SM)
# -----------------------------------------------------------------------------
# Add EDI sea ice data
# -----------------------------------------------------------------------------
srf_merged_SM = srf_merged_SM.merge(dfsi, on=['Year'], how='left')
# -----------------------------------------------------------------------------
# Assign grid regions based on roundedgridline/station numbers (using EDI dataframe as reference)
# -----------------------------------------------------------------------------
tempdf = srf_merged_SM
region_dict = {}

#identify rounded line/stations belonging to regions - add to mapping dictionary
for index, row in refdf.iterrows():
    key = (row['RoundedGridLine'], row['RoundedGridStation'])
    value = {'Region': row['Region'],'IORegion': row['IORegion'],'NSRegion': row['NSRegion']}
    region_dict[key] = value

#use mapping dictionary to assing regions in df
tempdf['Region'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('Region', None), axis=1)
tempdf['IORegion'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('IORegion', None), axis=1)
tempdf['NSRegion'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('NSRegion', None), axis=1)

#replace None instances with nan instances
tempdf['Region'] = tempdf['Region'].replace([None], np.nan)
tempdf['IORegion'] = tempdf['IORegion'].replace([None], np.nan)
tempdf['NSRegion'] = tempdf['NSRegion'].replace([None], np.nan)
# -----------------------------------------------------------------------------
# Change region info from string to float
# -----------------------------------------------------------------------------
NSRegion = tempdf['NSRegion']
IORegion = tempdf['IORegion']
Region = tempdf['Region']

NSRegion[NSRegion == 'N'] = 1
NSRegion[NSRegion == 'S'] = 2
NSRegion[NSRegion == 'FS'] = 3

IORegion[IORegion == 'C'] = 6
IORegion[IORegion == 'Sh'] = 5
IORegion[IORegion == 'Sl'] = 4

Region[Region == 'NC'] = 3
Region[Region == 'NSh'] = 2
Region[Region == 'NSl'] = 1
Region[Region == 'SC'] = 6
Region[Region == 'SSh'] = 5
Region[Region == 'SSl'] = 4
Region[Region == 'FSC'] = 9
Region[Region == 'FSSh'] = 8
Region[Region == 'FSSl'] = 7

NSRegion = tempdf.NSRegion.astype(float)
IORegion = tempdf.IORegion.astype(float)
Region = tempdf.Region.astype(float)

#   4       5       6    
#   -       -       - 
#   Sl      Sh      C   
# 
# |______|_______|_______|__
# |      |       |       | 
# |  1   |   2   |   3   |     N  - 1
# |______|_______|_______|__
# |      |       |       |
# |  4   |   5   |   6   |     S  - 2
# |______|_______|_______|__
# |      |       |       |
# |  7   |   8   |   9   |     FS - 3
# |______|_______|_______|__
#   ^Numeric representation of grid regions

tempdf['Region'] = tempdf['Region'].astype(float)
tempdf['NSRegion'] = tempdf['NSRegion'].astype(float)
tempdf['IORegion'] = tempdf['IORegion'].astype(float)

srf_merged_SM = tempdf
# -----------------------------------------------------------------------------
# Save surface averaged bio data
# -----------------------------------------------------------------------------
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_StationLevel_SurfaceAvgCoreDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

srf_merged_SM.to_csv(savepath, index=False)
# =============================================================================


# =============================================================================
# Build Event-Level Depth-Averaged Core Dataframe
# =============================================================================
#reduce base dataframes to event-level
dfbiodep = dfbiodep1.groupby(['Year','RoundedGridLine','RoundedGridStation','Event', 'Depth']).median().reset_index()
dfhyddep = dfhyddep1.groupby(['Year','RoundedGridLine','RoundedGridStation','Event', 'Depth']).median().reset_index()

#merge base dataframes
left = dfbiodep
right = dfhyddep
merge_on = ['Year','RoundedGridLine','RoundedGridStation','Event', 'Depth']

dep_merged_EM = left.merge(right, on=merge_on, how='outer', suffixes=('_bio', '_hydro'))

#fill missing values from hydro with bio values
dep_merged_EM['Temperature'] = dep_merged_EM['Temperature_hydro'].fillna(dep_merged_EM['Temperature_bio'])
dep_merged_EM['Salinity'] = dep_merged_EM['Salinity_hydro'].fillna(dep_merged_EM['Salinity_bio'])
dep_merged_EM['Density'] = dep_merged_EM['Density_hydro'].fillna(dep_merged_EM['Density_bio'])

#fill missing values from bio with hydro values
dep_merged_EM['Latitude'] = dep_merged_EM['Latitude_bio'].fillna(dep_merged_EM['Latitude_hydro'])
dep_merged_EM['Longitude'] = dep_merged_EM['Longitude_bio'].fillna(dep_merged_EM['Longitude_hydro'])
dep_merged_EM['Month'] = dep_merged_EM['Month_bio'].fillna(dep_merged_EM['Month_hydro'])
dep_merged_EM['Day'] = dep_merged_EM['Day_bio'].fillna(dep_merged_EM['Day_hydro'])
dep_merged_EM['Hour'] = dep_merged_EM['Hour_bio'].fillna(dep_merged_EM['Hour_hydro'])

dep_merged_EM = dep_merged_EM.drop(['Temperature_bio',
                                    'Salinity_bio',
                                    'Density_bio',
                                    'Temperature_hydro',
                                    'Salinity_hydro',
                                    'Density_hydro',
                                    'Latitude_bio',
                                    'Latitude_hydro',
                                    'Longitude_bio',
                                    'Longitude_hydro',
                                    'Month_hydro',
                                    'Day_hydro',
                                    'Hour_hydro',
                                    'Month_bio',
                                    'Day_bio',
                                    'Hour_bio'], axis=1)

dep_merged_EM = fill_missing_years(dep_merged_EM)
# -----------------------------------------------------------------------------
# Add EDI sea ice data
# -----------------------------------------------------------------------------
dep_merged_EM = dep_merged_EM.merge(dfsi, on=['Year'], how='left')
# -----------------------------------------------------------------------------
# Assign grid regions based on roundedgridline/station numbers (using EDI dataframe as reference)
# -----------------------------------------------------------------------------
tempdf = dep_merged_EM
region_dict = {}

#identify rounded line/stations belonging to regions - add to mapping dictionary
for index, row in refdf.iterrows():
    key = (row['RoundedGridLine'], row['RoundedGridStation'])
    value = {'Region': row['Region'],'IORegion': row['IORegion'],'NSRegion': row['NSRegion']}
    region_dict[key] = value

#use mapping dictionary to assing regions in df
tempdf['Region'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('Region', None), axis=1)
tempdf['IORegion'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('IORegion', None), axis=1)
tempdf['NSRegion'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('NSRegion', None), axis=1)

#replace None instances with nan instances
tempdf['Region'] = tempdf['Region'].replace([None], np.nan)
tempdf['IORegion'] = tempdf['IORegion'].replace([None], np.nan)
tempdf['NSRegion'] = tempdf['NSRegion'].replace([None], np.nan)
# -----------------------------------------------------------------------------
# Change region info from string to float
# -----------------------------------------------------------------------------
NSRegion = tempdf['NSRegion']
IORegion = tempdf['IORegion']
Region = tempdf['Region']

NSRegion[NSRegion == 'N'] = 1
NSRegion[NSRegion == 'S'] = 2
NSRegion[NSRegion == 'FS'] = 3

IORegion[IORegion == 'C'] = 6
IORegion[IORegion == 'Sh'] = 5
IORegion[IORegion == 'Sl'] = 4

Region[Region == 'NC'] = 3
Region[Region == 'NSh'] = 2
Region[Region == 'NSl'] = 1
Region[Region == 'SC'] = 6
Region[Region == 'SSh'] = 5
Region[Region == 'SSl'] = 4
Region[Region == 'FSC'] = 9
Region[Region == 'FSSh'] = 8
Region[Region == 'FSSl'] = 7

NSRegion = tempdf.NSRegion.astype(float)
IORegion = tempdf.IORegion.astype(float)
Region = tempdf.Region.astype(float)

#   4       5       6    
#   -       -       - 
#   Sl      Sh      C   
# 
# |______|_______|_______|__
# |      |       |       | 
# |  1   |   2   |   3   |     N  - 1
# |______|_______|_______|__
# |      |       |       |
# |  4   |   5   |   6   |     S  - 2
# |______|_______|_______|__
# |      |       |       |
# |  7   |   8   |   9   |     FS - 3
# |______|_______|_______|__
#   ^Numeric representation of grid regions

tempdf['Region'] = tempdf['Region'].astype(float)
tempdf['NSRegion'] = tempdf['NSRegion'].astype(float)
tempdf['IORegion'] = tempdf['IORegion'].astype(float)

dep_merged_EM = tempdf
# -----------------------------------------------------------------------------
# Save surface averaged bio data
# -----------------------------------------------------------------------------
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EventLevel_DepthAvgCoreDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

dep_merged_EM.to_csv(savepath, index=False)
# =============================================================================


# =============================================================================
# Build Station-Level Depth-Averaged Core Dataframe
# =============================================================================
#reduce base dataframes to station-level
dfbiodep = dfbiodep1.groupby(['Year','RoundedGridLine','RoundedGridStation','Event', 'Depth']).median().reset_index()
dfbiodep = dfbiodep.groupby(['Year','RoundedGridLine','RoundedGridStation', 'Depth']).median().reset_index()

dfhyddep = dfhyddep1.groupby(['Year','RoundedGridLine','RoundedGridStation','Event', 'Depth']).median().reset_index()
dfhyddep = dfhyddep.groupby(['Year','RoundedGridLine','RoundedGridStation', 'Depth']).median().reset_index()

#merge base dataframes
left = dfbiodep
right = dfhyddep
merge_on = ['Year','RoundedGridLine','RoundedGridStation', 'Depth']

dep_merged_SM = left.merge(right, on=merge_on, how='outer', suffixes=('_bio', '_hydro'))

#fill missing values from hydro with bio values
dep_merged_SM['Temperature'] = dep_merged_SM['Temperature_hydro'].fillna(dep_merged_SM['Temperature_bio'])
dep_merged_SM['Salinity'] = dep_merged_SM['Salinity_hydro'].fillna(dep_merged_SM['Salinity_bio'])
dep_merged_SM['Density'] = dep_merged_SM['Density_hydro'].fillna(dep_merged_SM['Density_bio'])

#fill missing values from bio with hydro values
dep_merged_SM['Latitude'] = dep_merged_SM['Latitude_bio'].fillna(dep_merged_SM['Latitude_hydro'])
dep_merged_SM['Longitude'] = dep_merged_SM['Longitude_bio'].fillna(dep_merged_SM['Longitude_hydro'])
dep_merged_SM['Month'] = dep_merged_SM['Month_bio'].fillna(dep_merged_SM['Month_hydro'])
dep_merged_SM['Day'] = dep_merged_SM['Day_bio'].fillna(dep_merged_SM['Day_hydro'])
dep_merged_SM['Hour'] = dep_merged_SM['Hour_bio'].fillna(dep_merged_SM['Hour_hydro'])

dep_merged_SM = dep_merged_SM.drop(['Temperature_bio',
                                    'Salinity_bio',
                                    'Density_bio',
                                    'Temperature_hydro',
                                    'Salinity_hydro',
                                    'Density_hydro',
                                    'Latitude_bio',
                                    'Latitude_hydro',
                                    'Longitude_bio',
                                    'Longitude_hydro',
                                    'Month_hydro',
                                    'Day_hydro',
                                    'Hour_hydro',
                                    'Month_bio',
                                    'Day_bio',
                                    'Hour_bio',
                                    'Event_hydro',
                                    'Event_bio',], axis=1)

dep_merged_SM = fill_missing_years(dep_merged_SM)
# -----------------------------------------------------------------------------
# Add EDI sea ice data
# -----------------------------------------------------------------------------
dep_merged_SM = dep_merged_SM.merge(dfsi, on=['Year'], how='left')
# -----------------------------------------------------------------------------
# Assign grid regions based on roundedgridline/station numbers (using EDI dataframe as reference)
# -----------------------------------------------------------------------------
tempdf = dep_merged_SM
region_dict = {}

#identify rounded line/stations belonging to regions - add to mapping dictionary
for index, row in refdf.iterrows():
    key = (row['RoundedGridLine'], row['RoundedGridStation'])
    value = {'Region': row['Region'],'IORegion': row['IORegion'],'NSRegion': row['NSRegion']}
    region_dict[key] = value

#use mapping dictionary to assing regions in df
tempdf['Region'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('Region', None), axis=1)
tempdf['IORegion'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('IORegion', None), axis=1)
tempdf['NSRegion'] = tempdf.apply(lambda row: region_dict.get((row['RoundedGridLine'], row['RoundedGridStation']), {}).get('NSRegion', None), axis=1)

#replace None instances with nan instances
tempdf['Region'] = tempdf['Region'].replace([None], np.nan)
tempdf['IORegion'] = tempdf['IORegion'].replace([None], np.nan)
tempdf['NSRegion'] = tempdf['NSRegion'].replace([None], np.nan)
# -----------------------------------------------------------------------------
# Change region info from string to float
# -----------------------------------------------------------------------------
NSRegion = tempdf['NSRegion']
IORegion = tempdf['IORegion']
Region = tempdf['Region']

NSRegion[NSRegion == 'N'] = 1
NSRegion[NSRegion == 'S'] = 2
NSRegion[NSRegion == 'FS'] = 3

IORegion[IORegion == 'C'] = 6
IORegion[IORegion == 'Sh'] = 5
IORegion[IORegion == 'Sl'] = 4

Region[Region == 'NC'] = 3
Region[Region == 'NSh'] = 2
Region[Region == 'NSl'] = 1
Region[Region == 'SC'] = 6
Region[Region == 'SSh'] = 5
Region[Region == 'SSl'] = 4
Region[Region == 'FSC'] = 9
Region[Region == 'FSSh'] = 8
Region[Region == 'FSSl'] = 7

NSRegion = tempdf.NSRegion.astype(float)
IORegion = tempdf.IORegion.astype(float)
Region = tempdf.Region.astype(float)

#   4       5       6    
#   -       -       - 
#   Sl      Sh      C   
# 
# |______|_______|_______|__
# |      |       |       | 
# |  1   |   2   |   3   |     N  - 1
# |______|_______|_______|__
# |      |       |       |
# |  4   |   5   |   6   |     S  - 2
# |______|_______|_______|__
# |      |       |       |
# |  7   |   8   |   9   |     FS - 3
# |______|_______|_______|__
#   ^Numeric representation of grid regions

tempdf['Region'] = tempdf['Region'].astype(float)
tempdf['NSRegion'] = tempdf['NSRegion'].astype(float)
tempdf['IORegion'] = tempdf['IORegion'].astype(float)

dep_merged_SM = tempdf
# -----------------------------------------------------------------------------
# Save surface averaged bio data
# -----------------------------------------------------------------------------
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_StationLevel_DepthAvgCoreDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

dep_merged_SM.to_csv(savepath, index=False)
# =============================================================================
