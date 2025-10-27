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
def filter_dataframe(df):
    #get the columns of the dataframe
    columns = df.columns
    
    #filter by year
    years = np.arange(1991, 2020 + 1)
    df = df[df['Year'].isin(years)].reset_index(drop=True)
    
    #filter by month/day
    if 'Month' in columns and 'Day' in columns:
        df = df.loc[(df['Month'] == 12) | (df['Month'] == 1) | (df['Month'] == 2)]
            #^only want January and bordering months
        df = df.loc[(df['Month'] == 2) & (df['Day'] <= 7) | (df['Month'] == 12) & (df['Day'] >= 24) | (df['Month'] == 1)]
            #^only want days in bordering months that are within a week of January
    
    #filter by grid line (north & south only)
    if 'RoundedGridLine' in columns:
        df = df[df['RoundedGridLine'].between(200, 600, inclusive='both')]
    
    #filter by grid station
    if 'RoundedGridStation' in columns:
        df = df[df['RoundedGridStation'].between(-100, 260, inclusive='both')]
    
    #filter by depth
    if 'Depth' in columns:
        df = df[df['Depth'] <= 200]
    
    df = df.reset_index(drop=True)
    
    return df

def depth_calcs(group, depth_col='Depth', missing_data_strategy='drop', 
          duplicate_depth_strategy='average', operation='median', 
          depth_lim='mld', depth_val=100):
    """
    Processes the depth profile data by either integrating, summing, or
    averaging the specified parameters.

    :param group: DataFrame containing the depth profile data.
    :param depth_col: Name of the column containing depth data.
    :param missing_data_strategy: Strategy for handling missing data ('drop', 'interpolate', 'skip').
    :param duplicate_depth_strategy: Strategy for handling duplicate depths ('drop', 'average').
    :param operation: Operation to perform on the data ('integrate', 'median', 'mean', 'sum').
    :return: Series containing the processed values for all parameters.
    """
    
    metaparams = ['Year', 'Cruise', 'Event', 'RoundedGridLine', 'RoundedGridStation', 
                  'Depth_Bin', 'SelDepth', 'Depth']
    parameters = [col for col in group.columns if col not in metaparams]
    
    result = {}
    
    # Filter based on Depth
    group = group.sort_values(by=depth_col)
    groupqi = group['QI'].median()
    groupmld = group['MLD'].median()
    
    if depth_lim == 'none':
        group = group
    elif depth_lim == 'static':
        depth = depth_val
        group = group[group[depth_col] <= depth].reset_index(drop=True)
    elif depth_lim == 'mld':
        if groupqi >= 0.5 and groupmld >= 2.5: # if MLD is available, use that; otherwise use time series median MLD
            depth = groupmld
        else:
            depth = 40 # median MLD for time series
        group = group[group[depth_col] <= depth].reset_index(drop=True)

    # Handle duplicate depths according to the chosen strategy
    if duplicate_depth_strategy == 'drop':
        group = group.drop_duplicates(subset=depth_col)
    elif duplicate_depth_strategy == 'average':
        group = group.groupby(depth_col, as_index=False)[parameters].mean()

    # Perform the specified operation for each parameter independently
    for param in parameters:
        if param in group.columns:
            sub_group = group[[depth_col, param]].copy()

            # Handle missing data according to the chosen strategy
            if missing_data_strategy == 'drop':
                sub_group = sub_group.dropna()
            elif missing_data_strategy == 'interpolate':
                sub_group[param] = sub_group[param].interpolate(limit_direction='both')
            elif missing_data_strategy == 'skip':
                pass  # trapz will ignore NaN values by default

            y_values = sub_group[param]
            x_values = sub_group[depth_col]
            
            # Ensure that there is data to perform the operation
            if not y_values.isna().all():
                if operation == 'integrate':
                    result[f'{param}'] = trapz(y=y_values, x=x_values)
                elif operation == 'sum':
                    result[f'{param}'] = y_values.sum()
                elif operation == 'median':
                    result[f'{param}'] = y_values.median()
                elif operation == 'mean':
                    result[f'{param}'] = y_values.mean()
            else:
                result[f'{param}'] = np.nan

    return pd.Series(result)

def fill_missing_years(df):    
    all_years = np.arange(1991, 2021)
    existing_years = df['Year'].unique()
    missing_years = np.setdiff1d(all_years, existing_years)
    missing_data = pd.DataFrame({'Year': missing_years})
    missing_data = missing_data.assign(**{column: np.nan for column in df.columns if column != 'Year'})
    df = pd.concat([df, missing_data], ignore_index=True)
    df = df.sort_values('Year')
    return df

def replace_outliers_with_nan(df, parameter_thresholds, params):
    """
    Replace values in specified DataFrame columns that fall outside given thresholds with NaN.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - parameter_thresholds (dict): A dictionary with parameter names as keys and tuples as values,
                                   where each tuple contains (low_threshold, high_threshold).
    - params (list): A list of parameter names (columns in df) to check for threshold violations.
    
    Returns:
    - pd.DataFrame: A DataFrame with values outside the thresholds replaced with NaN.
    """
    # Ensure working on a copy of the DataFrame to avoid modifying the original unintentionally
    modified_df = df.copy()
    
    # Iterate over each parameter specified
    for param in params:
        if param in parameter_thresholds and param in modified_df.columns:
            # Extract the thresholds
            low_threshold, high_threshold = parameter_thresholds[param]
            
            # Replace values outside the thresholds with NaN
            modified_df.loc[(modified_df[param] < low_threshold) | (modified_df[param] > high_threshold), param] = np.nan
    
    return modified_df

def replace_uncertain_mixedlayers(df):
    """
    Replace 'MLD' and 'max_N2' values with NaN in rows where 'QI' is less than 0.5.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    
    Returns:
    - pd.DataFrame: The modified DataFrame with 'MLD' and 'max_N2' replaced by NaN where 'QI' < 0.5.
    """
    # Copy the DataFrame to avoid modifying the original data
    modified_df = df.copy()

    # Find rows where 'QI' is less than 0.5
    condition = modified_df['QI'] < 0.5

    # Replace 'MLD' and 'max_N2' in these rows with NaN
    modified_df.loc[condition, ['MLD', 'max_N2']] = np.nan
    # modified_df.loc[condition, ['MLD']] = np.nan

    return modified_df
# =============================================================================


# =============================================================================
# Load Rutgers Cruise CTD Master Dataframe
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_RutgersCruiseCTDMasterDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)

df = pd.read_csv(loadpath)
# =============================================================================


# =============================================================================
# Conduct High-level Filtering of Dataframes
# =============================================================================
df = filter_dataframe(df)

# # drop sample points with MaxDepth below a threshold
# grouped = df.groupby(['Cruise', 'Year', 'Event']).agg({'Depth': 'max'})
# df = df.join(grouped, on=['Cruise', 'Year', 'Event'], rsuffix='_max')
# df = df.rename(columns={'Depth_max': 'MaxDepth'})
# df = df.reset_index(drop=True)
# df = df[df['MaxDepth'] >= 195].reset_index(drop=True)

#replace parameter outliers with nans
parameter_thresholds = {
    'MLD': (2.5, 200),
    'QI': (0, 1.0),
    'max_N2': (0, 0.0075),
    
    'WWUpper': (0, 200),
    'WWLower': (0, 200),
    'WWThickness': (0, 200),
    'WW%Obs': (0, 100),
    'WWMinTemp': (-5, -1.2),
    'WWMinTempDepth': (0, 200),
                
    'Temperature': (-5, 5), 
    'Salinity': (30, 36),
    'Density': (1025, 1030),
    'SigmaTheta': (25, 30), 
    'Fluorescence': (0, 9999),
    'BeamTransmission': (0, 9999),
    'Oxygen': (0, 9999),
    'PAR': (0, 9999999),    
                
    'Temp_SRF': (-5, 5), 
    'Sal_SRF': (30, 36),
    'Dens_SRF': (1025, 1030),
    'SigTheta_SRF': (25, 30),
    'Fluor_SRF': (0, 9999),
    'BeamTrans_SRF': (0, 9999),
    'Oxygen_SRF': (0, 9999),
    'PAR_SRF': (0, 9999999),   
                
    'Temp_ML': (-5, 5), 
    'Sal_ML': (30, 36),
    'Dens_ML': (1025, 1030),
    'SigTheta_ML': (25, 30),
    'Fluor_ML': (0, 9999),
    'BeamTrans_ML': (0, 9999),
    'Oxygen_ML': (0, 9999),
    'PAR_ML': (0, 9999999), 
    
    'MinTemp': (-5, 5),
    'MinTempDepth': (0, 200)}
filtparams = list(parameter_thresholds.keys())
df = replace_outliers_with_nan(df, parameter_thresholds, filtparams)
# =============================================================================


# # =============================================================================
# # Assign StandardLat and StandardLon from Reference Grid Points
# # =============================================================================
# absolute_path = Path("C:/Users/diouc/Dropbox/Data + Data Analysis/Code + Projects/PhD Dissertation/MS1 - Long Term Pigments/Data Handling - L&O/local data/")
# filename = Path("PALLTER_CruiseStandardGridPointCoordinates.xlsx")
# loadpath = str(absolute_path / filename)

# #load and define grid point coordinates for reference
# gridref = pd.read_excel(loadpath, dtype='str')
# gridref = gridref.astype('float64')

# temp = df[['RoundedGridLine','RoundedGridStation']]
# temp['StandardLat'] = ' '
# temp['StandardLon'] = ' '

# #assign standard lat/lon values based on rounded grid/line numbers
# length = np.arange(0,len(temp),1)
# for r in length:
#     gl = temp.loc[r,'RoundedGridLine']
#     gs = temp.loc[r,'RoundedGridStation']
#     refrow = gridref.loc[(gridref['GridLine'] == gl) & (gridref['GridStation'] == gs)]
#     slat = refrow['StandardLat'].reset_index(drop=True).iloc[0]
#     slon = refrow['StandardLon'].reset_index(drop=True).iloc[0]
#     temp.at[r,'StandardLat'] = slat
#     temp.at[r,'StandardLon'] = slon

# #change to float values
# temp = temp.astype('float64')

# temp2 = temp[['StandardLat','StandardLon']]
# df = pd.concat([df,temp2], axis=1)
# # =============================================================================


# =============================================================================
# Calculate Event Level WW Layer Properties
# =============================================================================
df['WWUpper'] = np.nan #shallowest depth of -1.2C threshold
df['WWLower'] = np.nan #deepest depth of -1.2C threshold
df['WWThickness'] = np.nan #depth difference between lower and upper
df['WWPresence'] = np.nan #whether WW was found (0 = no, 1 = yes)

df['WWMinTemp'] = np.nan #winter water minimum temperature
df['WWMedTemp'] = np.nan #winter water median temperature 
df['WWMedSal'] = np.nan #winter water median salinity 
df['WWMedDens'] = np.nan #winter water median density 

df['WWMinTempDepth'] = np.nan #winter water minimum temperature depth
# df['WWMinTempSal'] = np.nan #salinity at winter water minimum temperature depth
df['MinTemp'] = np.nan #minimum temperature
df['MinTempDepth'] = np.nan #minimum temperature depth

df['BttmTemp'] = np.nan #median temperature between 195-200m depth
df['BttmSal'] = np.nan #median salinity between 195-200m depth
df['BttmDens'] = np.nan #median density between 195-200m depth

grouped = df.groupby(['Year', 'Cruise', 'Event'])

for _, group in grouped:
    indices = group.index
    ww_data = group[group['Temperature'] <= -1.2]
    bttm_data = group[group['Depth'] >= 195] #bottom 5m (195m-200m)

    if not ww_data.empty:
        ww_upper = ww_data['Depth'].min()
        ww_lower = ww_data['Depth'].max()
        ww_thickness = ww_lower - ww_upper
        
        wwlayer_data = ww_data[ww_data['Depth'] >= ww_upper]
        wwlayer_data = ww_data[ww_data['Depth'] <= ww_lower]
        ww_medtemp = wwlayer_data['Temperature'].median()
        ww_medsal = wwlayer_data['Salinity'].median()
        ww_meddens = wwlayer_data['Density'].median()

        df.loc[indices, 'WWUpper'] = ww_upper
        df.loc[indices, 'WWLower'] = ww_lower
        df.loc[indices, 'WWThickness'] = ww_thickness
        df.loc[indices, 'WWPresence'] = 1
        df.loc[indices, 'WWMedTemp'] = ww_medtemp
        df.loc[indices, 'WWMedSal'] = ww_medsal
        df.loc[indices, 'WWMedDens'] = ww_meddens
    else:
        df.loc[indices, 'WWThickness'] = np.nan
        df.loc[indices, ['WWUpper', 'WWLower']] = np.nan
        df.loc[indices, 'WWPresence'] = 0
        df.loc[indices, 'WWMedTemp'] = np.nan
        df.loc[indices, 'WWMedSal'] = np.nan
        df.loc[indices, 'WWMedDens'] = np.nan
    
    if not bttm_data.empty:
        bttm_medtemp = bttm_data['Temperature'].median()
        bttm_medsal = bttm_data['Salinity'].median()
        bttm_meddens = bttm_data['Density'].median()
        df.loc[indices, 'BttmTemp'] = bttm_medtemp
        df.loc[indices, 'BttmSal'] = bttm_medsal
        df.loc[indices, 'BttmDens'] = bttm_meddens
    else:
        df.loc[indices, 'BttmTemp'] = np.nan
        df.loc[indices, 'BttmSal'] = np.nan
        df.loc[indices, 'BttmDens'] = np.nan

for _, group in grouped:
    indices = group.index
    ww_data = group[group['Temperature'] <= -1.2]

    if not ww_data.empty:
        wwmin_temp = ww_data['Temperature'].min()
        wwmin_temp_depth = ww_data.loc[ww_data['Temperature'].idxmin()]['Depth']
        # wwmin_temp_sal = ww_data.loc[ww_data['Temperature'].idxmin()]['Salinity']
        
        df.loc[indices, 'WWMinTemp'] = wwmin_temp
        df.loc[indices, 'WWMinTempDepth'] = wwmin_temp_depth
        # df.loc[indices, 'WWMinTempSal'] = wwmin_temp_sal
    else:
        df.loc[indices, 'WWMinTemp'] = np.nan
        df.loc[indices, 'WWMinTempDepth'] = np.nan
        # df.loc[indices, 'WWMinTempSal'] = np.nan

for _, group in grouped:
    indices = group.index
    ww_data = group[['Temperature', 'Depth']]

    if not ww_data.empty:
        min_temp = ww_data['Temperature'].min()
        min_temp_depth = ww_data.loc[ww_data['Temperature'].idxmin()]['Depth']
        
        df.loc[indices, 'MinTemp'] = min_temp
        df.loc[indices, 'MinTempDepth'] = min_temp_depth
    else:
        df.loc[indices, 'MinTemp'] = np.nan
        df.loc[indices, 'MinTempDepth'] = np.nan
# =============================================================================


# =============================================================================
# Calculate and Insert Yearly WW%Observations from Station-Level
# =============================================================================
#calculate event-level medians (event-level WW variables)
df_EM = df.groupby(['Year','RoundedGridLine','RoundedGridStation','Event']).median().reset_index()

#calculate station-level medians (station-level WW variables)
df_SM = df_EM.groupby(['Year','RoundedGridLine','RoundedGridStation']).median().reset_index()

#calculate yearly medians
df_YM = df_SM.groupby(['Year']).median().reset_index()

#calculate total number of stations sampled per year for temperature 
stationcount = df_SM.groupby(['Year'])['Temperature'].count().reset_index()

#calculate total number of stations per year where WW was found
wwcount = df_SM.groupby(['Year'])['WWUpper'].count().reset_index()

#calculate percent stations per year where WW was observed
df_YM['WW%Obs'] = (wwcount['WWUpper']/stationcount['Temperature'])*100

#merge WW%Obs back into main df
df = df.merge(df_YM[['Year', 'WW%Obs']], on='Year', how='left')
# =============================================================================


# =============================================================================
# Change Region Info from String to Float
# =============================================================================
NSRegion = df['NSRegion']
IORegion = df['IORegion']
Region = df['Region']

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

NSRegion = df.NSRegion.astype(float)
IORegion = df.IORegion.astype(float)
Region = df.Region.astype(float)

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
# 
#   ^Numeric representation of grid regions
# =============================================================================


# =============================================================================
# Compile Surface Average, Event-Level Dataframe
# =============================================================================
# -----------------------------------------------------------------------------
# Group by event and get surface median values
# -----------------------------------------------------------------------------
grouping = ['Year','RoundedGridLine','RoundedGridStation','Event']
df_srf_EM = df.groupby(grouping).apply(depth_calcs, 
                                       depth_col='Depth',
                                        missing_data_strategy='drop',
                                        duplicate_depth_strategy='average',
                                        operation='median', 
                                        depth_lim='static', 
                                        depth_val=7).reset_index()
# -----------------------------------------------------------------------------
# Replace outliers (and uncertain MLD estimates)
# -----------------------------------------------------------------------------
filtparams = list(parameter_thresholds.keys())
df_srf_EM = replace_outliers_with_nan(df_srf_EM, parameter_thresholds, filtparams)
df_srf_EM = replace_uncertain_mixedlayers(df_srf_EM)
# -----------------------------------------------------------------------------
# Fill in missing years with nans 
# -----------------------------------------------------------------------------
df_srf_EM = fill_missing_years(df_srf_EM)
# -----------------------------------------------------------------------------
# Save surface averaged bio data
# -----------------------------------------------------------------------------
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EventLevel_SurfaceAvgHydroDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

df_srf_EM.to_csv(savepath, index=False)
# =============================================================================


# =============================================================================
# Compile Depth-Averaged, Event-Level Dataframe
# =============================================================================
# -----------------------------------------------------------------------------
# Group by depth to get depth-resolved median values
# -----------------------------------------------------------------------------
grouping = ['Year','RoundedGridLine','RoundedGridStation','Event','Depth']
df_depth_EM = df.groupby(grouping).median().reset_index()

# # drop sample points with MaxDepth below a threshold
# grouped = df_depth_EM.groupby(['Year', 'Event']).agg({'Depth': 'max'})
# df_depth_EM = df_depth_EM.join(grouped, on=['Year', 'Event'], rsuffix='_max')
# df_depth_EM = df_depth_EM.rename(columns={'Depth_max': 'MaxDepth'})
# df_depth_EM = df_depth_EM.reset_index(drop=True)
# df_depth_EM = df_depth_EM[df_depth_EM['MaxDepth'] >= 195].reset_index(drop=True)
# -----------------------------------------------------------------------------
# Calculate and insert event-level, surface medians
# -----------------------------------------------------------------------------
#calculate surface medians
grouping = ['Year','RoundedGridLine','RoundedGridStation','Event']
df_srf_EM2 = df_depth_EM.groupby(grouping).apply(depth_calcs, 
                                       depth_col='Depth',
                                        missing_data_strategy='drop',
                                        duplicate_depth_strategy='average',
                                        operation='median', 
                                        depth_lim='static', 
                                        depth_val=7).reset_index()

#rename surface median columns (for easy identification when inserted back into df)
df_srf_EM2 = df_srf_EM2.rename(columns={'Salinity': 'Sal_SRF', 
                                      'Temperature': 'Temp_SRF',
                                      'Density': 'Dens_SRF',
                                      'SigmaTheta': 'SigTheta_SRF',
                                      'Fluorescence': 'Fluor_SRF',
                                      'BeamTransmission': 'BeamTrans_SRF',
                                      'Oxygen': 'Oxygen_SRF',
                                      'PAR': 'PAR_SRF'})

#select columns to insert back into df
df_srf_EM2 = df_srf_EM2[['Year','RoundedGridLine','RoundedGridStation','Event',
                         'Sal_SRF','Temp_SRF','Dens_SRF','SigTheta_SRF','Fluor_SRF',
                         'BeamTrans_SRF','Oxygen_SRF','PAR_SRF']].reset_index(drop=True)

#insert columns back into main df
df_depth_EM = df_depth_EM.merge(df_srf_EM2, on=grouping, how='left')
# # -----------------------------------------------------------------------------
# # Calculate and insert event-level, mixed layer medians
# # -----------------------------------------------------------------------------
# df_ml = df_depth_EM[['Year','RoundedGridLine','RoundedGridStation','Event',
#                      'Depth','MLD','QI','Salinity','Temperature','Density',
#                      'SigmaTheta','Fluorescence','BeamTransmission','Oxygen',
#                      'PAR']].reset_index(drop=True)

# #calculate mixed layer medians (MLD depth-lim if present, otherwise 40m depth-lim)
# grouping = ['Year','RoundedGridLine','RoundedGridStation','Event']
# df_ml_EM = df_ml.groupby(grouping).apply(depth_calcs, 
#                                        depth_col='Depth',
#                                         missing_data_strategy='drop',
#                                         duplicate_depth_strategy='average',
#                                         operation='median', 
#                                         depth_lim='mld', 
#                                         depth_val=7).reset_index()

# #rename surface median columns (for easy identification when inserted back into df)
# df_ml_EM = df_ml_EM.rename(columns={'Salinity': 'Sal_ML', 
#                                     'Temperature': 'Temp_ML',
#                                     'Density': 'Dens_ML',
#                                     'SigmaTheta': 'SigTheta_ML',
#                                     'Fluorescence': 'Fluor_ML',
#                                     'BeamTransmission': 'BeamTrans_ML',
#                                     'Oxygen': 'Oxygen_ML',
#                                     'PAR': 'PAR_ML'})

# #select columns to insert back into df
# df_ml_EM = df_ml_EM[['Year','RoundedGridLine','RoundedGridStation','Event',
#                      'Sal_ML','Temp_ML','Dens_ML','SigTheta_ML','Fluor_ML',
#                      'BeamTrans_ML','Oxygen_ML','PAR_ML']].reset_index(drop=True)

# #insert columns back into main df
# df_depth_EM = df_depth_EM.merge(df_ml_EM, on=grouping, how='left')
# -----------------------------------------------------------------------------
# Replace outliers (and uncertain MLD estimates)
# -----------------------------------------------------------------------------
filtparams = list(parameter_thresholds.keys())
df_depth_EM = replace_outliers_with_nan(df_depth_EM, parameter_thresholds, filtparams)
df_depth_EM = replace_uncertain_mixedlayers(df_depth_EM)
# -----------------------------------------------------------------------------
# Fill in missing years with nans 
# -----------------------------------------------------------------------------
df_depth_EM = fill_missing_years(df_depth_EM)
# -----------------------------------------------------------------------------
# Save surface averaged bio data
# -----------------------------------------------------------------------------
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EventLevel_DepthAvgHydroDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

df_depth_EM.to_csv(savepath, index=False)
# =============================================================================