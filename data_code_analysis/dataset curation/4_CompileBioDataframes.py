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

"""
******
"""

# =============================================================================
# Functions
# =============================================================================
def pigment_sums(df):
    """
    This function calculates HPLC pigment sums based on Table 4 & Table 5 of
    Hooker at al. (2012) [The Fifth SeaWiFS HPLC Analysis Round Robin Experiment 
    (SeaHARRE-5), NASA Technical Memorandum 2012–217503]
    
    Due to the fact that Chlorophyllc1 and Diviynl Chlorophyll-b were not measured 
    at all in the LTER, relevant sums were modified to exclude these pigments in 
    their calculations
    
    Due to the nature of column addition in pandas, sums that lack non-null
    values for all necessary pigments are calculated as NaN. If this stringincy
    is not required, sums can be calulcated using .sum() as shown below:
        df['Caro'] = df[['Betacarotene','Alphacarotene']].sum(skipna=True, min_count=2)
    OR using .filna(0):
        df['Caro'] =  df['Betacarotene'].fillna(0) + df['Alphacarotene'].fillna(0)

    Parameters
    ----------
    df : pandas.Dataframe
        A dataframe containing HPLC concentrations of necessary pigments

    Returns
    -------
    None.

    """
    df['TChla'] = df['Chlorophyllide'] + df['DivinylChlorophylla'] + df['Chlorophylla']
    df['TChlb'] = df['Chlorophyllb'] #DVChlb removed from sum, not measured in dataset
    df['TChlc'] = df['Chlorophyllc2'] + df['Chlorophyllc3'] #Chl_C1 removed from sum, not measured in dataset
    df['TChl'] = df['TChla'] + df['TChlb'] + df['TChlc']
    
    df['Caro'] = df['Betacarotene'] #alternative Caro calc only requiring BCaro
    # df['Caro'] = df['Betacarotene'] + df['Alphacarotene'] #standard Caro calc
    df['PPC'] = df['Alloxanthin'] + df['Diadinoxanthin'] + df['Diatoxanthin'] + df['Zeaxanthin'] + df['Caro']
    df['PSC'] = df['x19butanoyloxyfucoxanthin'] + df['Fucoxanthin'] + df['x19hexanoyloxyfucoxanthin'] + df['Peridinin']
    df['TCaro'] = df['PPC'] + df['PSC']
    df['TAcc'] = df['TCaro'] + df['TChlb'] + df['TChlc']
    df['TPig'] = df['TAcc'] + df['Chlorophylla']
    df['TPig2'] = df['TAcc'] + df['TChla']
    df['PSP'] = df['PSC'] + df['TChl']
    
    df['PrimPPC'] = df['Alloxanthin'] + df['Diadinoxanthin'] + df['Caro']
    df['SecPPC'] = df['Diatoxanthin'] + df['Zeaxanthin']
    df['Xanth'] = df['Diadinoxanthin'] + df['Diatoxanthin'] + df['Violaxanthin'] + df['Zeaxanthin']
    # df['Xanth'] = df['Diadinoxanthin'] + df['Diatoxanthin'] + df['Violaxanthin'] + df['Antheraxanthin'] + df['Zeaxanthin']
    df['PrimXanth'] = df['Diadinoxanthin'] + df['Violaxanthin']
    # df['SecXanth'] = df['Diatoxanthin'] + df['Antheraxanthin'] + df['Zeaxanthin']
    df['SecXanth'] = df['Diatoxanthin'] + df['Zeaxanthin']
    df['DiaXanth'] = df['Diadinoxanthin'] + df['Diatoxanthin']
    
    df['DP'] = df['PSC'] + df['Alloxanthin'] + df['Zeaxanthin'] + df['TChlb']

def pigment_ratios(df):
    """
    This function calculates HPLC pigment ratios based on Table 4 & Table 5 of
    Hooker at al. (2012) [The Fifth SeaWiFS HPLC Analysis Round Robin Experiment 
    (SeaHARRE-5), NASA Technical Memorandum 2012–217503]
    
    Parameters
    ----------
    df : pandas.Dataframe
        A dataframe containing HPLC concentrations of necessary pigments and
        pigment sums

    Returns
    -------
    None.

    """
    # Realtive to Chla
    df['TCaro:Chla'] = df['TCaro'] / df['Chlorophylla']
    df['PPC:Chla'] = df['PPC'] / df['Chlorophylla']
    df['PSC:Chla'] = df['PSC'] / df['Chlorophylla']
    df['Xanth:Chla'] = df['Xanth'] / df['Chlorophylla']
    
    df['PrimPPC:Chla'] = df['PrimPPC'] / df['Chlorophylla']
    df['SecPPC:Chla'] = df['SecPPC'] / df['Chlorophylla']
    df['PrimXanth:Chla'] = df['PrimXanth'] / df['Chlorophylla']
    df['SecXanth:Chla'] = df['SecXanth'] / df['Chlorophylla']
    
    df['Chlb'] = df['Chlorophyllb'] / df['Chlorophylla']
    df['Chlc2'] = df['Chlorophyllc2'] / df['Chlorophylla']
    df['Chlc3'] = df['Chlorophyllc3'] / df['Chlorophylla']
    df['Chlide'] = df['Chlorophyllide'] / df['Chlorophylla']
    df['DVChla'] = df['DivinylChlorophylla'] / df['Chlorophylla']
    df['BCar'] = df['Betacarotene'] / df['Chlorophylla']
    df['ACar'] = df['Alphacarotene'] / df['Chlorophylla']
    df['A+BCar'] = (df['Alphacarotene']+df['Betacarotene']) / df['Chlorophylla']
    df['Allo'] = df['Alloxanthin'] / df['Chlorophylla']
    df['Diadino'] = df['Diadinoxanthin'] / df['Chlorophylla']
    df['Diato'] = df['Diatoxanthin'] / df['Chlorophylla']
    df['DD+DT'] = df['DiaXanth'] / df['Chlorophylla']
    df['Zea'] = df['Zeaxanthin'] / df['Chlorophylla']
    df['Anth'] = df['Antheraxanthin'] / df['Chlorophylla']
    df['Viol'] = df['Violaxanthin'] / df['Chlorophylla']
    df['Fuco'] = df['Fucoxanthin'] / df['Chlorophylla']
    df['Perid'] = df['Peridinin'] / df['Chlorophylla']
    df['But-Fuco'] = df['x19butanoyloxyfucoxanthin'] / df['Chlorophylla']
    df['Hex-Fuco'] = df['x19hexanoyloxyfucoxanthin'] / df['Chlorophylla']
    
    # # Realtive to TChla
    df['TCaro:TChla'] = df['TCaro'] / df['TChla']
    df['PPC:TChla'] = df['PPC'] / df['TChla']
    df['PSC:TChla'] = df['PSC'] / df['TChla']
    df['Xanth:TChla'] = df['Xanth'] / df['TChla']
    
    df['PrimPPC:TChla'] = df['PrimPPC'] / df['TChla']
    df['SecPPC:TChla'] = df['SecPPC'] / df['TChla']
    df['PrimXanth:TChla'] = df['PrimXanth'] / df['TChla']
    df['SecXanth:TChla'] = df['SecXanth'] / df['TChla']
   
    # Realtive to TChl 
    df['TCaro:TChl'] = df['TCaro'] / df['TChl']
    df['PPC:TChl'] = df['PPC'] / df['TChl']
    df['PSC:TChl'] = df['PSC'] / df['TChl']
    df['Xanth:TChl'] = df['Xanth'] / df['TChl']
    
    df['PrimPPC:TChl'] = df['PrimPPC'] / df['TChl']
    df['SecPPC:TChl'] = df['SecPPC'] / df['TChl']
    df['PrimXanth:TChl'] = df['PrimXanth'] / df['TChl']
    df['SecXanth:TChl'] = df['SecXanth'] / df['TChl']
   
    # Realtive to TCaro
    df['PPC:TCaro'] = df['PPC'] / df['TCaro']
    df['PSC:TCaro'] = df['PSC'] / df['TCaro']
    df['Xanth:TCaro'] = df['Xanth'] / df['TCaro']
    
    df['PrimPPC:TCaro'] = df['PrimPPC'] / df['TCaro']
    df['SecPPC:TCaro'] = df['SecPPC'] / df['TCaro']
    df['PrimXanth:TCaro'] = df['PrimXanth'] / df['TCaro']
    df['SecXanth:TCaro'] = df['SecXanth'] / df['TCaro']
    
    # Realtive to TAcc
    df['TCaro:TAcc'] = df['TCaro'] / df['TAcc']
    df['PPC:TAcc'] = df['PPC'] / df['TAcc']
    df['PSC:TAcc'] = df['PSC'] / df['TAcc']
    
    # # Realtive to TPig
    df['TCaro:TPig'] = df['TCaro'] / df['TPig']
    df['PPC:TPig'] = df['PPC'] / df['TPig']
    df['PSC:TPig'] = df['PSC'] / df['TPig']
    df['Chla:TPig'] = df['Chlorophylla'] / df['TPig']
    df['TChla:TPig'] = df['TChla'] / df['TPig']
    df['TAcc:TPig'] = df['TAcc'] / df['TPig']
    
    # # Realtive to PSP
    df['PPC:PSP'] = df['PPC'] / df['PSP']
    df['PSC:PSP'] = df['PSC'] / df['PSP']

    # Realtive to PPC
    df['Allo:PPC'] = df['Alloxanthin'] / df['PPC']
    df['Diadino:PPC'] = df['Diadinoxanthin'] / df['PPC']
    df['Diato:PPC'] = df['Diatoxanthin'] / df['PPC']
    df['Zea:PPC'] = df['Zeaxanthin'] / df['PPC']
    df['BCar:PPC'] = df['Caro'] / df['PPC']
    df['DD+DT:PPC'] = df['DiaXanth'] / df['PPC']
    df['PrimPPC:PPC'] = df['PrimPPC'] / df['PPC']
    df['SecPPC:PPC'] = df['SecPPC'] / df['PPC']
    
    # df['PSC:PPC'] = df['PSC'] / df['PPC']
    df['PPC:PSC'] = df['PPC'] / df['PSC']
    
    # Realtive to PSC
    df['Fuco:PSC'] = df['Fucoxanthin'] / df['PSC']
    df['Perid:PSC'] = df['Peridinin'] / df['PSC']
    df['But-Fuco:PSC'] = df['x19butanoyloxyfucoxanthin'] / df['PSC']
    df['Hex-Fuco:PSC'] = df['x19hexanoyloxyfucoxanthin'] / df['PSC']
    
    df['Fuco:Hex'] = df['Fucoxanthin'] / df['x19hexanoyloxyfucoxanthin']
    
    # Realtive to TCaro
    df['Allo:TCaro'] = df['Alloxanthin'] / df['TCaro']
    df['Diadino:TCaro'] = df['Diadinoxanthin'] / df['TCaro']
    df['Diato:TCaro'] = df['Diatoxanthin'] / df['TCaro']
    df['DD+DT:TCaro'] = df['DiaXanth'] / df['TCaro']
    df['Zea:TCaro'] = df['Zeaxanthin'] / df['TCaro']
    df['BCar:TCaro'] = df['Caro'] / df['TCaro']
    df['Fuco:TCaro'] = df['Fucoxanthin'] / df['TCaro']
    df['Perid:TCaro'] = df['Peridinin'] / df['TCaro']
    df['But-Fuco:TCaro'] = df['x19butanoyloxyfucoxanthin'] / df['TCaro']
    df['Hex-Fuco:TCaro'] = df['x19hexanoyloxyfucoxanthin'] / df['TCaro']
    
    df['Diadino:PSC'] = df['Diadinoxanthin'] / df['PSC']
    df['Diato:PSC'] = df['Diatoxanthin'] / df['PSC']
    df['DD+DT:PSC'] = df['DiaXanth'] / df['PSC']
    
    # Sice Class Indices
    df['mPF'] = (df['Fucoxanthin'] + df['Peridinin']) / df['DP']
    df['nPF'] = (df['x19hexanoyloxyfucoxanthin'] + df['x19butanoyloxyfucoxanthin'] + df['Alloxanthin']) / df['DP']
    df['pPF'] = (df['Zeaxanthin'] + df['TChlb']) / df['DP']
    
    # Other
    # df['PrimXanth:Xanth'] = df['PrimXanth'] / df['Xanth']
    # df['SecXanth:Xanth'] = df['SecXanth'] / df['Xanth']
    # df['DD:DiaXanth'] = df['Diadinoxanthin'] / df['DiaXanth']
    # df['DT:DiaXanth'] = df['Diatoxanthin'] / df['DiaXanth']
    df['TAcc:Chla'] = df['TAcc'] / df['Chlorophylla']
    df['TAcc:TChla'] = df['TAcc'] / df['TChla']
    
    df['Chla:POC'] = df['Chlorophylla'] / df['POC']
    df['TCaro:POC'] = df['TCaro'] / df['POC']
    df['TAcc:POC'] = df['TAcc'] / df['POC']
    df['TPig:POC'] = df['TPig'] / df['POC']

def taxon_biomass(df):
    df['PrasinophyteBiomass'] = df['Chlorophylla'] * df['Prasinophytes']
    df['Type4HaptophyteBiomass'] = df['Chlorophylla'] * df['Type4Haptophytes']
    df['MixedFlagellateBiomass'] = df['Chlorophylla'] * df['MixedFlagellates']
    df['CryptophyteBiomass'] = df['Chlorophylla'] * df['Cryptophytes']
    df['DiatomBiomass'] = df['Chlorophylla'] * df['Diatoms']

def sdi_evenness(data):
    # https://www.omnicalculator.com/ecology/shannon-index#shannon-diversity-index-formula
    # Evenness values near 0 indicate highly uneven taxonomic distribution; 
    # values near 1 indicate very even distributions
    from math import log as ln 
    ds = {'Diatoms': float(data['Diatoms']),
          'Cryptophytes': float(data['Cryptophytes']),
          'MixedFlagellates': float(data['MixedFlagellates']),
          'Type4Haptophytes': float(data['Type4Haptophytes']),
          'Prasinophytes': float(data['Prasinophytes'])}
    def pi(i,N):
        if i == 0:
            return 0
        else:
            return (float(i)/N) * ln(float(i)/N)
    N = sum(ds.values())
    H = -sum(pi(i, N) for i in ds.values() if i != 0)
    E = H / ln(len(ds))
    return abs(E)

def get_sel_depth(depth_bin):
    # Define a function to selected depth tags based on Depth_Bin values
    for depth in selected_depths:
        if abs(depth_bin - depth) <= 2.5:
            return depth
    # If no selected depth is within +/- 2.4 of the Depth_Bin value, return NaN
    return np.nan

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
                  'IORegion', 'NSRegion','Depth_Bin', 'SelDepth', 'Depth']
    parameters = [col for col in group.columns if col not in metaparams]
    
    result = {}
    
    # Filter based on Depth
    group = group.sort_values(by=depth_col)
    # groupqi = group['QI'].median()
    # groupmld = group['MLD'].median()
    
    if depth_lim == 'none':
        group = group
    elif depth_lim == 'static':
        depth = depth_val
        group = group[group[depth_col] <= depth].reset_index(drop=True)
    # elif depth_lim == 'mld':
    #     if groupqi >= 0.5: # if MLD is available, use that; otherwise use time series median MLD
    #         depth = groupmld
    #     else:
    #         depth = 40 # median MLD for time series
    #     group = group[group[depth_col] <= depth].reset_index(drop=True)

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

    return modified_df
# =============================================================================


# =============================================================================
# Load EDI Cruise Dataframe
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EDICruiseDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)

df = pd.read_csv(loadpath)
# =============================================================================


# =============================================================================
# Conduct High-level Filtering of Dataframes
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

df = filter_dataframe(df)
# =============================================================================


# =============================================================================
# Assign StandardLat and StandardLon from Reference Grid Points
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_CruiseStandardGridPointCoordinates.xlsx")
loadpath = str(current_directory / absolute_path / filename)

#load and define grid point coordinates for reference
gridref = pd.read_excel(loadpath, dtype='str')
gridref = gridref.astype('float64')

temp = df[['RoundedGridLine','RoundedGridStation']]
temp['StandardLat'] = ' '
temp['StandardLon'] = ' '

#assign standard lat/lon values based on rounded grid/line numbers
length = np.arange(0,len(temp),1)
for r in length:
    gl = temp.loc[r,'RoundedGridLine']
    gs = temp.loc[r,'RoundedGridStation']
    refrow = gridref.loc[(gridref['GridLine'] == gl) & (gridref['GridStation'] == gs)]
    slat = refrow['StandardLat'].reset_index(drop=True).iloc[0]
    slon = refrow['StandardLon'].reset_index(drop=True).iloc[0]
    temp.at[r,'StandardLat'] = slat
    temp.at[r,'StandardLon'] = slon

#change to float values
temp = temp.astype('float64')

temp2 = temp[['StandardLat','StandardLon']]
df = pd.concat([df,temp2], axis=1)
# =============================================================================


# =============================================================================
# Filter Down Columns
# =============================================================================
contextrefs2 = ['Cruise',
                'Date',
                'Year',
                'Month',
                'Day',
                'Hour',
                'Event',
                'Latitude',
                'Longitude',
                'RoundedGridLine',
                'RoundedGridStation',
                'StandardLat',
                'StandardLon',
                'Region',
                'NSRegion',
                'IORegion',
                'Depth',
                'Temperature',
                'Salinity',
                'Density']

pigments2 = ['Chlorophyll',
             'Phaeopigment',
             'Chlorophylla',
             'Chlorophyllb',
             'Chlorophyllc2',
             'Chlorophyllc3',
             'Chlorophyllide',
             'DivinylChlorophylla',
             'Betacarotene',
             'Alphacarotene',
             'Alloxanthin',
             'Diadinoxanthin',
             'Diatoxanthin',
             'Zeaxanthin',
             'Antheraxanthin',
             'Violaxanthin',
             'Fucoxanthin',
             'Peridinin',
             'x19butanoyloxyfucoxanthin',
             'x19hexanoyloxyfucoxanthin',
             'Lutein',
             'Lycopene',
             'Phaeophytin',
             'Phaeophorbide',
             'Neoxanthin',
             'Prasinoxanthin',
             'Crocoxanthin',
             'Echinenone',]

chemtax2 = ['Prasinophytes',
            'Cryptophytes',
            'MixedFlagellates',
            'Diatoms',
            'Type4Haptophytes']

misc2 = ['PrimaryProduction',
         'PrimProdSTD',
         'FIRESigma',
         'FIRERho',
         'FIRE_FvFm',
         'DOC',
         'POC',
         'PO4',
         'SiO4',
         'NO2',
         'NO3',
         'NO3plusNO2']

corevar = contextrefs2 + pigments2 + chemtax2 + misc2
df = df[corevar].reset_index(drop=True)

#assign depth bins and seldepths based on depth
df['Depth_Bin'] = df['Depth'].round()
selected_depths = np.arange(0,1005,5)
    # ^data is tagged as selected depth if falls within 
    # +/- 2.4m (i.e., 4.8m depth bins centered on selected depths)
df['SelDepth'] = df['Depth_Bin'].apply(get_sel_depth)
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
grouping = ['Year','Region','NSRegion','IORegion',
            'RoundedGridLine','RoundedGridStation','Event']
df_srf_EM = df.groupby(grouping).apply(depth_calcs, 
                                       depth_col='Depth',
                                       missing_data_strategy='drop',
                                       duplicate_depth_strategy='average',
                                       operation='median', 
                                       depth_lim='static', 
                                       depth_val=7).reset_index()
# -----------------------------------------------------------------------------
# Run calculation functions with surface median data
# -----------------------------------------------------------------------------
taxon_biomass(df_srf_EM)

pigment_sums(df_srf_EM)

pigment_ratios(df_srf_EM)

df_srf_EM['Evenness'] = df_srf_EM.apply(lambda row : sdi_evenness(row), axis=1)

df_srf_EM['SpecPrimProd'] = df_srf_EM['PrimaryProduction'] / df_srf_EM['Chlorophylla']
# -----------------------------------------------------------------------------
# Replace outliers (and uncertain MLD estimates)
# -----------------------------------------------------------------------------
parameter_thresholds = {
    'TCaro:Chla': (0, 6), 
    'PSC:Chla': (0, 6),
    'PPC:Chla': (0, 2),
    
    'PrimXanth:Chla': (0, 2),
    'SecXanth:Chla': (0, 2),
    'PrimPPC:Chla': (0, 2),
    'SecPPC:Chla': (0, 2),
    
    'PPC:PSC': (0, 25),
    
    'PPC:TCaro': (0, 1),
    'PSC:TCaro': (0, 1),
    
    'Allo:PPC': (0, 1),
    'Diadino:PPC': (0, 1),
    'Diato:PPC': (0, 1),
    'DD+DT:PPC': (0, 1),
    'Zea:PPC': (0, 1),
    'BCar:PPC': (0, 1),
    'PrimPPC:PPC': (0, 1),
    'SecPPC:PPC': (0, 1),
    
    'Fuco:PSC': (0, 1),
    'Hex-Fuco:PSC': (0, 1),
    'But-Fuco:PSC': (0, 1),
    'Perid:PSC': (0, 1),
    
    'Allo:TCaro': (0, 1),
    'Diadino:TCaro': (0, 1),
    'Diato:TCaro': (0, 1),
    'DD+DT:TCaro': (0, 1),
    'Zea:TCaro': (0, 1),
    'BCar:TCaro': (0, 1),
    'Fuco:TCaro': (0, 1),
    'Hex-Fuco:TCaro': (0, 1),
    'But-Fuco:TCaro': (0, 1),
    'Perid:TCaro': (0, 1),
    
    'Allo': (0, 1.25),
    'Diadino': (0, 1.75),
    'Diato': (0, 0.6),
    'DD+DT': (0, 15),
    'Zea': (0, 0.4),
    'BCar': (0, 0.6),
    'Fuco': (0, 3.5),
    'Hex-Fuco': (0, 4.5),
    'But-Fuco': (0, 2),
    'Perid': (0, 3.25),
    
    'mPF': (0, 1),
    'nPF': (0, 1),
    'pPF': (0, 1),
    
    'Chlorophylla': (0.01, 60),
    'PrimaryProduction': (1, 600),
    'SpecPrimProd': (0.01, 400),
    
    'Diatoms': (0, 1),
    'Cryptophytes': (0, 1),
    'MixedFlagellates': (0, 1),
    'Type4Haptophytes': (0, 1),
    'Prasinophytes': (0, 1),
    
    'DiatomBiomass': (0.001, 60),
    'CryptophyteBiomass': (0.001, 60),
    'MixedFlagellateBiomass': (0.001, 60),
    'Type4HaptophyteBiomass': (0.001, 60),
    'PrasinophyteBiomass': (0.001, 60),
    
    'SiO4': (0, 9999),
    'PO4': (0, 9999),
    'NO2': (0, 0.5),
    'NO3': (0, 9999),
    'NO3plusNO2': (0, 9999),
    'POC': (0, 9999),
    
    'Evenness': (0, 2)}
filtparams = list(parameter_thresholds.keys())
df_srf_EM = replace_outliers_with_nan(df_srf_EM, parameter_thresholds, filtparams)
# -----------------------------------------------------------------------------
# Fill in missing years with nans 
# -----------------------------------------------------------------------------
df_srf_EM = fill_missing_years(df_srf_EM)
# -----------------------------------------------------------------------------
# Save surface averaged bio data
# -----------------------------------------------------------------------------
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EventLevel_SurfaceAvgBioDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

df_srf_EM.to_csv(savepath, index=False)
# =============================================================================


# =============================================================================
# Compile Depth-Averaged, Event-Level Dataframe
# =============================================================================
# -----------------------------------------------------------------------------
# Group by depth bins to get depth-resolved median values
# -----------------------------------------------------------------------------
grouping = ['Year','Region','NSRegion','IORegion',
            'RoundedGridLine','RoundedGridStation','Event','Depth']
df_depth_EM = df.groupby(grouping).median().reset_index()
# -----------------------------------------------------------------------------
# Run calculation functions on all depths
# -----------------------------------------------------------------------------
taxon_biomass(df_depth_EM)

pigment_sums(df_depth_EM)

pigment_ratios(df_depth_EM)

df_depth_EM['Evenness'] = df_depth_EM.apply(lambda row : sdi_evenness(row), axis=1)

df_depth_EM['SpecPrimProd'] = df_depth_EM['PrimaryProduction'] / df_depth_EM['Chlorophylla']
# -----------------------------------------------------------------------------
# Replace outliers (and uncertain MLD estimates)
# -----------------------------------------------------------------------------
filtparams = list(parameter_thresholds.keys())
df_depth_EM = replace_outliers_with_nan(df_depth_EM, parameter_thresholds, filtparams)
# -----------------------------------------------------------------------------
# Fill in missing years with nans 
# -----------------------------------------------------------------------------
df_depth_EM = fill_missing_years(df_depth_EM)
# -----------------------------------------------------------------------------
# Save surface averaged bio data
# -----------------------------------------------------------------------------
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EventLevel_DepthAvgBioDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

df_depth_EM.to_csv(savepath, index=False)
# =============================================================================






