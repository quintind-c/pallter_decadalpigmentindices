# =============================================================================
# Imports 
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy import stats
import warnings
from matplotlib.ticker import ScalarFormatter
warnings.filterwarnings('ignore')

# Set global font properties
fs = 12
plt.rcParams['font.size'] = fs
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.style'] = 'normal'
# =============================================================================

# =============================================================================
# Functions & Reference Dictionaries
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

def get_asterisks(p_value):
    if p_value < 0.0001:
        return '****'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def remove_bad_years(df):
    parambadyears = {
        'MLD': [1995, 2011],
        'QI': [1995],
        'max_N2': [1995, 2011],
        
        'WWUpper': [None],
        'WWLower': [None],
        'WWThickness': [None],
        'WW%Obs': [None],
        'WWMinTemp': [None],
        'WWMinTempDepth': [None],
        'SIAdvance': [None],
        'SIRetreat': [None],
        'SIDuration': [None],
        'IceDays': [None],
        'SIRetrProx': [None],
        'SIExtent': [None],
        'SIArea': [None],
        'OWArea': [None],
        'TotalSIConc': [None],
        
        'Temperature': [None], 
        'Salinity': [None],
        'Density': [1995], 
        
        # 'Temp_ML': [1995], 
        # 'Sal_ML': [1995],
        # 'Dens_ML': [1995],
        
        'MinTemp': [1995],
        'MinTempDepth': [1995],
        
        'TCaro:Chla': [None], 
        'PSC:Chla': [None],
        'PPC:Chla': [None], 
        'PrimPPC:Chla': [None],
        'SecPPC:Chla': [None],
        
        'PPC:TCaro': [None],
        'PSC:TCaro': [None],
        
        # 'TCaro:TAcc': [None],
        # 'PSC:TAcc': [None],
        # 'PPC:TAcc': [None],
        
        # 'Allo:PPC': [None],
        # 'Diadino:PPC': [None],
        # 'Diato:PPC': [None],
        # 'DD+DT:PPC': [None],
        # 'Zea:PPC': [None],
        # 'BCar:PPC': [None],
        # 'PrimPPC:PPC': [None],
        # 'SecPPC:PPC': [None],
        
        # 'Fuco:PSC': [None],
        # 'Hex-Fuco:PSC': [None],
        # 'But-Fuco:PSC': [None],
        # 'Perid:PSC': [None],
        
        # 'Allo:TCaro': [None],
        # 'Diadino:TCaro': [None],
        # 'Diato:TCaro': [None],
        # 'DD+DT:TCaro': [None],
        # 'Zea:TCaro': [None],
        # 'BCar:TCaro': [None],
        # 'Fuco:TCaro': [None],
        # 'Hex-Fuco:TCaro': [None],
        # 'But-Fuco:TCaro': [None],
        # 'Perid:TCaro': [None],
        
        # 'Allo': [None],
        # 'Diadino': [None],
        # 'Diato': [None],
        # 'DD+DT': [None],
        # 'Zea': [None],
        # 'BCar': [None],
        # 'Fuco': [None],
        # 'Hex-Fuco': [None],
        # 'But-Fuco': [None],
        # 'Perid': [None],
        
        # 'mPF': [None],
        # 'nPF': [None],
        # 'pPF': [None],
        
        'Chlorophylla': [None],
        'POC': [2009, 2015, 2016],
        'PrimaryProduction': [2019],
        'SpecPrimProd': [None],
        
        'Diatoms': [None],
        'Cryptophytes': [None],
        'MixedFlagellates': [None],
        'Type4Haptophytes': [None],
        'Prasinophytes': [None],
        
        # 'DiatomBiomass': [1993, 1994, 1998, 2015],
        # 'CryptophyteBiomass': [1993, 1994],
        # 'MixedFlagellateBiomass': [2013, 2014, 2015, 2016],
        # 'Type4HaptophyteBiomass': [1993, 1994, 1998, 2009],
        # 'PrasinophyteBiomass': [1993, 1998, 2013, 2017],
        
        # 'TAcc2:POC': [2009, 2015, 2016],
        
        # 'Chla:POC': [1994, 2009, 2015, 2016],
        # 'TCaro:POC': [2009, 2015, 2016],
        # 'TAcc:POC': [2009, 2015, 2016],
        # 'TPig:POC': [2009, 2015, 2016],
        # 'TAcc:Chla': [None],
        
        'SiO4': [1998],
        'PO4': [1998],
        'NO2': [1998],
        'NO3': [1998],
        'NO3plusNO2': [None],
        
        # 'FIRERho': [2015], #these have not been adjusted/blank corrected/filtered
        # 'FIRESigma': [2015], #these have not been adjusted/blank corrected/filtered
        # 'FIRE_FvFm': [2015], #these have not been adjusted/blank corrected/filtered
        
        'Evenness': [None]}
    
    for param, bad_years in parambadyears.items():
        if param in df.columns and bad_years is not None:
            df.loc[df['Year'].isin(bad_years), param] = np.nan
    
    return df

def filter_and_average_data(df, param):
    data = df
    
    # data_SM = fill_missing_years(data_SM)
    
    data_SM = remove_bad_years(data)
    
    return data_SM

def run_trend_analysis(df, param):
    x = 'Year'
    y = param
    data_YM = df.groupby('Year').median().reset_index()
    
    data_YM = fill_missing_years(data_YM)

    subset = data_YM[[x, y]]
    window_size = 4
    var = subset[y]
    RllAvg = var.rolling(window=window_size, center=True, min_periods=1).mean()
    data_YM['Rolling Average'] = RllAvg

    trenddata = data_YM.loc[:, [y]]
    results = mk.original_test(trenddata)
    p_value = results.p
    slope = results.slope
    
    return data_YM, slope, p_value

# =============================================================================

# =============================================================================
# Load Yearly Station Averaged Dataframe
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_StationLevel_SurfaceAvgCoreDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
df = pd.read_csv(loadpath)

# # -------------------------------------
# # Check Data Distributions
# # -------------------------------------
# # Plot Sample Distribution
# temp = df
# param = 'PPC:PSC'
# temp.sort_values('Year', inplace=True)
# temp = temp.reset_index(drop=True)
# # temp = temp.groupby(['Year','Cruise','RoundedGridLine','RoundedGridStation','Event']).median().reset_index()
# # temp = temp.groupby(['Year','RoundedGridLine','RoundedGridStation']).median().reset_index()
# temp = temp.dropna(subset=param)
# years = temp['Year'].unique()
# fig = plt.figure(figsize=(16,19), dpi=100.0, tight_layout=True)
# loc = 0
# for yr in years:
#     sub = temp[temp['Year'] == yr]
#     loc += 1
#     ax = fig.add_subplot(6,5, loc)
#     plt.scatter(x='RoundedGridStation', y='RoundedGridLine', data=sub)
#     plt.gca().invert_xaxis()
#     plt.xlim(260, -120)
#     plt.ylim(100, 700)
#     plt.title(yr)

# # Plot Yearly Station Counts Bar Plot
# param = 'RoundedGridStation'
# tempcount = temp.groupby('Year').count().reset_index()
# fig = plt.figure(figsize=(3,1.5), dpi=600, tight_layout=True)
# plt.bar(tempcount['Year'], tempcount[param], color='grey')
# plt.ylim(10,75)
# plt.xlim(1990,2021)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.ylabel('Count', fontsize=12)
# plt.title('Yearly Total Station Counts', fontsize=12, fontweight='bold')
# plt.minorticks_on()
# =============================================================================

# =============================================================================
# Plot Time Series ############################################################
# =============================================================================
plotrollingavg = False #True, False
plottheilsen = True #True, False

x = 'Year'
xticks = 'auto' #manual, auto

y = 'MixedFlagellates'
yscale = 'linear' #linear, log, symlog
ylims = 'manual' #manual, auto, dynamic
ymin = 0
ymax = 101

identifier = None

units = ''
# units = ' (%)'
# units = ' (m)'
# units = ' (ratio)'
# units = ' (unitless)'
# units = '\n(mgChl $\mathregular{m^{-3}}$)' #Chlorophylla
# units = '\n(mgC $\mathregular{m^{-3}}$ $\mathregular{day^{-1}}$)' #PrimaryProduction
# units = '\n(mgC $\mathregular{(mgChl)^{-1}}$ $\mathregular{day^{-1}}$)' #SpecPrimProd

xlabel = None
ylabel = y + units
# ylabel = 'MLD (m)'
# ylabel = 'Max Buoyancy \nFreq. ($\mathregular{s^{-1}}$)'
# ylabel = 'Surface\nSalinity (PSU)'
# ylabel = 'Chla-Specific Production' + units
# ylabel = 'Quality Index\n(unitless)'
# ylabel = 'Temperature\nMin. Depth (m)'
# ylabel = 'Temperature\nMin. (C)'
# ylabel = 'WW Temperature\nMin. (C)'
# ylabel = 'WW Temperature\nMin. Depth (m)'
# ylabel = 'WW Upper\nBoundary Depth (m)'
# ylabel = 'WW Lower\nBoundary Depth (m)'
# ylabel = 'WW Thickness (m)'
# ylabel = 'WW % Observation'

font_size = 12
ident_size = 24
markersize = 4
linewidth = 1

width = 3.5
length = 1.5

fig = plt.figure(figsize=(width, length), dpi=1200)
ax1 = fig.add_subplot(1, 1, 1)

# ------------------------------------------
#set up defined variables
yscale = yscale
plotrollingavg = plotrollingavg
plottheilsen = plottheilsen
fs = font_size
lw = linewidth
ms = markersize
# ------------------------------------------

#run data filtering and trend test
data_SM = filter_and_average_data(df, y)
data_YM, slope, p_value = run_trend_analysis(data_SM, y)
data_YM2 = data_YM[data_YM[y].notna()]

data = data_YM

#calculate mld anomaly per year
mld_median = data['MLD'].median()
data['MLD_Anomaly'] = data['MLD'] - mld_median
data = data[['Year', 'MLD', 'MLD_Anomaly']].reset_index(drop=True)
data = fill_missing_years(data)

#plotting mld anomalies with color differentiation
for i, row in data.iterrows():
    if np.isnan(row['MLD_Anomaly']):
        # If data is missing, plot a marker
        ax1.scatter(row['Year'], 0, color='grey', marker='.', alpha=0.5)
    else:
        # Otherwise, plot the bar with color based on the anomaly value
        ax1.bar(row['Year'], row['MLD_Anomaly'], color='tab:red' if row['MLD_Anomaly'] > 0 else 'tab:blue')

#adjust axis params
ax1.tick_params('both', labelsize=8)
ax1.set_ylabel('$\mathregular{MLD_{Anomaly}}$ (m)', 
                color='Black', fontweight='normal', fontsize=12, labelpad=2)
# ax1.set_ylim(0, 100)
ax1.set_xlim(1990, 2021)
ax1.minorticks_on()
ax1.tick_params('x', labelbottom=True)

l = 0 + 0.175
r = 1 - 0.03
t = 1 - 0.15
b = 0 + 0.125
plt.subplots_adjust(left=l, right=r, top=t, bottom=b)
# =============================================================================


# # =============================================================================
# # Save Figure as PDF for Compiling in Adobe Illustrator
# # =============================================================================
# current_directory = Path.cwd()
# absolute_path = Path("analysis/publication figures/Fig6/")
# filename = Path("Fig6d_MLDAnomTS.pdf")
# savepath = str(current_directory / absolute_path / filename)

# plt.savefig(savepath)
# # =============================================================================
