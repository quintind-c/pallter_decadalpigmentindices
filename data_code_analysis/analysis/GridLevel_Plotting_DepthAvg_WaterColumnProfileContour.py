# =============================================================================
# Imports
# =============================================================================
import gsw
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import warnings
warnings.filterwarnings('ignore')

# Set global font properties
fs = 12
plt.rcParams['font.size'] = fs
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.style'] = 'normal'
# =============================================================================


# =============================================================================
# Functions
# =============================================================================
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

def fill_missing_years(df):    
    all_years = np.arange(1991, 2021)
    existing_years = df['Year'].unique()
    missing_years = np.setdiff1d(all_years, existing_years)
    missing_data = pd.DataFrame({'Year': missing_years})
    missing_data = missing_data.assign(**{column: np.nan for column in df.columns if column != 'Year'})
    df = pd.concat([df, missing_data], ignore_index=True)
    df = df.sort_values('Year')
    return df

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
# =============================================================================


# =============================================================================
# Load Station-Level Depth-Averaged Core Dataset
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EventLevel_DepthAvgHydroDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
df = pd.read_csv(loadpath)

#reduce dataframe to necessary parameters
df = df[['Year','RoundedGridLine','RoundedGridStation','Depth','Latitude',
         'Longitude','Pressure','Temperature','Salinity','Density',
         'SigmaTheta','MLD','QI','max_N2']]

# # -----------------------------------------
# #convert practical salinity (SP) to absolute salinity (SA) using latitude and longitude
# df['AbsoluteSalinity'] = gsw.SA_from_SP(df['Salinity'], df['Pressure'], df['Longitude'], df['Latitude'])
# #convert in-situ temperature to conservative temperature (CT)
# df['ConservativeTemp'] = gsw.CT_from_t(df['AbsoluteSalinity'], df['Temperature'], df['Pressure'])
# #calculate potential density relative to surface pressure
# df['PotentialDensity'] = gsw.sigma0(df['AbsoluteSalinity'], df['ConservativeTemp'])

# #double check that calculated values match
# tempdf = df[['Year','Depth','Density','SigmaTheta','PotentialDensity']]
# # -----------------------------------------

#crop depths
data = df[df['Depth'] <= 200]
data = data[data['Depth'] >= 3]

#calculate yearly median depth profiles
data_YMDP = data.groupby(['Year','Depth']).median().reset_index()

#calculate yearly medians (for mld values)
data_YM = data.groupby(['Year']).median().reset_index()

#isolate mld data for plotting 
mlddata = remove_bad_years(data_YM)
mlddata = mlddata[['Year', 'MLD']]
mlddata = mlddata.sort_values(by='Year')
mlddata = mlddata.dropna(subset='MLD')
mlddata_filled = fill_missing_years(mlddata)
# =============================================================================


# =============================================================================
# Set Up Data for Contour Depth Profile Plots
# =============================================================================
#create pivoted matrix data
temp_matrix = data_YMDP.pivot_table(index='Depth', columns='Year', values='Temperature')
sal_matrix = data_YMDP.pivot_table(index='Depth', columns='Year', values='Salinity')
dens_matrix = data_YMDP.pivot_table(index='Depth', columns='Year', values='SigmaTheta')
# -----------------------------------------------------------------------------


# =============================================================================
# Create Figure Plotting Space
# =============================================================================
width = 5
height = 2
fig = plt.figure(figsize=(width, height), dpi=600)
ax1 = fig.add_subplot(1,1,1)

lw = 1
ms = 3.5

colorcontour = 'Temperature'

ax1.tick_params(axis='both', which='both', labelsize=8)
ax1.set_xlim(1993, 2020)
ax1.set_xticks([1995, 2000, 2005, 2010, 2015, 2020])
ax1.minorticks_on()

# -----------------------------------------------------------------------------
if colorcontour == 'Salinity':
    matrix = sal_matrix
    label = 'Salinity (g/kg)'
    ticks = np.arange(32.75, 34.75, 0.25)
if colorcontour == 'Temperature':
    matrix = temp_matrix
    label = 'Temperature (C)'
    ticks = np.arange(-5, 5, 0.5)

#plot density as line contours and salinity as colored filled contours
line_contour = ax1.contour(dens_matrix.columns.values, dens_matrix.index.values, dens_matrix.values, 
            levels=12, colors='k', alpha=0.25, linewidths=lw*0.5)
color_contour = ax1.contourf(matrix.columns.values, matrix.index.values, matrix.values, 
             levels=20, cmap='coolwarm')

#adjust axis params and add contour labels
ax1.set_ylabel('Depth (m)', fontweight='normal', fontsize=10)
ax1.clabel(line_contour, inline=True, fontsize=8, fmt='%1.1f')
ax1.set_ylim(0,200)
ax1.invert_yaxis()

#plot colorbar for contourf
cbar = fig.colorbar(color_contour, ax=ax1, orientation='vertical', pad=0.025, ticks=ticks)
cbar.set_label(label, fontweight='normal', fontsize=10)
cbar.ax.tick_params(labelsize=8)
cbar.ax.minorticks_on()

#visualize sampling points
points = data_YMDP.loc[data_YMDP['Depth'] % 10 == 0] #only plot points at 10m interval
ax1.scatter(points['Year'], points['Depth'], s=0.25, color='gray', alpha=0.25)

# #plot mld points
# mldline1, = ax1.plot(mlddata_filled['Year'], mlddata_filled['MLD'], marker='o', 
#         color='k', linestyle='-', linewidth=lw, markersize=ms, label='Median MLD')
# ax1.plot(mlddata['Year'], mlddata['MLD'], color='k', linestyle=':', linewidth=lw, label='_MLD')
# ax1.legend(handles=[mldline1], bbox_to_anchor=(0.46, -0.03), ncols=1, loc=4, fontsize=8, frameon=False)
# -----------------------------------------------------------------------------

l = 0 + 0.115
r = 1 - 0.01
t = 1 - 0.075
b = 0 + 0.125
plt.subplots_adjust(left=l, right=r, top=t, bottom=b)
# =============================================================================
