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
filename = Path("PALLTER_StationLevel_SurfaceAvgCoreDataframe.csv")
loadpath = str(current_directory / absolute_path / filename)
df2 = pd.read_csv(loadpath)

#reduce dataframe to necessary parameters
df2 = df2[['Year','SIAdvance','SIRetreat','SIDuration','IceDays','SIRetrProx',
         'SIExtent','SIArea','OWArea','TotalSIConc']]

#calculate yearly medians (for yearly sea ice and mld values)
data_SI = df2.groupby(['Year']).median().reset_index()

#adjust sea ice concentration values to %
data_SI['TotalSIConc'] = data_SI['TotalSIConc']*100
# =============================================================================


# =============================================================================
# Create Figure Plotting Space
# =============================================================================
width = 4.75
height = 1.5
fig = plt.figure(figsize=(width, height), dpi=1200)
ax1 = fig.add_subplot(1, 1, 1)
ax1_twin1 = ax1.twinx()
# ax1_twin2 = ax1.twinx()

ax1.tick_params(axis='x', labelbottom=True)

lw = 1
ms = 4

ax1.tick_params(axis='both', which='both', labelsize=10)
ax1_twin1.tick_params(axis='both', which='both', labelsize=10)
ax1.set_xlim(1993, 2020)
ax1.set_xticks([1995, 2000, 2005, 2010, 2015, 2020])
ax1.minorticks_on()

# -----------------------------------------------------------------------------
label = 'Sea Ice Duration'
c1 = adjust_lightness('Black', amount=1)
line1, = ax1.plot(data_SI['Year'], data_SI['SIDuration'], color=c1, lw=lw, 
          marker='o', markersize=ms*1.25, label=label, zorder=10)
# line1 = ax1.bar(data_SI['Year'], data_SI['SIDuration'], color=c1, lw=lw, label=label, zorder=10)

ticks = np.arange(0, 300, 50)
ax1.set_yticks(ticks)
ax1.set_ylabel('Duration (days)', color=c1, fontweight='normal', fontsize=fs, labelpad=4)
ax1.tick_params(axis='y', which='both', colors=c1)
ax1.set_ylim(80, 255)

label = 'Sea Ice Retreat Prox.'
c3 = adjust_lightness('Red', amount=1.25)
line3, = ax1_twin1.plot(data_SI['Year'], data_SI['SIRetrProx'], color=c3, lw=lw, 
         marker='s', markersize=ms, label=label, zorder=10)
# line3 = ax1_twin1.bar(data_SI['Year'], data_SI['SIRetrProx'], color=c3)
ax1_twin1.set_ylabel('Proximity\n(relative days)', color='Red', fontweight='normal', fontsize=fs)
ticks = np.arange(-100, 50, 25)
ax1_twin1.set_yticks(ticks)
ax1_twin1.tick_params(axis='y', which='major', colors='Red', length=4, width=1.05)
ax1_twin1.tick_params(axis='y', which='minor', colors='Red', length=2.5, width=1.05)
ax1_twin1.set_ylim(-90, 31)
ax1_twin1.minorticks_on()

# ax1.legend(handles=[line1, line2, line3], 
#            bbox_to_anchor=(0.62, -0.03), ncols=1, loc=4, fontsize=8, frameon=False)
# -----------------------------------------------------------------------------

# # -----------------------------------------------------------------------------
# label = 'Sea Ice Duration'
# c1 = adjust_lightness('Black', amount=1)
# ax1.plot(data_SI['Year'], data_SI['SIDuration'], color=c1, lw=lw, 
#          marker='o', markersize=ms, label=label, zorder=10)
# ax1.set_ylabel('Duration (days)', color=c1, fontweight='bold', fontsize=fs)
# ax1.tick_params(axis='y', which='both', colors=c1)
# ax1.set_ylim(90, 275)

# label = 'Winter Ice Days'
# c2 = adjust_lightness('Blue', amount=1.25)
# ax1_twin1.plot(data_SI['Year'], data_SI['IceDays'], color=c2, lw=lw, 
#          marker='D', markersize=ms, label=label, zorder=10)
# ax1_twin1.set_ylabel('Ice Days (days)', color=c2, fontweight='bold', fontsize=fs)
# ax1_twin1.tick_params(axis='y', which='both', colors=c2)
# ax1_twin1.set_ylim(90, 275)

# label = 'Sea Ice Retreat Prox.'
# c3 = adjust_lightness('Red', amount=1.25)
# ax1_twin2.spines['right'].set_position(('outward', 55))
# ax1_twin2.plot(data_SI['Year'], data_SI['SIRetrProx'], color=c3, lw=lw, 
#          marker='s', markersize=ms, label=label, zorder=10)
# ax1_twin2.set_ylabel('Retreat Prox. (relative days)', color=c3, fontweight='bold', fontsize=fs)
# ax1_twin2.tick_params(axis='y', which='both', colors=c3)
# ax1_twin2.set_ylim(-75, 29)
# # -----------------------------------------------------------------------------

# plt.subplots_adjust(left=0.175, right=0.825, top=0.95, bottom=0.1)

l = 0 + 0.135
r = 1 - 0.165
t = 1 - 0.15
b = 0 + 0.15
plt.subplots_adjust(left=l, right=r, top=t, bottom=b)
# =============================================================================


# =============================================================================
# Save Figure as PDF for Compiling in Adobe Illustrator
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("analysis/publication figures/Fig3/")
filename = Path("Fig3a_WSI.pdf")
savepath = str(current_directory / absolute_path / filename)

plt.savefig(savepath)
# =============================================================================