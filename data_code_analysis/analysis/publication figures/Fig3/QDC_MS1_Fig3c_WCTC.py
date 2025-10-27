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
         'SigmaTheta','MLD','QI','max_N2', 'WWUpper','WWLower']]

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
data = data[data['Depth'] >= 0]

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

#isolate WWUpper data for plotting 
wwudata = remove_bad_years(data_YM)
wwudata = wwudata[['Year', 'WWUpper']]
wwudata = wwudata.sort_values(by='Year')
wwudata = wwudata.dropna(subset='WWUpper')
wwudata_filled = fill_missing_years(wwudata)

#isolate WWLower data for plotting 
wwldata = remove_bad_years(data_YM)
wwldata = wwldata[['Year', 'WWLower']]
wwldata = wwldata.sort_values(by='Year')
wwldata = wwldata.dropna(subset='WWLower')
wwldata_filled = fill_missing_years(wwldata)
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
width = 4.75
height = 1.75
fig = plt.figure(figsize=(width, height), dpi=1200)
ax1 = fig.add_subplot(1,1,1)

lw = 1
ms = 4

ax1.tick_params(axis='both', which='both', labelsize=10)
ax1.set_xlim(1993, 2020)
ax1.set_xticks([1995, 2000, 2005, 2010, 2015, 2020])
ax1.minorticks_on()

# -----------------------------------------------------------------------------
#plot density as line contours and temperature as colored filled contours
# line_contour = ax1.contour(dens_matrix.columns.values, dens_matrix.index.values, dens_matrix.values, 
#             levels=20, colors='k', alpha=0.25, linewidths=lw*0.5)
color_contour = ax1.contourf(temp_matrix.columns.values, temp_matrix.index.values, temp_matrix.values, 
             levels=21, cmap='coolwarm')
line_contour = ax1.contour(temp_matrix.columns.values, temp_matrix.index.values, temp_matrix.values, 
              levels=21, colors='k', alpha=0.15, linewidths=.5, negative_linestyles='solid')

#adjust axis params and add contour labels
ax1.set_ylabel('Depth (m)', fontweight='normal', fontsize=12)
# ax1.clabel(line_contour, inline=True, fontsize=8, fmt='%1.1f')
ax1.set_ylim(0,200)
ax1.invert_yaxis()

#plot colorbar for contourf
ticks = np.arange(-1, 3, 1)
cbar = fig.colorbar(color_contour, ax=ax1, ticks=ticks, orientation='vertical', pad=0.025)
cbar.set_label('Temperature (°C)', fontweight='normal', fontsize=12, labelpad=8)
cbar.ax.tick_params(labelsize=10)
cbar.ax.minorticks_on()

# #visualize sampling points
# points = data_YMDP.loc[data_YMDP['Depth'] % 10 == 0] #only plot points at 10m interval
# ax1.scatter(points['Year'], points['Depth'], s=0.25, color='gray', alpha=0.25)

#plot mld points
mldline1, = ax1.plot(mlddata_filled['Year'], mlddata_filled['MLD'], marker='D', markeredgewidth=0.5, mec='k', 
        color='grey', linestyle='-', linewidth=.75, markersize=ms, label='MLD', zorder=10)
ax1.plot(mlddata['Year'], mlddata['MLD'], color='grey', linestyle=':', linewidth=lw, label='_MLD')
ax1.legend(handles=[mldline1], bbox_to_anchor=(0.19, -0.005), ncols=1, loc=4, fontsize=8, markerscale=0.8,
            frameon=True, framealpha=1, edgecolor='white', borderpad=0.1, handletextpad=0.25)

# #plot WWUpper points
# wwuline1, = ax1.plot(wwudata_filled['Year'], wwudata_filled['WWUpper'], marker='v', 
#         color='k', linestyle='-', linewidth=.5, markersize=ms, label='WWUpper')
# ax1.plot(wwudata['Year'], wwudata['WWUpper'], color='k', linestyle=':', linewidth=lw, label='_WWUpper')
# ax1.legend(handles=[wwuline1], bbox_to_anchor=(0.185, -0.02), ncols=1, loc=4, fontsize=8, 
#            frameon=True, framealpha=1, borderpad=0.2, handletextpad=0.5)

# #plot WWLower points
# wwlline1, = ax1.plot(wwldata_filled['Year'], wwldata_filled['WWLower'], marker='^', 
#         color='k', linestyle='-', linewidth=.5, markersize=ms, label='WWLower')
# ax1.plot(wwldata['Year'], wwldata['WWLower'], color='k', linestyle=':', linewidth=lw, label='_WWLower')
# ax1.legend(handles=[wwuline1, wwlline1], bbox_to_anchor=(0.185, -0.02), ncols=1, loc=4, fontsize=8, 
#            frameon=True, framealpha=1, borderpad=0.2, handletextpad=0.5)
# -----------------------------------------------------------------------------

l = 0 + 0.135
r = 1 - 0.0175
t = 1 - 0.075
b = 0 + 0.125
plt.subplots_adjust(left=l, right=r, top=t, bottom=b)
# =============================================================================


# =============================================================================
# Save Figure as PDF for Compiling in Adobe Illustrator
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("analysis/publication figures/Fig3/")
filename = Path("Fig3c_TempCon.pdf")
savepath = str(current_directory / absolute_path / filename)

plt.savefig(savepath)
# =============================================================================
