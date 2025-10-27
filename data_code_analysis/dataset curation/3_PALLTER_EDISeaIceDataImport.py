# =============================================================================
# Imports 
# =============================================================================
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd 
# =============================================================================


# =============================================================================
# Import Annual Seasonal Sea Ice Indices from EDI
# =============================================================================
# Package ID: knb-lter-pal.151.9 Cataloging System:https://pasta.edirepository.org.
# Data set title: Seasonal sea ice indices including the timing of ice-edge advance and ice-edge retreat (in year day), the ice season duration (in days) and number of actual ice days (versus open water days) within the ice season, extracted for various PAL LTER sub-regions West of the Antarctic Peninsula and derived from passive microwave satellite data for 1979/80 to 2023/24 ice seasons..
# Data set creator:    - Palmer Station Antarctica LTER 
# Data set creator:  Sharon Stammerjohn - INSTAAR, University of Colorado Boulder 
# Metadata Provider:    - Palmer Station Antarctica LTER 
# Contact:    - PAL LTER Information Manager Palmer Station Antarctica LTER  - pallter@marine.rutgers.edu
# Stylesheet v1.0 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu      

infile1  ="https://pasta.lternet.edu/package/data/eml/knb-lter-pal/151/9/13bb2b05f1e930574150d9cd8ab04b8a".strip() 
infile1  = infile1.replace("https://","http://")
                 
dt1 =pd.read_csv(infile1, skiprows=1, sep=",", quotechar='"', 
                 names=["Ice_Year","Subregion","Advance","Retreat","Duration","Ice_Days"],
                 na_values={'Advance':['-999',],
                            'Retreat':['-999',],
                            'Duration':['-999',],
                            'Ice_Days':['-999',],})

#coerce the data into the types specified in the metadata  
dt1.Ice_Year=dt1.Ice_Year.astype('category')  
dt1.Subregion=dt1.Subregion.astype('category') 
dt1.Advance=pd.to_numeric(dt1.Advance,errors='coerce',downcast='integer') 
dt1.Retreat=pd.to_numeric(dt1.Retreat,errors='coerce',downcast='integer') 
dt1.Duration=pd.to_numeric(dt1.Duration,errors='coerce',downcast='integer') 
dt1.Ice_Days=pd.to_numeric(dt1.Ice_Days,errors='coerce',downcast='integer') 

#rename for editing and merging
siseasonaldf = dt1

#rename columns
siseasonaldf = siseasonaldf.rename(columns={'Ice_Days':'IceDays'})

#add in Year distinction for summer impact
siseasonaldf['Year'] = siseasonaldf['Ice_Year'].apply(lambda x: int(x[:4]) + 1)
#   ^Year corresponds to the cruise year impacted by that respective ice season,  
#   since ice years last from mid-Feb to mid-Feb; e.g., January 2009 conditions 
#   will be linked to the preceeding sea ice season in ice year 2008-09
siseasonaldf['Year'] = siseasonaldf['Year'].astype(int)

#add in retreat proximity to january (for more intuitive representation of retreat)
siseasonaldf['RetreatProximity'] = siseasonaldf['Retreat'] - 365 
#   ^RetreatProximity is the number of days before or after the start of January 
#   (YearDay 365) that sea ice retreat began (i.e., a negative value indicates that sea ice 
#   retreat began ## number of days BEFORE Jan 1st; a positive values indicates that sea ice
#   retreat began ## number of days AFTER Jan 1st)

#select only specific regions
subregions = ['Pdsr']
# subregions = ['Pdsr', 'Pori', 'Pnew']
siseasonaldf = siseasonaldf[siseasonaldf['Subregion'].isin(subregions)].reset_index(drop=True)

#select only specific years
years = np.arange(1991, 2020 + 1)
siseasonaldf = siseasonaldf[siseasonaldf['Year'].isin(years)].reset_index(drop=True)

#drop excess columns
siseasonaldf = siseasonaldf.drop(columns=['Ice_Year'])
# =============================================================================


# =============================================================================
# Import Monthly Sea Ice Coverage from EDI
# =============================================================================
# Package ID: knb-lter-pal.34.9 Cataloging System:https://pasta.edirepository.org.
# Data set title: Average monthly sea ice coverage for various PAL LTER sub-regions West of the Antarctic Peninsula derived from passive microwave satellite data, 1978 - June 2024..
# Data set creator:    - Palmer Station Antarctica LTER 
# Data set creator:  Sharon Stammerjohn - INSTAAR, University of Colorado Boulder 
# Metadata Provider:    - Palmer Station Antarctica LTER 
# Contact:    - PAL LTER Information Manager Palmer Station Antarctica LTER  - pallter@marine.rutgers.edu
# Stylesheet v1.0 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu      

infile1  ="https://pasta.lternet.edu/package/data/eml/knb-lter-pal/34/9/0cd186e85bf7e635fd18883073eaea19".strip() 
infile1  = infile1.replace("https://","http://")
                 
dt1 =pd.read_csv(infile1, skiprows=1, sep=",", quotechar='"',
                 names=["Calendar_Year","Month","Subregion","Sea_hyphen_ice_extent",     
                    "Sea_hyphen_ice_area","Open_water_area"],
                 parse_dates=['Calendar_Year',],
                 na_values={'Sea_hyphen_ice_extent':['-999',],
                            'Sea_hyphen_ice_area':['-999',],
                            'Open_water_area':['-999',],})

#coerce the data into the types specified in the metadata 
#   ^since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
#   this new column is added to the dataframe but does not show up in automated summaries below. 
dt1=dt1.assign(Calendar_Year_datetime=pd.to_datetime(dt1.Calendar_Year,errors='coerce')) 
dt1.Month=pd.to_numeric(dt1.Month,errors='coerce')  
dt1.Subregion=dt1.Subregion.astype('category') 
dt1.Sea_hyphen_ice_extent=pd.to_numeric(dt1.Sea_hyphen_ice_extent,errors='coerce',downcast='integer') 
dt1.Sea_hyphen_ice_area=pd.to_numeric(dt1.Sea_hyphen_ice_area,errors='coerce',downcast='integer') 
dt1.Open_water_area=pd.to_numeric(dt1.Open_water_area,errors='coerce',downcast='integer') 

#rename for editing and merging
sicoveragedf = dt1

#rename columns and assign years
sicoveragedf = sicoveragedf.rename(columns={'Sea_hyphen_ice_extent':'SIExtent',
                                            'Sea_hyphen_ice_area':'SIArea',
                                            'Open_water_area':'OWArea'})
sicoveragedf['Year'] = sicoveragedf['Calendar_Year_datetime'].dt.year

#calculate sea ice concentration from extent and area
sicoveragedf['TotalSIConc'] = sicoveragedf['SIArea'] / sicoveragedf['SIExtent']

#select only specific region
subregions = ['Pdsr']
# subregions = ['Pdsr', 'Pori', 'Pnew']
sicoveragedf = sicoveragedf[sicoveragedf['Subregion'].isin(subregions)].reset_index(drop=True)

#select only specific years
years = np.arange(1991, 2020 + 1)
sicoveragedf = sicoveragedf[sicoveragedf['Year'].isin(years)].reset_index(drop=True)

#select only specific month (January)
months = [1]
sicoveragedf = sicoveragedf[sicoveragedf['Month'].isin(months)].reset_index(drop=True)

#drop excess columns
sicoveragedf = sicoveragedf.drop(columns=['Calendar_Year', 'Calendar_Year_datetime', 'Month'])
# =============================================================================


# =============================================================================
# Merge Dataframes By Year
# =============================================================================
sidf = siseasonaldf.merge(sicoveragedf, on=['Year', 'Subregion'])

#drop excess columns
sidf = sidf.drop(columns=['Subregion'])

#rename columns
sidf = sidf.rename(columns={'Advance':'SIAdvance',
                            'Retreat':'SIRetreat',
                            'Duration':'SIDuration',
                            'RetreatProximity':'SIRetrProx'})

#reorder columns
columns_to_move = ['Year']
sidf = sidf[columns_to_move + [col for col in sidf.columns if col not in columns_to_move]]
# =============================================================================


# =============================================================================
# Save Data as EDI Sea Ice Dataframe
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EDISeaIceDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

sidf.to_csv(savepath, index=False)
# =============================================================================

