# =============================================================================
# Imports 
# =============================================================================
import numpy as np
import pandas as pd
from pathlib import Path
import math
# =============================================================================


# =============================================================================
# Import Merged Discrete Cruise Water-Column Data from EDI
# =============================================================================
# Package ID: knb-lter-pal.310.1 Cataloging System:https://pasta.edirepository.org.
# Data set title: Merged discrete water-column data from PAL LTER research cruises along the Western Antarctic Peninsula, from 1991 to 2020..
# Data set creator:    - Palmer Station Antarctica LTER 
# Data set creator:  Nicole Waite - Rutgers University 
# Metadata Provider:    - Palmer Station Antarctica LTER 
# Contact:    - PAL LTER Information Manager Palmer Station Antarctica LTER  - pallter@marine.rutgers.edu
# Stylesheet v1.0 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu      

infile1  ="https://pasta.lternet.edu/package/data/eml/knb-lter-pal/310/1/06aea5a60263d2c1f6b7bbc93e047a90".strip() 
infile1  = infile1.replace("https://","http://")
                 
dt1 =pd.read_csv(infile1 
          ,skiprows=1
            ,sep=","  
                ,quotechar='"' 
           , names=[
                    "studyName",     
                    "DatetimeGMT",     
                    "JulianDay",     
                    "Event",     
                    "GridLine",     
                    "GridStation",     
                    "RoundedGridLine",     
                    "RoundedGridStation",     
                    "GridRegion",     
                    "NorthSouthRegion",     
                    "InshoreOffshoreRegion",     
                    "Latitude",     
                    "Longitude",     
                    "BottleNumber",     
                    "Depth",     
                    "Temperature",     
                    "Salinity",     
                    "Density",     
                    "BeamTransmission",     
                    "Oxygen",     
                    "OxygenSaturation",     
                    "PAR",     
                    "Fluorescence",     
                    "Chlorophyll",     
                    "Phaeopigment",     
                    "FilterCode",     
                    "Chlorophyllc3",     
                    "Chlorophyllide",     
                    "Chlorophyllc2",     
                    "Peridinin",     
                    "Phaeophorbide",     
                    "x19butanoyloxyfucoxanthin",     
                    "Fucoxanthin",     
                    "Neoxanthin",     
                    "Prasinoxanthin",     
                    "x19hexanoyloxyfucoxanthin",     
                    "Violaxanthin",     
                    "Diadinoxanthin",     
                    "Antheraxanthin",     
                    "Alloxanthin",     
                    "Diatoxanthin",     
                    "Zeaxanthin",     
                    "Lutein",     
                    "Crocoxanthin",     
                    "Chlorophyllb",     
                    "Echinenone",     
                    "DivinylChlorophylla",     
                    "Chlorophylla",     
                    "Lycopene",     
                    "Phaeophytin",     
                    "AlphaCarotene",     
                    "BetaCarotene",     
                    "Prasinophytes",     
                    "Cryptophytes",     
                    "MixedFlagellates",     
                    "Diatoms",     
                    "Haptophytes",     
                    "PercentIrradiance",     
                    "PrimaryProduction",     
                    "PrimProdSTD",     
                    "FIREGain",     
                    "FIREFvOverFm",     
                    "FIRESigma",     
                    "FIRERho",     
                    "BacterialAbundance",     
                    "ThymidineIncorporation",     
                    "LeucineIncorporation",     
                    "HNA",     
                    "LNA",     
                    "DOC",     
                    "POC",     
                    "N",     
                    "PO4",     
                    "SiO4",     
                    "NO2",     
                    "NO3",     
                    "NH4",     
                    "NO3plusNO2",     
                    "O2mlperl",     
                    "O2umolperl",     
                    "DIC1",     
                    "DIC2",     
                    "Alkalinity1",     
                    "Alkalinity2",     
                    "DICTemperature",     
                    "DICSalinity",     
                    "Notes1",     
                    "Notes2"    ]

          ,parse_dates=[
                        'DatetimeGMT',
                ] 
            ,na_values={
                  'studyName':[
                          'NaN',],
                  'DatetimeGMT':[
                          'NaT',],
                  'JulianDay':[
                          'NaN',],
                  'Event':[
                          'NaN',],
                  'GridLine':[
                          'NaN',],
                  'GridStation':[
                          'NaN',],
                  'RoundedGridLine':[
                          'NaN',],
                  'RoundedGridStation':[
                          'NaN',],
                  'Latitude':[
                          'NaN',],
                  'Longitude':[
                          'NaN',],
                  'BottleNumber':[
                          'NaN',],
                  'Depth':[
                          'NaN',],
                  'Temperature':[
                          'NaN',],
                  'Salinity':[
                          'NaN',],
                  'Density':[
                          'NaN',],
                  'BeamTransmission':[
                          'NaN',],
                  'Oxygen':[
                          'NaN',],
                  'OxygenSaturation':[
                          'NaN',],
                  'PAR':[
                          'NaN',],
                  'Fluorescence':[
                          'NaN',],
                  'Chlorophyll':[
                          'NaN',],
                  'Phaeopigment':[
                          'NaN',],
                  'FilterCode':[
                          'NaN',],
                  'Chlorophyllc3':[
                          'NaN',],
                  'Chlorophyllide':[
                          'NaN',],
                  'Chlorophyllc2':[
                          'NaN',],
                  'Peridinin':[
                          'NaN',],
                  'Phaeophorbide':[
                          'NaN',],
                  'x19butanoyloxyfucoxanthin':[
                          'NaN',],
                  'Fucoxanthin':[
                          'NaN',],
                  'Neoxanthin':[
                          'NaN',],
                  'Prasinoxanthin':[
                          'NaN',],
                  'x19hexanoyloxyfucoxanthin':[
                          'NaN',],
                  'Violaxanthin':[
                          'NaN',],
                  'Diadinoxanthin':[
                          'NaN',],
                  'Antheraxanthin':[
                          'NaN',],
                  'Alloxanthin':[
                          'NaN',],
                  'Diatoxanthin':[
                          'NaN',],
                  'Zeaxanthin':[
                          'NaN',],
                  'Lutein':[
                          'NaN',],
                  'Crocoxanthin':[
                          'NaN',],
                  'Chlorophyllb':[
                          'NaN',],
                  'Echinenone':[
                          'NaN',],
                  'DivinylChlorophylla':[
                          'NaN',],
                  'Chlorophylla':[
                          'NaN',],
                  'Lycopene':[
                          'NaN',],
                  'Phaeophytin':[
                          'NaN',],
                  'AlphaCarotene':[
                          'NaN',],
                  'BetaCarotene':[
                          'NaN',],
                  'Prasinophytes':[
                          'NaN',],
                  'Cryptophytes':[
                          'NaN',],
                  'MixedFlagellates':[
                          'NaN',],
                  'Diatoms':[
                          'NaN',],
                  'Haptophytes':[
                          'NaN',],
                  'PercentIrradiance':[
                          'NaN',],
                  'PrimaryProduction':[
                          'NaN',],
                  'PrimProdSTD':[
                          'NaN',],
                  'FIREGain':[
                          'NaN',],
                  'FIREFvOverFm':[
                          'NaN',],
                  'FIRESigma':[
                          'NaN',],
                  'FIRERho':[
                          'NaN',],
                  'BacterialAbundance':[
                          'NaN',],
                  'ThymidineIncorporation':[
                          'NaN',],
                  'LeucineIncorporation':[
                          'NaN',],
                  'HNA':[
                          'NaN',],
                  'LNA':[
                          'NaN',],
                  'DOC':[
                          'NaN',],
                  'POC':[
                          'NaN',],
                  'N':[
                          'NaN',],
                  'PO4':[
                          'NaN',],
                  'SiO4':[
                          'NaN',],
                  'NO2':[
                          'NaN',],
                  'NO3':[
                          'NaN',],
                  'NH4':[
                          'NaN',],
                  'NO3plusNO2':[
                          'NaN',],
                  'O2mlperl':[
                          'NaN',],
                  'O2umolperl':[
                          'NaN',],
                  'DIC1':[
                          'NaN',],
                  'DIC2':[
                          'NaN',],
                  'Alkalinity1':[
                          'NaN',],
                  'Alkalinity2':[
                          'NaN',],
                  'DICTemperature':[
                          'NaN',],
                  'DICSalinity':[
                          'NaN',],
                  'Notes1':[
                          'NaN',],
                  'Notes2':[
                          'NaN',],})

#coerce the data into the types specified in the metadata  
#   ^since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
#   this new column is added to the dataframe but does not show up in automated summaries below. 
dt1.studyName=dt1.studyName.astype('category') 
dt1=dt1.assign(DatetimeGMT_datetime=pd.to_datetime(dt1.DatetimeGMT,errors='coerce')) 
dt1.JulianDay=pd.to_numeric(dt1.JulianDay,errors='coerce') 
dt1.Event=pd.to_numeric(dt1.Event,errors='coerce') 
dt1.GridLine=pd.to_numeric(dt1.GridLine,errors='coerce') 
dt1.GridStation=pd.to_numeric(dt1.GridStation,errors='coerce') 
dt1.RoundedGridLine=pd.to_numeric(dt1.RoundedGridLine,errors='coerce') 
dt1.RoundedGridStation=pd.to_numeric(dt1.RoundedGridStation,errors='coerce')  
dt1.GridRegion=dt1.GridRegion.astype('category')  
dt1.NorthSouthRegion=dt1.NorthSouthRegion.astype('category')  
dt1.InshoreOffshoreRegion=dt1.InshoreOffshoreRegion.astype('category') 
dt1.Latitude=pd.to_numeric(dt1.Latitude,errors='coerce') 
dt1.Longitude=pd.to_numeric(dt1.Longitude,errors='coerce') 
dt1.BottleNumber=pd.to_numeric(dt1.BottleNumber,errors='coerce') 
dt1.Depth=pd.to_numeric(dt1.Depth,errors='coerce') 
dt1.Temperature=pd.to_numeric(dt1.Temperature,errors='coerce') 
dt1.Salinity=pd.to_numeric(dt1.Salinity,errors='coerce') 
dt1.Density=pd.to_numeric(dt1.Density,errors='coerce') 
dt1.BeamTransmission=pd.to_numeric(dt1.BeamTransmission,errors='coerce') 
dt1.Oxygen=pd.to_numeric(dt1.Oxygen,errors='coerce') 
dt1.OxygenSaturation=pd.to_numeric(dt1.OxygenSaturation,errors='coerce') 
dt1.PAR=pd.to_numeric(dt1.PAR,errors='coerce') 
dt1.Fluorescence=pd.to_numeric(dt1.Fluorescence,errors='coerce') 
dt1.Chlorophyll=pd.to_numeric(dt1.Chlorophyll,errors='coerce') 
dt1.Phaeopigment=pd.to_numeric(dt1.Phaeopigment,errors='coerce')  
dt1.FilterCode=dt1.FilterCode.astype('category') 
dt1.Chlorophyllc3=pd.to_numeric(dt1.Chlorophyllc3,errors='coerce') 
dt1.Chlorophyllide=pd.to_numeric(dt1.Chlorophyllide,errors='coerce') 
dt1.Chlorophyllc2=pd.to_numeric(dt1.Chlorophyllc2,errors='coerce') 
dt1.Peridinin=pd.to_numeric(dt1.Peridinin,errors='coerce') 
dt1.Phaeophorbide=pd.to_numeric(dt1.Phaeophorbide,errors='coerce') 
dt1.x19butanoyloxyfucoxanthin=pd.to_numeric(dt1.x19butanoyloxyfucoxanthin,errors='coerce') 
dt1.Fucoxanthin=pd.to_numeric(dt1.Fucoxanthin,errors='coerce') 
dt1.Neoxanthin=pd.to_numeric(dt1.Neoxanthin,errors='coerce') 
dt1.Prasinoxanthin=pd.to_numeric(dt1.Prasinoxanthin,errors='coerce') 
dt1.x19hexanoyloxyfucoxanthin=pd.to_numeric(dt1.x19hexanoyloxyfucoxanthin,errors='coerce') 
dt1.Violaxanthin=pd.to_numeric(dt1.Violaxanthin,errors='coerce') 
dt1.Diadinoxanthin=pd.to_numeric(dt1.Diadinoxanthin,errors='coerce') 
dt1.Antheraxanthin=pd.to_numeric(dt1.Antheraxanthin,errors='coerce') 
dt1.Alloxanthin=pd.to_numeric(dt1.Alloxanthin,errors='coerce') 
dt1.Diatoxanthin=pd.to_numeric(dt1.Diatoxanthin,errors='coerce') 
dt1.Zeaxanthin=pd.to_numeric(dt1.Zeaxanthin,errors='coerce') 
dt1.Lutein=pd.to_numeric(dt1.Lutein,errors='coerce') 
dt1.Crocoxanthin=pd.to_numeric(dt1.Crocoxanthin,errors='coerce') 
dt1.Chlorophyllb=pd.to_numeric(dt1.Chlorophyllb,errors='coerce') 
dt1.Echinenone=pd.to_numeric(dt1.Echinenone,errors='coerce') 
dt1.DivinylChlorophylla=pd.to_numeric(dt1.DivinylChlorophylla,errors='coerce') 
dt1.Chlorophylla=pd.to_numeric(dt1.Chlorophylla,errors='coerce') 
dt1.Lycopene=pd.to_numeric(dt1.Lycopene,errors='coerce') 
dt1.Phaeophytin=pd.to_numeric(dt1.Phaeophytin,errors='coerce') 
dt1.AlphaCarotene=pd.to_numeric(dt1.AlphaCarotene,errors='coerce') 
dt1.BetaCarotene=pd.to_numeric(dt1.BetaCarotene,errors='coerce') 
dt1.Prasinophytes=pd.to_numeric(dt1.Prasinophytes,errors='coerce') 
dt1.Cryptophytes=pd.to_numeric(dt1.Cryptophytes,errors='coerce') 
dt1.MixedFlagellates=pd.to_numeric(dt1.MixedFlagellates,errors='coerce') 
dt1.Diatoms=pd.to_numeric(dt1.Diatoms,errors='coerce') 
dt1.Haptophytes=pd.to_numeric(dt1.Haptophytes,errors='coerce') 
dt1.PercentIrradiance=pd.to_numeric(dt1.PercentIrradiance,errors='coerce') 
dt1.PrimaryProduction=pd.to_numeric(dt1.PrimaryProduction,errors='coerce') 
dt1.PrimProdSTD=pd.to_numeric(dt1.PrimProdSTD,errors='coerce') 
dt1.FIREGain=pd.to_numeric(dt1.FIREGain,errors='coerce') 
dt1.FIREFvOverFm=pd.to_numeric(dt1.FIREFvOverFm,errors='coerce') 
dt1.FIRESigma=pd.to_numeric(dt1.FIRESigma,errors='coerce') 
dt1.FIRERho=pd.to_numeric(dt1.FIRERho,errors='coerce') 
dt1.BacterialAbundance=pd.to_numeric(dt1.BacterialAbundance,errors='coerce') 
dt1.ThymidineIncorporation=pd.to_numeric(dt1.ThymidineIncorporation,errors='coerce') 
dt1.LeucineIncorporation=pd.to_numeric(dt1.LeucineIncorporation,errors='coerce') 
dt1.HNA=pd.to_numeric(dt1.HNA,errors='coerce') 
dt1.LNA=pd.to_numeric(dt1.LNA,errors='coerce') 
dt1.DOC=pd.to_numeric(dt1.DOC,errors='coerce') 
dt1.POC=pd.to_numeric(dt1.POC,errors='coerce') 
dt1.N=pd.to_numeric(dt1.N,errors='coerce') 
dt1.PO4=pd.to_numeric(dt1.PO4,errors='coerce') 
dt1.SiO4=pd.to_numeric(dt1.SiO4,errors='coerce') 
dt1.NO2=pd.to_numeric(dt1.NO2,errors='coerce') 
dt1.NO3=pd.to_numeric(dt1.NO3,errors='coerce') 
dt1.NH4=pd.to_numeric(dt1.NH4,errors='coerce') 
dt1.NO3plusNO2=pd.to_numeric(dt1.NO3plusNO2,errors='coerce') 
dt1.O2mlperl=pd.to_numeric(dt1.O2mlperl,errors='coerce') 
dt1.O2umolperl=pd.to_numeric(dt1.O2umolperl,errors='coerce') 
dt1.DIC1=pd.to_numeric(dt1.DIC1,errors='coerce') 
dt1.DIC2=pd.to_numeric(dt1.DIC2,errors='coerce') 
dt1.Alkalinity1=pd.to_numeric(dt1.Alkalinity1,errors='coerce') 
dt1.Alkalinity2=pd.to_numeric(dt1.Alkalinity2,errors='coerce') 
dt1.DICTemperature=pd.to_numeric(dt1.DICTemperature,errors='coerce') 
dt1.DICSalinity=pd.to_numeric(dt1.DICSalinity,errors='coerce')  
dt1.Notes1=dt1.Notes1.astype('category')  
dt1.Notes2=dt1.Notes2.astype('category')
# =============================================================================


# =============================================================================
# Misc. Data Handling
# =============================================================================
#drop rows lacking values in core framework parameters
dt1 = dt1.dropna(subset=['DatetimeGMT','Latitude','Longitude','Depth'])
dt1 = dt1.reset_index(drop=True)

#change to DatetimeGMT to datetime object
dt1['DatetimeGMT'] = pd.to_datetime(dt1['DatetimeGMT'])

#rename columns
dt1 = dt1.rename(columns={'studyName': 'Cruise', 
                        'DatetimeGMT': 'Date',
                        'GridRegion': 'Region',
                        'NorthSouthRegion': 'NSRegion', 
                        'InshoreOffshoreRegion': 'IORegion', 
                        'FIREFvOverFm': 'FIRE_FvFm',
                        'Haptophytes': 'Type4Haptophytes', 
                        'PercentIrradiance': '%Irradiance', 
                        'AlphaCarotene': 'Alphacarotene',
                        'BetaCarotene': 'Betacarotene',
                        'BetaCarotene': 'Betacarotene',
                        'PAR': 'ctdPAR'})

#extract datetime info
dt1['Year'] = dt1['Date'].dt.year
dt1['Month'] = dt1['Date'].dt.month
dt1['Day'] = dt1['Date'].dt.day
dt1['Hour'] = dt1['Date'].dt.hour

#fix cruise strings
dt1['Cruise'] = dt1['Cruise'].str.replace('-','')

#fill some missing region info based on rounded grid line/station
condition = (dt1['RoundedGridLine']==-200) & (dt1['RoundedGridStation']==-40)
dt1.loc[condition, 'Region'] = 'FSC'
dt1.loc[condition, 'NSRegion'] = 'FS'
dt1.loc[condition, 'IORegion'] = 'C'

#change negative Divinyl Chla values to zero
dt1.loc[dt1['DivinylChlorophylla'] < 0,'DivinylChlorophylla'] = 0
#   ^ negative values come from manner of calculation; negative is equivalent to zero

#change Chla values of 0 to NaN (zero Chla is highly unlikely in reality)
dt1['Chlorophylla'] = dt1['Chlorophylla'].where(dt1['Chlorophylla'] > 0, other=np.NaN)

#fix negative depth values
dt1 = dt1[dt1['Depth'] > -10]
dt1.loc[dt1['Depth'] < 0,'Depth'] = 0

#assign a depth bin to each depth based on specific depth bin sizes
bin_size = 1
dt1['Depth'] = dt1['Depth'].apply(lambda x: math.ceil(x/bin_size))

#round depths to whole numbers
dt1['Depth'] = dt1['Depth'].round(0)

#drop excess columns
droprefs = ['JulianDay',
            'BottleNumber',
            'FilterCode',
            'Notes1',
            'Notes2',
            'DatetimeGMT_datetime']
dt1 = dt1.drop(columns=droprefs)

#reorder columns
columns_to_move = ['Cruise', 'Date', 'Year', 'Month', 'Day', 'Hour']
dt1 = dt1[columns_to_move + [col for col in dt1.columns if col not in columns_to_move]]
# =============================================================================


# =============================================================================
# Correct Bad HPLC Data in 2018
# =============================================================================
"""
REQUIRED  *(if the 2018 HPLC data in 2018 has not been updated on EDI)

Replaces bad values of alphacarotene and betacarotene in 2018 with correct values

For details, see the 'README_DataCorrection_2018HPLC' text file in the 
[local data] folder
"""

#load correct 2018 HPLC data
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("DataCorrection_2018HPLC_AlphaBetaCarotenes.csv")
loadpath = str(current_directory / absolute_path / filename)
corrhplc = pd.read_csv(loadpath)

# -----------------------------------------------------------------------------
# Conduct same misc. data handling
# -----------------------------------------------------------------------------
#drop rows lacking values in core framework parameters
corrhplc = corrhplc.dropna(subset=['DatetimeGMT','Latitude','Longitude','Depth'])
corrhplc = corrhplc.reset_index(drop=True)

#change to DatetimeGMT to datetime object
corrhplc['DatetimeGMT'] = pd.to_datetime(corrhplc['DatetimeGMT'])

#rename columns
corrhplc = corrhplc.rename(columns={'studyName': 'Cruise', 
                        'DatetimeGMT': 'Date',
                        'GridRegion': 'Region',
                        'NorthSouthRegion': 'NSRegion', 
                        'InshoreOffshoreRegion': 'IORegion',
                        'AlphaCarotene': 'Alphacarotene',
                        'BetaCarotene': 'Betacarotene'})

#extract datetime info
corrhplc['Year'] = corrhplc['Date'].dt.year
corrhplc['Month'] = corrhplc['Date'].dt.month
corrhplc['Day'] = corrhplc['Date'].dt.day
corrhplc['Hour'] = corrhplc['Date'].dt.hour

#fix cruise strings
corrhplc['Cruise'] = corrhplc['Cruise'].str.replace('-','')

#fill some missing region info based on rounded grid line/station
condition = (corrhplc['RoundedGridLine']==-200) & (corrhplc['RoundedGridStation']==-40)
corrhplc.loc[condition, 'Region'] = 'FSC'
corrhplc.loc[condition, 'NSRegion'] = 'FS'
corrhplc.loc[condition, 'IORegion'] = 'C'

#change negative Divinyl Chla values to zero
corrhplc.loc[corrhplc['DivinylChlorophylla'] < 0,'DivinylChlorophylla'] = 0
#   ^ negative values come from manner of calculation; negative is equivalent to zero

#change Chla values of 0 to NaN (zero Chla is highly unlikely in reality)
corrhplc['Chlorophylla'] = corrhplc['Chlorophylla'].where(corrhplc['Chlorophylla'] > 0, other=np.NaN)

#fix negative depth values
corrhplc = corrhplc[corrhplc['Depth'] > -10]
corrhplc.loc[corrhplc['Depth'] < 0,'Depth'] = 0

#assign a depth bin to each depth based on specific depth bin sizes
bin_size = 1
corrhplc['Depth'] = corrhplc['Depth'].apply(lambda x: math.ceil(x/bin_size))

#round depths to whole numbers
corrhplc['Depth'] = corrhplc['Depth'].round(0)

#drop excess columns
droprefs = ['JulianDay',
            'BottleNumber',
            'FilterCode',
            'Notes1',
            'Notes2']
corrhplc = corrhplc.drop(columns=droprefs)

#reorder columns
columns_to_move = ['Cruise', 'Date', 'Year', 'Month', 'Day', 'Hour']
corrhplc = corrhplc[columns_to_move + [col for col in corrhplc.columns if col not in columns_to_move]]
# -----------------------------------------------------------------------------
# Overwrite bad values with correct ones in dataframe
# -----------------------------------------------------------------------------
#merge corrected data
dt1_2018 = dt1[dt1['Year'] == 2018]
merged_df_2018 = pd.merge(dt1_2018, corrhplc[['Year', 'Event', 'Depth', 'Alphacarotene', 'Betacarotene']],
                          on=['Year', 'Event', 'Depth'], suffixes=('', '_correct'))

#replace bad values with correct ones
merged_df_2018['Alphacarotene'] = merged_df_2018['Alphacarotene_correct']
merged_df_2018['Betacarotene'] = merged_df_2018['Betacarotene_correct']

#drop excess columns
merged_df_2018 = merged_df_2018.drop(columns=['Alphacarotene_correct', 'Betacarotene_correct'])

#remove old rows
dt1_corr = dt1[dt1['Year'] != 2018]

#append corrected 2018 rows
dt1_corr = pd.concat([dt1_corr, merged_df_2018], ignore_index=True)

#reorder dataframe
dt1 = dt1_corr.sort_values(by='Year').reset_index(drop=True)
# =============================================================================


# =============================================================================
# Save Data as EDI Cruise Dataframe
# =============================================================================
current_directory = Path.cwd()
absolute_path = Path("local data/")
filename = Path("PALLTER_EDICruiseDataframe.csv")
savepath = str(current_directory / absolute_path / filename)

dt1.to_csv(savepath, index=False)
# =============================================================================
