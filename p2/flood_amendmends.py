#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:02:38 2022

@author: evelynm
"""

import os
import pickle
from climada.hazard import Hazard
import pandas as pd

parent_folder_path = '/cluster/work/climate/evelynm/nw_outputs/' 
successful_calcs = []
unsuccessful_calcs = []

iso_list = ['AFG',
 'AGO',
 'ALB',
 'AND',
 'ARE',
 'ARG',
 'ARM',
 'ATG',
 'AUS',
 'AUT',
 'AZE',
 'BDI',
 'BEL',
 'BEN',
 'BFA',
 'BGD',
 'BGR',
 'BHR',
 'BHS',
 'BIH',
 'BLR',
 'BLZ',
 'BOL',
 'BRN',
 'BTN',
 'BWA',
 'CAF',
 'CAN',
 'CHE',
 'CHL',
 'CIV',
 'CMR',
 'COD',
 'COG',
 'COL',
 'CRI',
 'CUB',
 'CYP',
 'CZE',
 'DEU',
 'DJI',
 'DMA',
 'DNK',
 'DOM',
 'DZA',
 'ECU',
 'EGY',
 'ERI',
 'ESH',
 'ESP',
 'EST',
 'ETH',
 'FIN',
 'FJI',
 'FRA',
 'GBR',
 'GEO',
 'GHA',
 'GIN',
 'GLP',
 'GMB',
 'GNB',
 'GRC',
 'GTM',
 'GUY',
 'HKG',
 'HND',
 'HRV',
 'HTI',
 'HUN',
 'IDN',
 'IMN',
 'IRL',
 'IRN',
 'IRQ',
 'ISR',
 'ITA',
 'JAM',
 'JOR',
 'JPN',
 'KAZ',
 'KEN',
 'KGZ',
 'KHM',
 'KOR',
 'KWT',
 'LAO',
 'LBN',
 'LBR',
 'LKA',
 'LSO',
 'LTU',
 'LUX',
 'LVA',
 'MAC',
 'MAR',
 'MDA',
 'MDG',
 'MEX',
 'MKD',
 'MLI',
 'MMR',
 'MNG',
 'MOZ',
 'MRT',
 'MWI',
 'MYS',
 'NAM',
 'NER',
 'NGA',
 'NIC',
 'NLD',
 'NPL',
 'NZL',
 'OMN',
 'PAK',
 'PAN',
 'PER',
 'PHL',
 'PNG',
 'POL',
 'PRI',
 'PRK',
 'PRT',
 'PRY',
 'PSE',
 'QAT',
 'ROU',
 'RWA',
 'SAU',
 'SDN',
 'SEN',
 'SGP',
 'SLE',
 'SLV',
 'SMR',
 'SOM',
 'SRB',
 'SUR',
 'SVK',
 'SVN',
 'SWZ',
 'SYR',
 'TCA',
 'TCD',
 'TGO',
 'THA',
 'TJK',
 'TKM',
 'TUN',
 'TUR',
 'TWN',
 'TZA',
 'UGA',
 'UKR',
 'URY',
 'UZB',
 'VEN',
 'VNM',
 'YEM',
 'ZAF',
 'ZMB',
 'ZWE']

def search_cntry_floods(iso3, df_floodinfo):
    """
    select relevant flood files from all flood files
    """
    return df_floodinfo[df_floodinfo.iso3.apply(lambda x: iso3 in x)]

df_floodinfo = pd.read_csv(parent_folder_path+'/flood_metainfo.csv') 

for iso3 in iso_list:
    d = os.path.join(parent_folder_path, iso3)
    if os.path.isdir(d):
        if os.path.isfile(d+f'/flood_{iso3}.hdf5'):
            cntry_floodinfo = search_cntry_floods(iso3, df_floodinfo)
            try:
                haz = Hazard.from_hdf5(d+f'/flood_{iso3}.hdf5')
                haz.date = cntry_floodinfo.date.values
                haz.event_id = cntry_floodinfo.id.values
                haz.check()
                haz.write_hdf5(d+f'/flood_{iso3}.hdf5')
                successful_calcs.append(iso3)
                del haz
            except:
                unsuccessful_calcs.append(iso3)
                continue

with open(f"{parent_folder_path}success", "wb") as fp:   #Pickling
    pickle.dump(successful_calcs, fp)
 
with open(f"{parent_folder_path}fail", "wb") as fp:   #Pickling
    pickle.dump(unsuccessful_calcs, fp)        
            
