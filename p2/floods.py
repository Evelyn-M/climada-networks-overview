#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:55:10 2022

@author: evelynm
"""

import json
import glob
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from datetime import datetime
import shapely

from climada.hazard import Hazard
from climada.util.coordinates import get_admin1_info, country_to_iso

import warnings
warnings.filterwarnings('ignore')

PATH_FLOODS = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data/hazards/cloud_to_street_db'
PATH_SAVE = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data/hazards'

# =============================================================================
# Define flood handling functions
# =============================================================================

def get_flood_infodf(folder_path):
    """
    get an overview dataframe on all cloud to street floods available,
    incl. eventid, countries affected and tif-file location
    """
    folder_names = glob.glob(folder_path + '/DFO*')
    json_filepaths = [folder_name+'/'+folder_name[-34:-26]+'_properties.json' 
                      for folder_name in folder_names]
    tif_filepaths = [folder_name+'/'+folder_name[-34:]+'.tif'
                     for folder_name in folder_names]
    # overview dataframe on all file contents
    with open(json_filepaths[0]) as json_file:
            df_floodinfo = pd.DataFrame(json.load(json_file), index=[0])        
    for ix, json_filepath in enumerate(json_filepaths[1:]):
        with open(json_filepath) as json_file:
            df_floodinfo = df_floodinfo.append(pd.DataFrame(json.load(json_file), 
                                                            index=[ix+1]))
    df_floodinfo['countries'] =[country.split(', ') if type(country)==str else country
                                for country in df_floodinfo.countries]
    df_floodinfo.countries.fillna('', inplace=True)
    df_floodinfo['flood_file'] = tif_filepaths
    df_floodinfo['date'] = df_floodinfo.began.apply(lambda dtstr: 
                                                    datetime.strptime(dtstr, '%Y-%m-%d').date().toordinal())
    df_floodinfo['iso3'] = df_floodinfo.cc.apply(lambda isostr: 
                                              isostr.split(', '))
    df_floodinfo.pop('cc')
    return df_floodinfo

def search_cntry_floods(iso3, df_floodinfo):
    """
    select relevant flood files from all flood files
    """
    return df_floodinfo[df_floodinfo.iso3.apply(lambda x: iso3 in x)]

def get_flood_haz(df_cntryflood, cntry_shape):
    """
    load all relevant floods as a climada hazard object
    """
    list_fl = []
    for __, row in df_cntryflood.iterrows():
        perm_water = Hazard.from_raster(
            files_intensity=row.flood_file, 
            files_fraction=None, band=[5], haz_type='FL', geometry=cntry_shape)
        perm_water.intensity.data = np.nan_to_num(perm_water.intensity.data)  
        
        haz  = Hazard.from_raster(
            files_intensity=row.flood_file, files_fraction=None, band=[1], 
            haz_type='FL', geometry=cntry_shape)
        
        haz.intensity.data = np.nan_to_num(haz.intensity.data)
        haz.intensity = haz.intensity-perm_water.intensity
        haz.intensity.data[haz.intensity.data<0] = 0
        haz.event_name = [row.flood_file[-38:-30]]
        haz.event_id = np.array([row.id])
        haz.date = np.array([row.date])
        haz.centroids.set_meta_to_lat_lon()
        haz.check()
        list_fl.append(haz)
        
    return list_fl, Hazard('FL').concat(list_fl)

def unique_countries(df_floodinfo):
    return set([cntry for cntry_list in df_floodinfo.countries.values.flatten() 
            for cntry in cntry_list])
# =============================================================================
# Run from command line
# =============================================================================
import sys 

cntry = sys.argv[1]
cntry_iso3 = country_to_iso(cntry)

df_floodinfo = get_flood_infodf(PATH_FLOODS) 
#df_floodinfo.to_csv(path_floods+'/flood_metainfo.csv')
#df_floodinfo = pd.read_csv(path_floods+'/flood_metainfo.csv') 

df_cntryflood = search_cntry_floods((cntry_iso3), df_floodinfo)
__, cntry_shape = get_admin1_info([cntry_iso3])

cntry_shape = unary_union([shp for shp in cntry_shape[cntry_iso3]])
if type(cntry_shape) == shapely.geometry.polygon.Polygon:
    cntry_shape = [cntry_shape]  
    
fl_list, floods = get_flood_haz(df_cntryflood, cntry_shape)
floods.write_hdf5(f'{PATH_SAVE}/flood_{cntry_iso3}.hdf5')
    
del floods