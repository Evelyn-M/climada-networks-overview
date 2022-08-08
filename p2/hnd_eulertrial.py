#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trial euler
@author: evelynm
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import pickle
import shapely
from cartopy.io import shapereader
from climada_petals.entity.exposures.black_marble import country_iso_geom

# on climada_petals branch feature/networks until merged!!!
from climada_petals.engine.networks.nw_preps import NetworkPreprocess, PowerlinePreprocess, RoadPreprocess
from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import Graph
from climada_petals.engine.networks import nw_utils as nwu
from climada_petals.entity.exposures.openstreetmap.osm_dataloader import OSMFileQuery

# on climada_python branch feature/lines_polygons_exp until merged into develop!!
from climada.util import coordinates as u_coords
from climada.util import lines_polys_handler as u_lp_handler
from climada.hazard import TCTracks, Centroids, TropCyclone
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.engine import Impact

# general paths & constants
PATH_DATA = '/cluster/work/climate/evelynm/'
PATH_INPUTS = PATH_DATA+'nw_inputs/'
PATH_OUTPUTS = PATH_DATA+'nw_outputs/'
PATH_HVMV_DATA = PATH_INPUTS +'power/hvmv_global.gpkg'
PATH_DATA_PP = PATH_INPUTS +'power/global_power_plant_database.csv'
PATH_DATA_CT = PATH_INPUTS +'cell_towers/opencellid_global_1km_int.tif'

# country-specific
path_worldpop_hnd = PATH_INPUTS + 'worldpop/hnd_ppp_2020_1km_Aggregated_UNadj.tif'
path_osm_hnd = PATH_INPUTS +'osm_countries/honduras-latest.osm.pbf'
path_deps_hnd = PATH_INPUTS+'dependencies/dependencies_HND.csv'
path_save_hnd = PATH_OUTPUTS + 'HND/'
path_el_consump_hnd = PATH_INPUTS +'power/Electricity final consumption by sector - Honduras.csv'
path_el_imp_exp_hnd = PATH_INPUTS + 'power/Electricity imports vs. exports - Honduras.csv'

# CI network retrieval - HND
# Instantiate object for OSM filequeries
hnd_shape = shapereader.natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
hnd_shape = shapereader.Reader(hnd_shape)
# # province names for Metro Manila
prov_names = {'Honduras': []}
polygon_hnd, __ = country_iso_geom(prov_names, hnd_shape)
hnd_shape = polygon_hnd['HND'][-1]

HNDFileQuery = OSMFileQuery(path_osm_hnd)
# POWER LINES
gdf_powerlines = gpd.read_file(PATH_HVMV_DATA, mask=hnd_shape) # from gridfinder
gdf_powerlines['osm_id'] = 'n/a'
gdf_powerlines.to_feather(path_save_hnd+'powerlines')

# POWER PLANTS
# from pp database
gdf_pp_world = gpd.read_file(PATH_DATA_PP, crs='EPSG:4326')
gdf_pp = gdf_pp_world[gdf_pp_world.country=='HND'][
    ['estimated_generation_gwh_2017','latitude','longitude',
     'name']] # del gdf_pp_world
gdf_pp = gpd.GeoDataFrame(gdf_pp, geometry=gpd.points_from_xy(gdf_pp.longitude,
                                                          gdf_pp.latitude))
gdf_pp = nwu.PowerFunctionalData().assign_esupply_iea(
    gdf_pp, path_el_imp_exp_hnd, path_el_consump_hnd)
del gdf_pp_world
gdf_pp.to_feather(path_save_hnd+'powerplants')

# PEOPLE
# reproject 1km2 to 5x5km grid:
gdf_people = nwu.load_resampled_raster(path_worldpop_hnd, 1)
gdf_people = nwu.PowerFunctionalData().assign_edemand_iea(
    gdf_people, path_el_consump_hnd)
gdf_people.to_feather(path_save_hnd+'people')

# HEALTH FACILITIES
# from osm
gdf_health = HNDFileQuery.retrieve_cis('healthcare') 
gdf_health['geometry'] = gdf_health.geometry.apply(lambda geom: geom.centroid)
gdf_health = gdf_health[['name', 'geometry']]
gdf_health.to_feather(path_save_hnd+'health')

# EDUC. FACILITIES
# from osm
gdf_educ = HNDFileQuery.retrieve_cis('education')
gdf_educ['geometry'] = gdf_educ.geometry.apply(lambda geom: geom.centroid)
gdf_educ = gdf_educ[['name', 'geometry']]
gdf_educ.to_feather(path_save_hnd+'education')

# TELECOM
# cells from rasterized opencellID (via WB)
meta_ct, arr_ct = u_coords.read_raster(PATH_DATA_CT, src_crs={'epsg':'4326'},
                                           geometry=[mp for mp in hnd_shape])

# ROADS
# from osm
gdf_roads = HNDFileQuery.retrieve_cis('road') 
gdf_roads = gdf_roads[['osm_id','highway','name', 'geometry']]
gdf_roads = gdf_roads[[x in ['primary', 'secondary', 'tertiary',
                             'primary_link', 'secondary_link', 'tertiary_link'] for x in gdf_roads.highway]]
gdf_roads.to_feather(path_save_hnd+'roads')

