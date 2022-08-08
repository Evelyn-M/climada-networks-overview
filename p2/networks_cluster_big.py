#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 08:39:43 2022

@author: evelynm
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import shapely
import os
import pickle 

# on climada_petals branch feature/networks until merged into develop!!
from climada_petals.engine.networks.nw_preps import NetworkPreprocess, PowerlinePreprocess, RoadPreprocess
from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import Graph
from climada_petals.engine.networks import nw_utils as nwu
from climada_petals.entity.exposures.openstreetmap import osm_dataloader as osm
from climada_petals.util.constants import DICT_GEOFABRIK

from climada.util import coordinates as u_coords

# general paths & constants
PATH_DATA = '/cluster/work/climate/evelynm/nw_inputs/'
PATH_DATA_OSM = PATH_DATA +'osm_countries/'
PATH_DATA_HVMV = PATH_DATA +'power/hvmv_global.shp'
PATH_DATA_PP = PATH_DATA +'power/global_power_plant_database.csv'
PATH_DATA_CT = PATH_DATA +'opencellid_global_1km_int.tif'
PATH_DATA_POP = PATH_DATA + 'worldpop/'
PATH_DEPS = PATH_DATA+ 'dependencies/dependencies_default.csv'
PATH_SAVE = '/cluster/work/climate/evelynm/nw_outputs/'
PATH_EL_CONS_GLOBAL = PATH_DATA +'power/final_consumption_iea_global.csv'

import sys
cntry = sys.argv[1]

# =============================================================================
# Load Infra Data
# =============================================================================

iso3 = u_coords.country_to_iso(cntry)

path_osm_cntry = PATH_DATA_OSM+DICT_GEOFABRIK[iso3][-1]+'-latest.osm.pbf'
path_worldpop_cntry = PATH_DATA_POP + f'{iso3.lower()}_ppp_2020_1km_Aggregated_UNadj.tif'
path_el_consump_cntry = PATH_DATA + f'power/Electricity consumption per capita - {cntry}.csv'
path_elgen_cntry = PATH_DATA + f'power/Electricity generation by source - {cntry}.csv'
path_el_imp_exp_cntry = PATH_DATA + f'power/Electricity imports vs. exports - {cntry}.csv'
path_save_cntry = PATH_SAVE + f'{iso3}/'

if not os.path.isdir(path_save_cntry):
    os.mkdir(path_save_cntry)
    
__, cntry_shape = u_coords.get_admin1_info([cntry])
cntry_shape = shapely.ops.unary_union([shp for shp in cntry_shape[iso3]])
osm.OSMRaw().get_data_geofabrik(iso3, file_format='pbf', save_path=PATH_DATA_OSM)
CntryFileQuery = osm.OSMFileQuery(path_osm_cntry)

# POWER LINES
gdf_powerlines = gpd.read_file(PATH_DATA_HVMV, mask=cntry_shape)
gdf_powerlines['osm_id'] = 'n/a'
gdf_powerlines['ci_type'] = 'n/a' #random, preprocessing needs another column
gdf_powerlines = gdf_powerlines[['osm_id', 'geometry', 'ci_type']]

# POWER PLANTS
# try WRI pp database, then OSM
gdf_pp_world = gpd.read_file(PATH_DATA_PP, crs='EPSG:4326')
gdf_pp = gdf_pp_world[gdf_pp_world.country==f'{iso3}'][
    ['estimated_generation_gwh_2017','latitude','longitude',
     'name']] 
del gdf_pp_world
if not gdf_pp.empty:
    gdf_pp = gpd.GeoDataFrame(
        gdf_pp, geometry=gpd.points_from_xy(gdf_pp.longitude, gdf_pp.latitude))
else:
    gdf_pp = CntryFileQuery.retrieve_cis('power')
    if len(gdf_pp[gdf_pp.power=='plant'])>1:
        gdf_pp = gdf_pp[gdf_pp.power=='plant']
    else:
        # last 'resort': take generators frmom OSM
        gdf_pp = gdf_pp[gdf_pp.power=='generator']
    gdf_pp['geometry'] = gdf_pp.geometry.apply(lambda geom: geom.centroid)
    gdf_pp = gdf_pp[['name', 'power', 'geometry']]

# PEOPLE
nwu.get_worldpop_data(iso3, PATH_DATA_POP)
gdf_people = nwu.load_resampled_raster(path_worldpop_cntry, 1)
gdf_people = gdf_people[gdf_people.counts>=10].reset_index(drop=True)

# assign electricity consumption & production
gdf_people, gdf_pp = nwu.PowerFunctionalData().assign_el_prod_consump(
    gdf_people, gdf_pp, iso3, PATH_EL_CONS_GLOBAL)

# HEALTH FACILITIES
# from osm
gdf_health = CntryFileQuery.retrieve_cis('healthcare') 
gdf_health['geometry'] = gdf_health.geometry.apply(lambda geom: geom.centroid)
gdf_health = gdf_health[['name', 'geometry']]
gdf_health = gdf_health[gdf_health.geometry.within(cntry_shape)]

# EDUC. FACILITIES
# from osm
gdf_educ = CntryFileQuery.retrieve_cis('education')
gdf_educ['geometry'] = gdf_educ.geometry.apply(lambda geom: geom.centroid)
gdf_educ = gdf_educ[['name', 'geometry']]
gdf_educ = gdf_educ[gdf_educ.geometry.within(cntry_shape)]

# TELECOM
# cells from rasterized opencellID (via WB)
path_ct_cntry = path_save_cntry+'celltowers.tif'
if not Path(path_ct_cntry).is_file():
    if cntry_shape.type=='Polygon':
        geo_mask = [cntry_shape]
    else:
        geo_mask = [mp for mp in cntry_shape]
    meta_ct, arr_ct = u_coords.read_raster(PATH_DATA_CT, src_crs={'epsg':'4326'},
                                           geometry=geo_mask)
    u_coords.write_raster(path_ct_cntry, arr_ct, meta_ct)
gdf_cells = nwu.load_resampled_raster(path_ct_cntry, 1/5)

# ROADS
# from osm
gdf_roads = CntryFileQuery.retrieve_cis('main_road')
gdf_roads = gdf_roads[gdf_roads.geometry.type=='LineString']
gdf_roads = gdf_roads[['osm_id','highway', 'geometry']]
gdf_roads = gdf_roads[gdf_roads.within(cntry_shape)]

# =============================================================================
# # Graphs
# =============================================================================

# POWER LINES
gdf_power_edges, gdf_power_nodes = PowerlinePreprocess().preprocess(
    gdf_edges=gdf_powerlines)
power_network = Network(gdf_power_edges, gdf_power_nodes)
power_graph = Graph(power_network, directed=False)
iter_count = 0
while (len(power_graph.graph.clusters())>1) and (iter_count<8):
    iter_count+=1
    power_graph.link_clusters(dist_thresh=200000)
power_network = Network().from_graphs([power_graph.graph.as_directed()])
power_network.nodes.to_feather(path_save_cntry+'power_nw_nodes')
power_network.edges.to_feather(path_save_cntry+'power_nw_edges')

# power_graph.graph.clusters().summary(): 

# PEOPLE
__, gdf_people_nodes = NetworkPreprocess('people').preprocess(
    gdf_nodes=gdf_people)
people_network = Network(nodes=gdf_people_nodes)
people_network.nodes.to_feather(path_save_cntry+'people_nw_nodes')

# POWER PLANTS
__, gdf_pp_nodes = NetworkPreprocess('power_plant').preprocess(
    gdf_nodes=gdf_pp)
pplant_network = Network(nodes=gpd.GeoDataFrame(gdf_pp_nodes))
pplant_network.nodes.to_feather(path_save_cntry+'pplant_nw_nodes')

# HEALTHCARE
__, gdf_health_nodes = NetworkPreprocess('health').preprocess(
    gdf_nodes=gdf_health)
health_network = Network(nodes=gdf_health_nodes)
health_network.nodes.to_feather(path_save_cntry+'health_nw_nodes')

# EDUC
__, gdf_educ_nodes = NetworkPreprocess('education').preprocess(
    gdf_nodes=gdf_educ)
educ_network = Network(nodes=gdf_educ_nodes)
educ_network.nodes.to_feather(path_save_cntry+'educ_nw_nodes')

# ROAD
gdf_road_edges, gdf_road_nodes = RoadPreprocess().preprocess(
    gdf_edges=gdf_roads)
road_network = Network(gdf_road_edges, gdf_road_nodes)
# easy workaround for doubling edges
road_graph = Graph(road_network, directed=False)
iter_count = 0
while (len(road_graph.graph.clusters())>1) and (iter_count<4):
    iter_count+=1
    road_graph.link_clusters(dist_thresh=30000)
road_network = Network().from_graphs([road_graph.graph.as_directed()])
road_network.nodes.to_feather(path_save_cntry+'road_nw_nodes')
road_network.edges.to_feather(path_save_cntry+'road_nw_edges')

# TELECOM
__, gdf_tele_nodes = NetworkPreprocess('celltower').preprocess(gdf_nodes=gdf_cells)
tele_network = Network(nodes=gdf_tele_nodes)
tele_network.nodes.to_feather(path_save_cntry+'tele_nw_nodes')

# MULTINET
cis_network = Network.from_nws([pplant_network, power_network,
                                      people_network, health_network, educ_network,
                                      road_network, tele_network])

cis_network.initialize_funcstates()


# =============================================================================
# Interdependencies
# =============================================================================
df_dependencies = pd.read_csv(PATH_DEPS, sep=',', header=0)

cis_network.nodes = cis_network.nodes.drop('name', axis=1)
cis_graph = Graph(cis_network, directed=True)
# create "missing physical structures" - needed for real world flows
cis_graph.link_vertices_closest_k('power_line', 'power_plant', link_name='power_line', bidir=True, k=1)
cis_graph.link_vertices_closest_k('road', 'people',  link_name='road', 
                                  dist_thresh=df_dependencies[
                                      df_dependencies.source=='road'].thresh_dist.values[0],
                                  bidir=True, k=1)
cis_graph.link_vertices_closest_k('road', 'health',  link_name='road',  bidir=True, k=1)
cis_graph.link_vertices_closest_k('road', 'education', link_name='road',  bidir=True, k=1)


for __, row in df_dependencies.iterrows():
    cis_graph.place_dependency(row.source, row.target, 
                               single_link=row.single_link,
                               access_cnstr=row.access_cnstr, 
                               dist_thresh=row.thresh_dist,
                               preselect=False)
cis_network = cis_graph.return_network()

# =============================================================================
# Base State
# =============================================================================

cis_network.initialize_funcstates()
for __, row in df_dependencies.iterrows():
    cis_network.initialize_capacity(row.source, row.target)
for __, row in df_dependencies[
        df_dependencies['type_I']=='enduser'].iterrows():
    cis_network.initialize_supply(row.source)
cis_network.nodes = cis_network.nodes.drop('name', axis=1)
    
cis_graph = Graph(cis_network, directed=True)
cis_graph.cascade(df_dependencies, p_source='power_plant', p_sink='power_line', 
                  source_var='el_generation', demand_var='el_consumption',
                  preselect=False)
cis_network = cis_graph.return_network()

cis_network.nodes.to_feather(path_save_cntry+'cis_nw_nodes')
cis_network.edges.to_feather(path_save_cntry+'cis_nw_edges')

base_stats = nwu.number_noservices(cis_graph,
                         services=['power', 'healthcare', 'education', 'telecom', 'mobility'])

with open(path_save_cntry +f'base_stats_{iso3}.pkl', 'wb') as f:
    pickle.dump(base_stats, f) 
