#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 08:39:43 2022

@author: evelynm

big networks, starting from cis_network that is yet missing dependencies
"""

import geopandas as gpd
import pandas as pd
import pickle 

# on climada_petals branch feature/networks until merged into develop!!
from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import Graph
from climada_petals.engine.networks import nw_utils as nwu

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
iso3 = u_coords.country_to_iso(cntry)
path_save_cntry = PATH_SAVE + f'{iso3}/'

# =============================================================================
# Interdependencies
# =============================================================================

cis_network = Network(edges=gpd.read_feather(path_save_cntry+'pre_cis_nw_edges'), 
                      nodes=gpd.read_feather(path_save_cntry+'pre_cis_nw_nodes'))

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
