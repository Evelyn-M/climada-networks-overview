#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 18:25:07 2022

@author: evelynm
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import igraph as ig

# on climada_petals branch feature/networks until merged!!!
from climada_petals.engine.networks.nw_preps import NetworkPreprocess                                                  
from climada_petals.engine.networks.base import Network
from climada_petals.engine.networks.nw_calcs import Graph, GraphCalcs

# on climada_python branch feature/lines_polygons_exp until merged into develop!!
from climada.util import lines_polys_handler as u_lp_handler
from climada.hazard import TropCyclone
from climada.entity.entity_def import Entity
from climada.entity.exposures.base import Exposures
from climada.engine import Impact
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet

# =============================================================================
# Constants & filepaths
# =============================================================================
path_input_data =  '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data'
path_output_data = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/output_multi_cis/FLALGA'
path_deps = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/dependencies_FLALGA.csv'

save_path_cis = path_input_data+'/multi_cis_extracts/FLALGA'
save_path_graphs = path_output_data+'/graphs'
save_path_exps = path_output_data+'/exps'
save_path_figs = path_output_data+'/figures'

PER_CAP_ECONSUMP = 11.519

# RELOAD IF NECESSARY
cis_graph = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/basestate_graph_flalga.graphml'))
cis_network = cis_graph.return_network()
cis_network.nodes['geometry'] = cis_network.nodes.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)
cis_network.edges['geometry'] = cis_network.edges.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)

cis_graph_disr = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_flalga.graphml'))
cis_network_disr = cis_graph_disr.return_network()

cis_network.nodes['imp_dir' ] = cis_network_disr.nodes['imp_dir']
cis_network.edges['imp_dir' ] = cis_network_disr.edges['imp_dir']

del cis_graph_disr
del cis_network_disr

# IMPACT - FUNCTIONALITY LEVEL (low thresh)
# THRESH_PLINE = 0.15
# THRESH_ROAD = 0.15
# THRESH_HEALTH = 0.15
# THRESH_EDUC =0.15
# THRESH_WW = 0.15
# THRESH_CT = 0.15

# IMPACT - FUNCTIONALITY LEVEL (low thresh)
THRESH_PLINE = 0.5
THRESH_ROAD =0.5
THRESH_HEALTH = 0.5
THRESH_EDUC =0.5
THRESH_WW = 0.5
THRESH_CT =0.5


cond_fail_pline = ((cis_network.edges.ci_type=='power line') & 
                   (cis_network.edges.imp_dir>=THRESH_PLINE))

cond_fail_road = ((cis_network.edges.ci_type=='road') & 
                   (cis_network.edges.imp_dir>=THRESH_ROAD))

cond_fail_health = ((cis_network.nodes.ci_type=='health') & 
                   (cis_network.nodes.imp_dir>THRESH_HEALTH))

cond_fail_educ = ((cis_network.nodes.ci_type=='educ') & 
                   (cis_network.nodes.imp_dir>THRESH_EDUC))

cond_fail_wwater = ((cis_network.nodes.ci_type=='wastewater') & 
                   (cis_network.nodes.imp_dir>THRESH_WW))

cond_fail_tele = ((cis_network.nodes.ci_type=='celltower') & 
                   (cis_network.nodes.imp_dir>THRESH_CT))

for fail_cond in [cond_fail_pline, cond_fail_road]:
    cis_network.edges.func_internal.loc[fail_cond] = 0

for fail_cond in [cond_fail_health, cond_fail_educ, cond_fail_wwater, 
                  cond_fail_tele]:
    cis_network.nodes.func_internal.loc[fail_cond] = 0

# TOTAL FUNC-STATES
cis_network.edges['func_tot'] = [np.min([func_internal, func_tot]) 
                                 for func_internal, func_tot in 
                                 zip(cis_network.edges.func_internal, 
                                     cis_network.edges.func_tot)]

cis_network.nodes['func_tot'] = [np.min([func_internal, func_tot]) 
                                 for func_internal, func_tot in 
                                 zip(cis_network.nodes.func_internal, 
                                     cis_network.nodes.func_tot)]

# =============================================================================
# Failure Cascades                  
# =============================================================================
df_dependencies = pd.read_csv(path_deps, sep=',', header=0)

cis_network.nodes = cis_network.nodes.drop('name', axis=1)
cis_graph = Graph(cis_network, directed=True)
cis_graph.cascade(df_dependencies, p_source='power plant', p_sink='power line',
                                        per_cap_cons=PER_CAP_ECONSUMP, source_var='NET_GEN')
cis_network = cis_graph.return_network()

# SAVE 
def geoms_to_wkt(geom):
    if not geom:
        return 'GEOMETRYCOLLECTION EMPTY'
    elif not geom.is_empty:
        return geom.wkt
    else:
        return 'GEOMETRYCOLLECTION EMPTY'
    
cis_network.nodes['geom_wkt'] = cis_network.nodes.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network.edges['geom_wkt'] = cis_network.edges.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network.nodes = cis_network.nodes.drop('name', axis=1)
cis_graph = Graph(cis_network, directed=True)
cis_graph.graph.write(save_path_graphs+'/disrstate_graph_flalga_S2h.graphml', format='graphml')


# =============================================================================
# Basic service Access stats - disrupted state       
# =============================================================================

# Selectors
bool_powersupply = cis_network.nodes['actual_supply_power line_people']>0
bool_healthsupply = cis_network.nodes.actual_supply_health_people>0
bool_educsupply = cis_network.nodes.actual_supply_education_people>0
bool_telesupply = cis_network.nodes.actual_supply_celltower_people>0
bool_watersupply = cis_network.nodes.actual_supply_wastewater_people>0
bool_mobilitysupply = cis_network.nodes.actual_supply_road_people>0

# Stats
ppl_served_power2 = cis_network.nodes[bool_powersupply].counts.sum() # 
ppl_nonserved_power2 = cis_network.nodes[~bool_powersupply].counts.sum() # 

ppl_served_health2 = cis_network.nodes[bool_healthsupply].counts.sum() # 
ppl_nonserved_health2 = cis_network.nodes[~bool_healthsupply].counts.sum() # 

ppl_served_education2 = cis_network.nodes[bool_educsupply].counts.sum() # 
ppl_nonserved_education2 = cis_network.nodes[~bool_educsupply].counts.sum() # 

ppl_served_telecom2 = cis_network.nodes[bool_telesupply].counts.sum() # 
ppl_nonserved_telecom2 = cis_network.nodes[~bool_telesupply].counts.sum() # 

ppl_served_water2 = cis_network.nodes[bool_watersupply].counts.sum() # 
ppl_nonserved_water2 = cis_network.nodes[~bool_watersupply].counts.sum() # 

ppl_served_road2 = cis_network.nodes[bool_mobilitysupply].counts.sum() # 
ppl_nonserved_road2 = cis_network.nodes[~bool_mobilitysupply].counts.sum() # 
