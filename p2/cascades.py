#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:20:40 2022

@author: evelynm
"""

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import Graph
import climada_petals.engine.networks.nw_utils as nwu
import climada.util.coordinates as u_coords
import pickle
import pandas as pd
import geopandas as gpd

def all_cascades(graph_dict, graph_orig, df_dependencies, p_source='power_plant', p_sink='power_line', 
                  source_var='el_generation', demand_var='el_consumption',
                  preselect=False, store_graphs=False, store_results=True, path_save=None):
    service_dict = {}
    for event_name, graph_disr in graph_dict.items():
        graph_disr.cascade(df_dependencies, p_source, p_sink, source_var, 
                           demand_var, preselect=False)
        
        graph_disr = nwu.mark_disaster_impact_services(graph_disr, graph_orig)
        service_dict[event_name] = nwu.disaster_impact_allservices(
            graph_orig, graph_disr, services =['power', 'healthcare', 
                                               'education', 'telecom', 
                                               'mobility'])
        if store_graphs:
            ci_net = graph_disr.return_network()
            ci_net.nodes.to_feather(path_save+f'cis_nw_nodes_{event_name}')
            ci_net.edges.to_feather(path_save+f'cis_nw_edges_{event_name}')
        
        if store_results:
            ci_net = graph_disr.return_network()
            vars_to_keep_edges = ['ci_type', 'func_internal', 'func_tot', 'imp_dir', 'geometry']
            vars_to_keep_nodes = vars_to_keep_edges.copy() 
            vars_to_keep_nodes.extend([colname for colname in 
                                       ci_net.nodes.columns if 'actual_supply_' in colname])
            vars_to_keep_nodes.extend(['counts'])
            df_res = ci_net.nodes[ci_net.nodes.ci_type=='people'][vars_to_keep_nodes]
            df_res = df_res.append(ci_net.nodes[ci_net.nodes.ci_type=='health'][vars_to_keep_nodes])
            df_res = df_res.append(ci_net.nodes[ci_net.nodes.ci_type=='education'][vars_to_keep_nodes])
            df_res = df_res.append(ci_net.nodes[ci_net.nodes.ci_type=='celltower'][vars_to_keep_nodes])
            df_res = df_res.append(ci_net.nodes[ci_net.nodes.ci_type=='power_plant'][vars_to_keep_nodes])
            df_res = df_res.append(ci_net.edges[ci_net.edges.ci_type=='power_line'][vars_to_keep_edges])
            df_res = df_res.append(ci_net.edges[ci_net.edges.ci_type=='road'][vars_to_keep_edges])
            df_res.to_feather(path_save+f'cascade_results_{event_name}')

    return service_dict
    
# =============================================================================
# Run from command line
# =============================================================================

import sys
graph_dict, path_graph, cntry = sys.argv[1:]

PATH_DEPS = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/dependencies_default.csv'
PATH_SAVE = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/p2/'
iso3 = u_coords.country_to_iso(cntry)
path_save_cntry = PATH_SAVE + f'{iso3}/'

df_dependencies = pd.read_csv(PATH_DEPS)

cis_network = Network(edges=gpd.read_feather(path_save_cntry+'cis_nw_edges'), 
                     nodes=gpd.read_feather(path_save_cntry+'cis_nw_nodes'))
cis_network.nodes = cis_network.nodes.drop('name', axis=1)
graph_orig = Graph(cis_network, directed=True) 

service_disr_dict = all_cascades(graph_dict, graph_orig, df_dependencies, 
                            p_source='power_plant', p_sink='power_line', 
                            source_var='el_generation', demand_var='el_consumption',
                            preselect=False, store_graphs=False, 
                            store_results=True, path_save=path_save_cntry)

with open(path_save_cntry+f'service_stats_{iso3}.pkl', 'wb') as f:
     pickle.dump(service_disr_dict, f)


del graph_dict
del graph_orig