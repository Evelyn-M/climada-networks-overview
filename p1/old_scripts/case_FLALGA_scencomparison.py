#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:07:34 2022

@author: evelynm
"""
import numpy as np
import geopandas as gpd
import igraph as ig

# on climada_petals branch feature/networks until merged!!!
from climada_petals.engine.networks.base import Network
from climada_petals.engine.networks.nw_calcs import Graph, GraphCalcs

path_output_data = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/output_multi_cis/FLALGA'
save_path_cis = path_input_data+'/multi_cis_extracts/FLALGA'
save_path_graphs = path_output_data+'/graphs'


graph_s0_post = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_flalga.graphml'))
graph_s0_pre = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/basestate_graph_flalga.graphml'))
graph_s3h_pre = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/basestate_graph_flalga_S3h.graphml'))
graph_s3l_pre = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/basestate_graph_flalga_S3l.graphml'))


graph_s1l_post = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_flalga_S1l.graphml'))
graph_s2l_post = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_flalga_S2l.graphml'))
graph_s3l_post = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_flalga_S3l.graphml'))

graph_s1h_post = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_flalga_S1h.graphml'))
graph_s2h_post = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_flalga_S2h.graphml'))
graph_s3h_post = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_S3h.graphml'))


def service_dict():
    return {'power':'actual_supply_power line_people',
                    'healthcare': 'actual_supply_health_people',
                    'education':'actual_supply_education_people',
                    'telecom' : 'actual_supply_celltower_people',
                    'mobility' : 'actual_supply_road_people',
                    'water' : 'actual_supply_wastewater_people'}


def get_disaster_imp_tot(service, pre_graph, post_graph):
    
    serv_dict = service_dict()
    
    no_service_post = (1-np.array(post_graph.graph.vs.select(
        ci_type='people')[serv_dict[service]]))
    no_service_pre = (1-np.array(pre_graph.graph.vs.select(
        ci_type='people')[serv_dict[service]]))
    pop = np.array(pre_graph.graph.vs.select(
        ci_type='people')['counts'])
    
    return ((no_service_post-no_service_pre)*pop).sum()

def get_service_tot(service, graph):
    
    serv_dict = service_dict()
    
    no_service = (1-np.array(graph.graph.vs.select(
        ci_type='people')[serv_dict[service]]))
    pop = np.array(graph.graph.vs.select(
        ci_type='people')['counts'])
    
    return (no_service*pop).sum()


def get_disaster_imp_loc(service, pre_graph, post_graph):
    
    serv_dict = service_dict()
    
    no_service_post = (1-np.array(post_graph.graph.vs.select(
        ci_type='people')[serv_dict[service]]))
    no_service_pre = (1-np.array(pre_graph.graph.vs.select(
        ci_type='people')[serv_dict[service]]))
    
    geom = np.array(post_graph.graph.vs.select(
        ci_type='people')['geom_wkt'])
    
    return gpd.GeoSeries.from_wkt(
        geom[np.where((no_service_post-no_service_pre)>0)])
    

def get_impdict(pre_graph, post_graph, 
                services=['power', 'healthcare', 'education', 'telecom', 'mobility', 'water']):
    
    imp_dict = {}
    
    for service in services:
        imp_dict[service] = get_disaster_imp_tot(service, pre_graph, post_graph)
    
    return imp_dict

def get_servicetats_dict(graph, 
                         services=['power', 'healthcare', 'education', 'telecom', 'mobility', 'water']):
    
    servstats_dict = {}
    
    for service in services:
        servstats_dict[service] = get_service_tot(service, graph)
    
    return servstats_dict

def get_graphstats(graph):
    from collections import Counter
    stats_dict = {}
    stats_dict['no_edges'] = len(graph.graph.es)
    stats_dict['no_nodes'] = len(graph.graph.vs)
    stats_dict['edge_types'] = Counter(graph.graph.es['ci_type'])
    stats_dict['node_types'] = Counter(graph.graph.vs['ci_type'])
    return stats_dict
    
    
get_impdict(graph_s0_pre, graph_s0_post)
get_impdict(graph_s0_pre, graph_s1l_post)
get_impdict(graph_s0_pre, graph_s1h_post)
get_impdict(graph_s0_pre, graph_s2l_post)
get_impdict(graph_s0_pre, graph_s2h_post)
get_impdict(graph_s0_pre, graph_s3l_post)

get_impdict(graph_s3h_pre, graph_s3h_post)
get_impdict(graph_s3l_pre, graph_s3l_post)

get_servicetats_dict(graph_s3l_pre)
get_servicetats_dict(graph_s3l_post)

base_stats = get_graphstats(graph_s0_pre)

base_stats = get_graphstats(graph_s3l_pre)


mobility_loss = get_disaster_imp_loc( 'mobility', graph_s0_pre, graph_s0_post)
gpd.GeoSeries(mobility_loss.buffer(0.05).unary_union).to_file(path_output_data+'/exps/mobility_loss_model.shp')

hc_loss = get_disaster_imp_loc( 'healthcare', graph_s0_pre, graph_s0_post)
gpd.GeoSeries(hc_loss.buffer(0.05).unary_union).to_file(path_output_data+'/exps/healthcare_loss_model.shp')

educ_loss = get_disaster_imp_loc('education', graph_s0_pre, graph_s0_post)
gpd.GeoSeries(educ_loss.buffer(0.05).unary_union).to_file(path_output_data+'/exps/educ_loss_model.shp')

water_loss = get_disaster_imp_loc( 'water', graph_s0_pre, graph_s0_post)
gpd.GeoSeries(water_loss.buffer(0.05).unary_union).to_file(path_output_data+'/exps/water_loss_model.shp')


