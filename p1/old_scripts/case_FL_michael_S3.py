#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis:
    Change Dependencies f
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

# path_deps = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/dependencies_FLALGA_4.csv'
# path_deps = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/dependencies_FLALGA_S3l.csv'
path_deps = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/dependencies_FLALGA_S3l.csv'

save_path_cis = path_input_data+'/multi_cis_extracts/FLALGA'
save_path_graphs = path_output_data+'/graphs'
save_path_exps = path_output_data+'/exps'
save_path_figs = path_output_data+'/figures'

PER_CAP_ECONSUMP = 11.519

base_graph_name = 'basestate_graph_flalga_S3l'
disr_graph_name = 'disrstate_graph_flalga_S3l'


# =============================================================================
# CI Data
# =============================================================================
"""
For sources & curation of CI data, see script 
case_FLALGA_michael_preprocessing.py
"""
# PEOPLE 
gdf_people = gpd.read_file(save_path_cis+'/people_flalga.shp')

# POWER PLANTS
gdf_pplants = gpd.read_file(save_path_cis+'/powerplants_flalga.shp') 

# HEALTH FACILITIES
gdf_health = gpd.read_file(save_path_cis+'/healthfacilities_flalga.shp')

# EDUC. FACILITIES
gdf_educ = gpd.read_file(save_path_cis+'/educationfacilities_flalga.shp') 

# TELECOM
gdf_telecom = gpd.read_file(save_path_cis+'/celltowers_flalga.shp') 

# WASTEWATER
gdf_wastewater = gpd.read_file(save_path_cis+'/wastewater_flalga.shp') 

# =============================================================================
# Networks Preprocessing
# =============================================================================
"""
For preprocessing of network data of power lines & roads, see script 
case_FLALGA_michael_preprocessing.py
"""

# POWER LINES
gdf_power_edges, gdf_power_nodes = PowerlinePreprocess().preprocess(gdf_edges=gdf_powerlines)
power_network = Network(gdf_power_edges, gdf_power_nodes)
power_network.edges = power_network.edges[['from_id', 'to_id', 'osm_id','geometry','distance', 'ci_type']]
power_network.nodes = power_network.nodes.drop('name', axis=1)
power_graph = Graph(power_network, directed=False)
power_graph.link_clusters()
power_network = Network().from_graphs([power_graph.graph.as_directed()])
power_network.nodes = power_network.nodes.drop('name', axis=1)
# power_graph = Graph(power_network, directed=True) 
# power_graph.graph.clusters().summary():   
# 'Clustering with 3896 elements and 1 clusters'

# ROAD
gdf_road_edges, gdf_road_nodes = RoadPreprocess().preprocess(
    gdf_edges=gdf_mainroads_osm)
road_network = Network(gdf_road_edges, gdf_road_nodes)
# # easy workaround for doubling edges
road_graph = Graph(road_network, directed=False)
road_graph.link_clusters()
road_network = Network().from_graphs([road_graph.graph.as_directed()])
road_network.nodes = road_network.nodes.drop('name', axis=1)
road_network = Network(gdf_road_edges, gdf_road_nodes)
# road_graph = Graph(road_network, directed=True) 
# road_graph.graph.clusters().summary(): 
# 'Clustering with 51920 elements and 1 clusters'

# PEOPLE
__, gdf_people_nodes = NetworkPreprocess('people').preprocess(
    gdf_nodes=gdf_people)
people_network = Network(nodes=gdf_people_nodes)

# POWER PLANTS
__, gdf_pp_nodes = NetworkPreprocess('power plant').preprocess(
    gdf_nodes=gdf_pplants)
pplant_network = Network(nodes=gpd.GeoDataFrame(gdf_pp_nodes))

# HEALTHCARE
__, gdf_health_nodes = NetworkPreprocess('health').preprocess(
    gdf_nodes=gdf_health)
health_network = Network(nodes=gdf_health_nodes)

# EDUC
__, gdf_educ_nodes = NetworkPreprocess('education').preprocess(
    gdf_nodes=gdf_educ)
educ_network = Network(nodes=gdf_educ_nodes)

# TELECOM
__, gdf_tele_nodes = NetworkPreprocess('celltower').preprocess(
    gdf_nodes=gdf_telecom)
tele_network = Network(nodes=gdf_tele_nodes)

# WASTEWATER
__, gdf_ww_nodes = NetworkPreprocess('wastewater').preprocess(
    gdf_nodes=gdf_wastewater)
wastewater_network = Network(nodes=gdf_ww_nodes)


# MULTINET
cis_network = Network.from_nws([pplant_network, power_network, wastewater_network,
                                      people_network, health_network, educ_network,
                                      road_network, tele_network])
cis_network.initialize_funcstates()

# =============================================================================
# Interdependencies
# =============================================================================

cis_graph = Graph(cis_network, directed=True)

# create "missing physical structures" - needed for real world flows
cis_graph.link_vertices_closest_k('power line', 'power plant', link_name='power line', bidir=True, k=1)
cis_graph.link_vertices_closest_k('road', 'people',  link_name='road', bidir=True, k=1)
cis_graph.link_vertices_closest_k('road', 'health',  link_name='road',  bidir=True, k=1)
cis_graph.link_vertices_closest_k('road', 'education', link_name='road',  bidir=True, k=1)

# TODO: think about implementing data class for dps
df_dependencies = pd.read_csv(path_deps, sep=',', header=0)

for __, row in df_dependencies.iterrows():
    cis_graph.place_dependency(row.source, row.target, 
                               single_link=row.single_link,
                               access_cnstr=row.access_cnstr, 
                               dist_thresh=row.thresh_dist)
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

# =============================================================================
# Initial Demand / Supply / Flow check
# =============================================================================

cis_network.initialize_funcstates()

for __, row in df_dependencies.iterrows():
    cis_network.initialize_capacity(row.source, row.target)
cis_network.nodes = cis_network.nodes.drop('name', axis=1)
    
cis_graph = Graph(cis_network, directed=True)
cis_graph.cascade(df_dependencies, p_source='power plant', p_sink='power line',
                                        per_cap_cons=PER_CAP_ECONSUMP, source_var='NET_GEN')

cis_network = cis_graph.return_network()

# SAVE 
cis_network.nodes['geom_wkt'] = cis_network.nodes.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network.edges['geom_wkt'] = cis_network.edges.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network.nodes = cis_network.nodes.drop('name', axis=1)
cis_graph = Graph(cis_network, directed=True)
cis_graph.graph.write(save_path_graphs+'/'+base_graph_name+'.graphml', format='graphml')

# RELOAD IF NECESSARY
cis_graph = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/'+base_graph_name+'.graphml'))
cis_network = cis_graph.return_network()
cis_network.nodes['geometry'] = cis_network.nodes.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)
cis_network.edges['geometry'] = cis_network.edges.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)

# =============================================================================
# Service Access & stats
# =============================================================================

# TODO: write function to automatically output access stats for basic services

# Selectors


# =============================================================================
# Create Hazard
# =============================================================================
"""See preprocessing file for loading of TC"""
tc_michael = TropCyclone.from_hdf5(path_input_data+'/hazards/tc_michael_1h.hdf5')
tc_michael.check()
tc_michael.plot_intensity('2018280N18273')  

# =============================================================================
# Impact functions
# =============================================================================

def p_fail_pl(v_eval, v_crit=30, v_coll=60):
    """
    adapted from  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7801854
    and Vulnerability Assessment for Power Transmission Lines under Typhoon 
    Weather Based on a Cascading Failure State Transition Diagram
    """
    p_fail = []
    for v in v_eval:
        p = 0
        if (v > v_crit) & (v < v_coll):
            p = np.exp(0.6931*(v-v_crit)/v_crit)-1
        elif v > v_coll:
            p = 1
        p_fail.append(p)
        
    return p_fail

v_eval = np.linspace(0, 120, num=120)
p_fail_powerlines = p_fail_pl(v_eval, v_crit=40, v_coll=80)

# Step func
impf_prob = ImpactFunc() 
impf_prob.id = 1
impf_prob.tag = 'PL_Prob'
impf_prob.haz_type = 'TC'
impf_prob.name = 'power line failure prob'
impf_prob.intensity_unit = 'm/s'
impf_prob.intensity = np.array(v_eval)
impf_prob.mdd = np.array(p_fail_powerlines)
impf_prob.paa = np.sort(np.linspace(1, 1, num=120))
impf_prob.check()
impf_prob.plot()

# adapted from figure H.13 (residential 2-story building) loss function, Hazus TC 2.1 (p.940)
# medium terrain roughness parameter (z_theta = 0.35)
impf_educ = ImpactFunc() 
impf_educ.id = 2
impf_educ.tag = 'TC educ'
impf_educ.haz_type = 'TC'
impf_educ.name = 'Loss func. residental building z0 = 0.35'
impf_educ.intensity_unit = 'm/s'
impf_educ.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
impf_educ.mdd =       np.array([0, 0,   0,  0,   5,  20,  50,  80,  98, 100, 100, 100, 100]) / 100
impf_educ.paa = np.sort(np.linspace(1, 1, num=13))
impf_educ.check()
impf_educ.plot()

# adapted from figure N.1 (industrial 2 building) loss function, Hazus TC 2.1 (p.1115)
# medium terrain roughness parameter (z_theta = 0.35)
impf_indus = ImpactFunc() 
impf_indus.id = 3
impf_indus.haz_type = 'TC'
impf_indus.name = 'Loss func. industrial building z0 = 0.35'
impf_indus.intensity_unit = 'm/s'
impf_indus.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
impf_indus.mdd =       np.array([0, 0,   0,  0,   5,  15,  70,  98, 100, 100, 100, 100, 100]) / 100
impf_indus.paa = np.sort(np.linspace(1, 1, num=13))
impf_indus.check()
impf_indus.plot()

# adapted from Koks et al. 2019 (tree blowdown on road > 42 m/s)
impf_road = ImpactFunc() 
impf_road.id = 4
impf_road.haz_type = 'TC'
impf_road.name = 'Loss func. for roads from tree blowdown'
impf_road.intensity_unit = 'm/s'
impf_road.intensity = np.array([0, 30, 41, 44.5, 48, 120])
impf_road.mdd =       np.array([0, 0,   0, 50,100, 100]) / 100
impf_road.paa = np.sort(np.linspace(1, 1, num=6))
impf_road.check()
impf_road.plot()

# adapted from newspaper articles ("cell towers to withstand up to 110 mph")
impf_tele = ImpactFunc() 
impf_tele.id = 5
impf_tele.haz_type = 'TC'
impf_tele.name = 'Loss func. cell tower'
impf_tele.intensity_unit = 'm/s'
impf_tele.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
impf_tele.mdd =       np.array([0, 0,   0,  0,   0,  100,  100,  100, 100, 100, 100, 100, 100]) / 100
impf_tele.paa = np.sort(np.linspace(1, 1, num=13))
impf_tele.check()
impf_tele.plot()

# Mapping of wind field >= hurricane scale 1 (33 m/s)
impf_ppl = ImpactFunc() 
impf_ppl.id = 6
impf_ppl.haz_type = 'TC'
impf_ppl.name = 'People - Windfield Mapping >= TC'
impf_ppl.intensity_unit = 'm/s'
impf_ppl.intensity = np.array([0, 32, 33, 80, 100, 120, 140, 160]) 
impf_ppl.mdd = np.array([0, 0,   100,  100,   100,  100,  100,  100]) / 100
impf_ppl.paa = np.sort(np.linspace(1, 1, num=8))
impf_ppl.check()
impf_ppl.plot()

# Mapping of wind field >= Tropical Storm (17.4 m/s)
impf_ppl2 = ImpactFunc() 
impf_ppl2.id = 7
impf_ppl2.haz_type = 'TC'
impf_ppl2.name = 'People - Windfield Mapping >= TS'
impf_ppl2.intensity_unit = 'm/s'
impf_ppl2.intensity = np.array([0, 17, 17.4, 80, 100, 120, 140, 160]) 
impf_ppl2.mdd = np.array([0, 0,   100,  100,   100,  100,  100,  100]) / 100
impf_ppl2.paa = np.sort(np.linspace(1, 1, num=8))
impf_ppl2.check()
impf_ppl2.plot()


impfuncSet = ImpactFuncSet()
#impfuncSet.append(impfunc)
impfuncSet.append(impf_prob)
impfuncSet.append(impf_educ)
impfuncSet.append(impf_indus)
impfuncSet.append(impf_road)
impfuncSet.append(impf_tele)
impfuncSet.append(impf_ppl2)

# plot all in one
fig, ax = plt.subplots()
plt.plot(np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260])/ 2.237,  
         np.array([0, 0,   0,  0,   5,  20,  50,  80,  98, 100, 100, 100, 100]) / 100, 
         np.array(v_eval),
         np.array(p_fail_powerlines),
         np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 ,
         np.array([0, 0,   0,  0,   5,  15,  70,  98, 100, 100, 100, 100, 100]) / 100,
         np.array([0, 30, 41, 44.5, 48, 120]),
         np.array([0, 0,   0, 50,100, 100]) / 100,
         np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237, #np.linspace(0, 120, num=13)
         np.array([0, 0,   0,  0,   0,  100,  100,  100, 100, 100, 100, 100, 100]) / 100,
         np.array([0, 17, 17.4, 80, 100, 120, 140, 160]),
         np.array([0, 0,   100,  100,   100,  100,  100,  100]) / 100
         )
plt.title('Vulnerability curves to wind damage for CI components')
plt.legend(labels=['residential building', 'power line', 'industrial building',
                   'road', 'cell tower', 'windfield mapping'])
 
# =============================================================================
# Exposure (CIs to points)                     
# =============================================================================

"""
see case_FLALGA_michael_preprocessing.py for conversion of lines exposures
 to points
"""
# POWER LINES
# 500m p point, TC func 1
entity_pl = Entity()
exp_pl = Exposures()
exp_pl.gdf = gpd.read_file(save_path_exps+'/exp_pl_gdf.shp')
entity_pl.exposures.gdf = exp_pl.gdf
entity_pl.impact_funcs = impfuncSet
entity_pl.check()

# ROAD
# 500m p point, TC func 4
entity_road = Entity()
exp_road = Exposures()
exp_road.set_from_lines(
    gpd.GeoDataFrame(cis_network.edges[cis_network.edges.ci_type=='road'], 
                      crs='EPSG:4326'), m_per_point=500, disagg_values='cnst',
                      m_value=1)
exp_road.gdf['impf_TC'] = 4
entity_road.exposures.gdf = exp_road.gdf
entity_road.impact_funcs = impfuncSet
entity_road.check()


# HEALTHCARE
entity_hc = Entity()
exp_hc = Exposures()
exp_hc.gdf = cis_network.nodes[cis_network.nodes.ci_type=='health']
exp_hc.gdf['value'] = 1
exp_hc.gdf['impf_TC'] = 3
entity_hc.exposures.gdf = exp_hc.gdf
entity_hc.exposures.set_lat_lon()
entity_hc.impact_funcs = impfuncSet
entity_hc.check()


# EDUCATION
entity_educ = Entity()
exp_educ = Exposures()
exp_educ.gdf = cis_network.nodes[cis_network.nodes.ci_type=='education']
exp_educ.gdf['value'] = 1
exp_educ.gdf['impf_TC'] = 2
entity_educ.exposures.gdf = exp_educ.gdf
entity_educ.exposures.set_lat_lon()
entity_educ.impact_funcs = impfuncSet
entity_educ.check()

# TELECOM
entity_tele = Entity()
exp_tele = Exposures()
exp_tele.gdf = cis_network.nodes[cis_network.nodes.ci_type=='celltower']
exp_tele.gdf['value'] = 1
exp_tele.gdf['impf_TC'] = 5
entity_tele.exposures.gdf = exp_tele.gdf
entity_tele.exposures.set_lat_lon()
entity_tele.impact_funcs = impfuncSet
entity_tele.check()

# WASTEWATER
entity_wwater = Entity()
exp_wwater = Exposures()
exp_wwater.gdf = cis_network.nodes[cis_network.nodes.ci_type=='wastewater']
exp_wwater.gdf['value'] = 1
exp_wwater.gdf['impf_TC'] = 3
entity_wwater.exposures.gdf = exp_wwater.gdf
entity_wwater.exposures.set_lat_lon()
entity_wwater.impact_funcs = impfuncSet
entity_wwater.check()

# PEOPLE (for reference / validation purposes)
entity_ppl = Entity()
exp_ppl = Exposures()
exp_ppl.gdf = gdf_people.rename({'counts':'value'}, axis=1)
exp_ppl.gdf['impf_TC'] = 7
entity_ppl.exposures.gdf = exp_ppl.gdf
entity_ppl.exposures.set_lat_lon()
entity_ppl.impact_funcs = impfuncSet
entity_ppl.check()

# =============================================================================
# Impact (Point exposure & lines), Functional levels           
# =============================================================================

cis_network.edges['imp_dir'] = 0
cis_network.nodes['imp_dir'] = 0

# DIRECT IMPACT 
def binary_impact_from_prob(impact, seed=47):
    np.random.seed = seed
    rand = np.random.random(impact.eai_exp.size)
    return np.array([1 if p_fail > rnd else 0 for p_fail, rnd in 
                     zip(impact.eai_exp, rand)])

# POWERLINES
impact_pl_michael = Impact()
impact_pl_michael.calc(entity_pl.exposures, entity_pl.impact_funcs, tc_michael)
impact_pl_michael.plot_scatter_eai_exposure()
bin_imp = binary_impact_from_prob(impact_pl_michael)
entity_pl.exposures.gdf['imp_dir'] = bin_imp*entity_pl.exposures.gdf.value
impact_pl_michael_agg = entity_pl.exposures.gdf.groupby('edge ID').imp_dir.sum()
impact_pl_michael_rel = impact_pl_michael_agg/(entity_pl.exposures.gdf.groupby('edge ID').count()['geometry']*entity_pl.exposures.gdf.groupby('edge ID').value.mean())


# ROAD
impact_road_michael = Impact()
impact_road_michael.calc(entity_road.exposures, entity_road.impact_funcs, tc_michael)
impact_road_michael.plot_scatter_eai_exposure()
entity_road.exposures.gdf['imp_dir'] = impact_road_michael.eai_exp
impact_road_michael_agg = entity_road.exposures.gdf.groupby('edge ID').imp_dir.sum()
impact_road_michael_rel = impact_road_michael_agg/(entity_road.exposures.gdf.groupby('edge ID').count()['geometry']*entity_road.exposures.gdf.groupby('edge ID').value.mean())
impact_road_michael_rel = [min(imp, 1) for imp in impact_road_michael_rel]

# HEALTHCARE
impact_hc_michael = Impact()
impact_hc_michael.calc(entity_hc.exposures, entity_hc.impact_funcs, tc_michael)
impact_hc_michael.plot_scatter_eai_exposure()

# EDUCATION
impact_educ_michael = Impact()
impact_educ_michael.calc(entity_educ.exposures, entity_educ.impact_funcs, tc_michael)
impact_educ_michael.plot_scatter_eai_exposure()

# TELECOM
impact_tele_michael = Impact()
impact_tele_michael.calc(entity_tele.exposures, entity_tele.impact_funcs, tc_michael)
impact_tele_michael.plot_scatter_eai_exposure()

# WASTEWATER
impact_wwater_michael = Impact()
impact_wwater_michael.calc(entity_wwater.exposures, entity_wwater.impact_funcs, tc_michael)
impact_wwater_michael.plot_scatter_eai_exposure()

# PEOPLE
impact_ppl_michael = Impact()
impact_ppl_michael.calc(entity_ppl.exposures, entity_ppl.impact_funcs, tc_michael)
impact_ppl_michael.plot_scatter_eai_exposure()

  
# Assign Impacts to network
cis_network.edges.loc[cis_network.edges.ci_type=='power line', 'imp_dir'] = \
    impact_pl_michael_rel
cis_network.edges.loc[cis_network.edges.ci_type=='road', 'imp_dir'] = \
    impact_road_michael_rel
cis_network.nodes.imp_dir.loc[cis_network.nodes.ci_type=='health'] = \
    impact_hc_michael.eai_exp
cis_network.nodes.imp_dir.loc[cis_network.nodes.ci_type=='wastewater'] = \
    impact_wwater_michael.eai_exp
cis_network.nodes.imp_dir.loc[cis_network.nodes.ci_type=='celltower'] = \
    impact_tele_michael.eai_exp
cis_network.nodes.imp_dir.loc[cis_network.nodes.ci_type=='education'] = \
    impact_educ_michael.eai_exp
    

# IMPACT - FUNCTIONALITY LEVEL (INTERNAL)
THRESH_PLINE = 0.3
THRESH_ROAD = 0.3
THRESH_HEALTH = 0.3
THRESH_EDUC = 0.3
THRESH_WW = 0.3
THRESH_CT = 0.3

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
cis_network.nodes = cis_network.nodes.drop('name', axis=1)
cis_graph = Graph(cis_network, directed=True)
cis_graph.cascade(df_dependencies, p_source='power plant', p_sink='power line',
                                        per_cap_cons=PER_CAP_ECONSUMP, source_var='NET_GEN')
cis_network = cis_graph.return_network()

# SAVE 
cis_network.nodes['geom_wkt'] = cis_network.nodes.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network.edges['geom_wkt'] = cis_network.edges.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network.nodes = cis_network.nodes.drop('name', axis=1)
cis_graph = Graph(cis_network, directed=True)
cis_graph.graph.write(save_path_graphs+'/'+disr_graph_name+'.graphml', format='graphml')

# =============================================================================
# Basic service Access stats - disrupted state       
# =============================================================================

"""
S3h
{'power': 2969951.0453549027,
 'healthcare': 1625257.6368973362,
 'education': 6680729.906341586,
 'telecom': 1370294.8594914302,
 'mobility': 807896.64505082,
 'water': 3492704.3484603222}
"""

"""
S3l
{'power': 2969951.0453549027,
 'healthcare': 1625257.6368973362,
 'education': 6680729.906341586,
 'telecom': 1370294.8594914302,
 'mobility': 807896.64505082,
 'water': 3492704.3484603222}
"""
