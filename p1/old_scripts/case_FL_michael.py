#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:45:13 2021
@author: evelynm

Main workflow CI failure & basic service disruptions from TC Michael '18
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

# =============================================================================
# CI Data
# =============================================================================
"""
For sources & curation of CI data, see script 
case_FLALGA_michael_preprocessing.py
"""
# PEOPLE 
gdf_people = gpd.read_file(save_path_cis+'/people_flalga.shp')

# POWER LINES
gdf_powerlines = gpd.read_file(save_path_cis+'/powerlines_flalga.shp')
gdf_powerlines = gdf_powerlines[['TYPE','geometry','VOLT_CLASS', 'VOLTAGE']]
gdf_powerlines['osm_id'] = 'n/a'

# POWER PLANTS
gdf_pplants = gpd.read_file(save_path_cis+'/powerplants_flalga.shp') 

# HEALTH FACILITIES
gdf_health = gpd.read_file(save_path_cis+'/healthfacilities_flalga.shp')

# EDUC. FACILITIES
gdf_educ = gpd.read_file(save_path_cis+'/educationfacilities_flalga.shp') 

# TELECOM
gdf_telecom = gpd.read_file(save_path_cis+'/celltowers_flalga.shp') 

# ROADS
gdf_mainroads_osm = gpd.read_file(save_path_cis+'/mainroads_osm_flalga.shp') 

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
# TODO: loading edges doesnt work atm because of shapely issue. re.load from preprocessing file!
gdf_power_edges = gpd.read_file(save_path_cis+'/powerlines_flalga_processede.shp')
gdf_power_nodes = gpd.read_file(save_path_cis+'/powerlines_flalga_processedn.shp')
power_network = Network(gdf_power_edges, gdf_power_nodes)
# power_graph = Graph(power_network, directed=True) 
# power_graph.graph.clusters().summary(): 
# 'Clustering with 3896 elements and 1 clusters'

# ROAD
gdf_road_edges = gpd.read_file(save_path_cis+'/mainroads_osm_flalga_processede.shp')
gdf_road_nodes = gpd.read_file(save_path_cis+'/mainroads_osm_flalga_processedn.shp')
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
cis_graph.graph.write(save_path_graphs+'/deps_graph_flalga.graphml', format='graphml')

# RE-LOAD IF NECESSARY
cis_graph = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/deps_graph_flalga.graphml'))
cis_network = cis_graph.return_network()
cis_network.nodes['geometry'] = cis_network.nodes.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)
cis_network.edges['geometry'] = cis_network.edges.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)

# plots of connected CIS - exemplary
#  cis_network.plot_cis(ci_types=['power line', 'power plant', 'dependency_power line_people'], 
#                     title='Power & ppl Fl, Al, Ga')

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
cis_graph.graph.write(save_path_graphs+'/basestate_graph_flalga.graphml', format='graphml')

# RELOAD IF NECESSARY
cis_graph = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/basestate_graph_flalga.graphml'))
cis_network = cis_graph.return_network()
cis_network.nodes['geometry'] = cis_network.nodes.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)
cis_network.edges['geometry'] = cis_network.edges.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)

# SAVE gdf if necessary
cis_network.nodes.to_file(save_path_cis+'/cis_network_nodes_predisaster.shp')
cis_network.edges.to_file(save_path_cis+'/cis_network_edges_predisaster.shp')

# =============================================================================
# Service Access & stats
# =============================================================================

# TODO: write function to automatically output access stats for basic services

# Selectors
bool_people = cis_network.nodes.ci_type=='people'
bool_health = cis_network.nodes.ci_type=='health'
bool_education = cis_network.nodes.ci_type=='education'
bool_celltower = cis_network.nodes.ci_type=='celltower'
bool_health = cis_network.nodes.ci_type=='health'
bool_wastewater = cis_network.nodes.ci_type=='wastewater'


bool_powersupply = cis_network.nodes['actual_supply_power line_people']>0
bool_healthsupply = cis_network.nodes.actual_supply_health_people>0
bool_educsupply = cis_network.nodes.actual_supply_education_people>0
bool_telesupply = cis_network.nodes.actual_supply_celltower_people>0
bool_roadsupply = cis_network.nodes.actual_supply_road_people>0
bool_watersupply = cis_network.nodes.actual_supply_wastewater_people>0


# Stats
ppl_served_power = cis_network.nodes[bool_powersupply].counts.sum() # 37868718, 100%
ppl_nonserved_power = cis_network.nodes[~bool_powersupply].counts.sum() # 0

ppl_served_health = cis_network.nodes[bool_healthsupply].counts.sum() # 37'865'223
ppl_nonserved_health = cis_network.nodes[~bool_healthsupply].counts.sum() # 3'495

ppl_served_education = cis_network.nodes[bool_educsupply].counts.sum() # 37'787'571
ppl_nonserved_education = cis_network.nodes[~bool_educsupply].counts.sum() # 81'146

ppl_served_telecom = cis_network.nodes[bool_telesupply].counts.sum() # 37868463
ppl_nonserved_telecom = cis_network.nodes[~bool_telesupply].counts.sum() # 254

ppl_served_road = cis_network.nodes[bool_roadsupply].counts.sum() # 37868718
ppl_nonserved_road = cis_network.nodes[~bool_roadsupply].counts.sum() # 0

ppl_served_water = cis_network.nodes[bool_watersupply].counts.sum() # 37868718
ppl_nonserved_water = cis_network.nodes[~bool_watersupply].counts.sum() # 0

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
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
     
fig, ax = plt.subplots(1,1, figsize=(15,15))
for i in range(1,6):
    ax.plot(impfuncSet.get_func(fun_id=i)[0].intensity,
            impfuncSet.get_func(fun_id=i)[0].mdd,
            label=f'{impfuncSet.get_func(fun_id=i)[0].name}', color=f'{0.8-i*0.05}', linewidth=3,
            linestyle=linestyle_tuple[i][1])
    ax.legend(frameon=False, fontsize=22)
    ax.set_xlabel('Wind intensity (m/s)',fontsize=20)
    ax.set_ylabel('Structural Damage Fraction',fontsize=20)
 
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
exp_road.gdf = gpd.read_file(save_path_exps+'/exp_road_gdf.shp')
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
cis_network.edges.imp_dir[cis_network.edges.ci_type=='road'] = \
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
cis_graph.graph.write(save_path_graphs+'/disrstate_graph_flalga.graphml', format='graphml')

# RELOAD IF NECESSARY
cis_graph = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_flalga.graphml'))
cis_network = cis_graph.return_network()
cis_network.nodes['geometry'] = cis_network.nodes.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)
cis_network.edges['geometry'] = cis_network.edges.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)



# for saving df --> rename
rename_dict = {'actual_supply_power line_people':'sup_pl_ppl',
               'actual_supply_health_people': 'sup_hc_ppl',
               'actual_supply_education_people' : 'sup_ed_ppl',
               'actual_supply_celltower_people' : 'sup_ct_ppl',
               'actual_supply_wastewater_people' : 'sup_ww_ppl',
               'actual_supply_road_people' : 'sup_rd_ppl'
    }

cis_network.nodes = cis_network.nodes.rename(rename_dict, axis=1).drop([
    'capacity_power line_celltower',
       'capacity_celltower_education', 'capacity_power line_education',
       'capacity_wastewater_education', 'capacity_celltower_health',
       'capacity_power line_health', 'capacity_wastewater_health',
       'capacity_celltower_power plant', 'capacity_celltower_wastewater',
       'capacity_power line_wastewater', 'capacity_celltower_people',
       'capacity_education_people', 'capacity_health_people',
       'capacity_power line_people', 'capacity_road_people',
       'capacity_wastewater_people', 'NET_GEN'], axis=1)

# SAVE gdf if necessary
cis_network.nodes.to_file(save_path_cis+'/cis_network_nodes_postdisaster.shp')
cis_network.edges.to_file(save_path_cis+'/cis_network_edges_postdisaster.shp')


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
ppl_nonserved_power2 = cis_network.nodes[~bool_powersupply].counts.sum() # 1961066

ppl_served_health2 = cis_network.nodes[bool_healthsupply].counts.sum() # 
ppl_nonserved_health2 = cis_network.nodes[~bool_healthsupply].counts.sum() # 350629

ppl_served_education2 = cis_network.nodes[bool_educsupply].counts.sum() # 
ppl_nonserved_education2 = cis_network.nodes[~bool_educsupply].counts.sum() # 2060153

ppl_served_telecom2 = cis_network.nodes[bool_telesupply].counts.sum() # 
ppl_nonserved_telecom2 = cis_network.nodes[~bool_telesupply].counts.sum() # 812'226

ppl_served_water2 = cis_network.nodes[bool_watersupply].counts.sum() # 
ppl_nonserved_water2 = cis_network.nodes[~bool_watersupply].counts.sum() # 1218349

ppl_served_road2 = cis_network.nodes[bool_mobilitysupply].counts.sum() # 
ppl_nonserved_road2 = cis_network.nodes[~bool_mobilitysupply].counts.sum() # 749'065

# =============================================================================
# Validation     
# =============================================================================

# spatial overlap - CI impacts: celltowers
gdf_counties_flalga = gpd.read_file(path_input_data+'/population/Shape_AL/GU_CountyOrEquivalent.shp').append(
    gpd.read_file(path_input_data+'/population/Shape_FL/GU_CountyOrEquivalent.shp')).append(
    gpd.read_file(path_input_data+'/population/Shape_GA/GU_CountyOrEquivalent.shp'))
gdf_counties_flalga['unique_ID'] = np.arange(len(gdf_counties_flalga))      

gdf_ct = cis_network.nodes[cis_network.nodes.ci_type=='celltower'][['func_tot', 'func_internal', 'geometry']]
gdf_ct = gdf_ct.sjoin(gdf_counties_flalga, how="inner", predicate='intersects')
gdf_ct = gdf_ct.groupby(['unique_ID']).apply(lambda row: row.func_tot.mean()).reset_index().rename({0:'func_frac'}, axis=1)
gdf_ct['pout_model'] = (gdf_ct.func_frac-1)*-1

gdf_ctfail = gdf_ct.merge(gdf_counties_flalga[['geometry', 'County_Nam', 'unique_ID', 'State_Name']], on='unique_ID')
gdf_ctfail = gpd.GeoDataFrame(gdf_ctfail, crs='EPSG:4326')
gdf_ctfail['County_Nam'] = gdf_ctfail.apply(lambda row: row.County_Nam.lower(), axis=1)
gdf_ctfail['State_Name'] = gdf_ctfail['State_Name'].map({'Alabama':'AL', 'Georgia':'GA', 'Florida':'FL'})

df_outages = pd.read_csv('/Users/evelynm/Documents/WCR/3_PhD/0_Ideas, Overviews, Reviews/Cases/TC_Michael/tele_outage.csv')
df_outages['County_Nam']= df_outages.apply(lambda row: row.County_Nam.lower(), axis=1)
df_outages = df_outages.rename({'percent_out':'pout_rep'},axis=1)

gdf_ctfail = gpd.GeoDataFrame(pd.merge(gdf_ctfail, df_outages, on=['County_Nam', 'State_Name'], how='outer'))
gdf_ctfail.loc[np.isnan(gdf_ctfail.pout_rep), 'pout_rep'] = 0
gdf_ctfail['model_agr'] = 1-np.abs(gdf_ctfail.pout_rep - gdf_ctfail.pout_model)

gdf_ctfail.plot('model_agr', legend=True, vmin=0, vmax=1)
gdf_ctfail.plot('pout_model', legend=True, vmin=0, vmax=1)
gdf_ctfail.plot('pout_rep', legend=True, vmin=0, vmax=1)

gdf_ctfail.to_file(save_path_exps+'/ct_failures_valid.shp')

# spatial overlap - CI impacts: power outages
gdf_ppl = cis_network.nodes[cis_network.nodes.ci_type=='people'][['actual_supply_power line_people', 'geometry', 'counts']]
gdf_ppl['nonserved'] = (1-gdf_ppl['actual_supply_power line_people'])*gdf_ppl.counts
gdf_states_flalga = gpd.read_file('/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data/population/Shape_AL/GU_StateOrTerritory.shp')
gdf_states_flalga = gdf_states_flalga.iloc[2:]
gdf_pfail = gdf_ppl.sjoin(gdf_states_flalga, how="inner", predicate='intersects')
gdf_pfail = (gdf_pfail.groupby('State_Name').nonserved.sum()/gdf_pfail.groupby('State_Name').counts.sum()).reset_index().rename({0:'pout_model'}, axis=1)

gdf_pfail = gpd.GeoDataFrame(pd.merge(gdf_pfail, gdf_states_flalga[['State_Name', 'geometry']], on=['State_Name'], how='outer'))
gdf_pfail['pout_rep'] = [0.0309, 0.0368, 0.0644]
gdf_pfail.to_file(save_path_exps+'/pow_failures_valid.shp')


# spatial overlap - CI impacts: hospitals
hc_dict = {'emerald' : [30.183813, -85.658416],
           'encompass health' : [30.182210, -85.669732],
           'george' : [29.725137, -84.993350],
           'sacred hearts' : [30.266596, -85.973973],
           'calhoun liberty' : [30.459398, -85.051140],
           'gulf coast' : [30.187691, -85.664917],
           'jackson': [30.787270, -85.242467],
           'sacread heart gulf' : [29.779481, -85.288641]
           }
for key, value in hc_dict.items():
    hc_dict.update({key:[value[1], value[0]]})
    
hc_facilities = gpd.GeoSeries(shapely.geometry.Point(value) for value in hc_dict.values())
hc_facilities.to_file(save_path_exps+'/healthfacilities_dmg.shp')

hc_access_dict = {'insulin springfield' : [30.169017, -85.608711],
                  'ind deaths gadsden' : [30.597218, -84.627448],
                  'ind deaths liberty': [30.199844, -84.871796],
                  'ind deaths bay' : [30.338638, -85.626592]}

for key, value in hc_access_dict.items():
    hc_access_dict.update({key:[value[1], value[0]]})
    
hc_incidents = gpd.GeoSeries(shapely.geometry.Point(value) for value in hc_access_dict.values())
hc_incidents.to_file(save_path_exps+'/health_incidents.shp')

gdf_ppl_hc = cis_network.nodes[cis_network.nodes.ci_type=='people'][['actual_supply_health_people', 'geometry', 'counts']]
gpd.GeoSeries(shapely.geometry.MultiPoint(gdf_ppl_hc[gdf_ppl_hc.actual_supply_health_people==0].geometry.values).buffer(0.1)).to_file(save_path_exps+'/healthcare_loss_model.shp')

# mobility
gdf_ppl_rd = cis_network.nodes[cis_network.nodes.ci_type=='people'][['actual_supply_road_people', 'geometry', 'counts']]
gpd.GeoSeries(shapely.geometry.MultiPoint(gdf_ppl_hc[gdf_ppl_rd.actual_supply_road_people==0].geometry.values).buffer(0.1)).to_file(save_path_exps+'/mobility_loss_model.shp')
rd_access_dict = { 'liberty' : [-84.879446, 30.295582],
                  'calhoun' : [-85.182083, 30.402748]
                  }
rd_dmg_dict = {'walkulla' : [-84.291311, 30.156605],
                'eastpoint' : [ -84.879035, 29.752147]}

rd_incidents = gpd.GeoSeries(shapely.geometry.Point(value) for value in rd_access_dict.values())
rd_incidents.to_file(save_path_exps+'/road_incidents.shp')

rd_dgm = gpd.GeoSeries(shapely.geometry.Point(value) for value in rd_dmg_dict.values())
rd_dgm.to_file(save_path_exps+'/road_dmg.shp')


# Water
gdf_ppl_ww = cis_network.nodes[cis_network.nodes.ci_type=='people'][['actual_supply_wastewater_people', 'geometry', 'counts']]
gpd.GeoSeries(shapely.geometry.MultiPoint(gdf_ppl_ww[gdf_ppl_ww.actual_supply_wastewater_people==0].geometry.values).buffer(0.1)).to_file(save_path_exps+'/water_loss_model.shp')

water_dict = { 'Panama City' : [-85.689582, 30.183509],
                  'BAY COUNTY' : [ -85.708326, 30.308780],
                  'Mexico Beach' : [-85.412208, 29.947434]}
water_incidents = gpd.GeoSeries(shapely.geometry.Point(value) for value in water_dict.values())
water_incidents.to_file(save_path_exps+'/water_incidents.shp')

# School

educ_dict = { 'liberty' : [-84.879446, 30.295582],
                  'BAY COUNTY' : [ -85.708326, 30.308780],
                  'calhoun' : [-85.182083, 30.402748],
                  'jackson': [-85.242467, 30.787270],
                  'gadsden' : [-84.640505, 30.573252],
                  'gulf' : [ -85.231351, 29.948606],
                  'franklin' : [-84.803209, 29.911632],
                  'washington' : [-85.667901, 30.600797]}

gdf_ppl_educ = cis_network.nodes[cis_network.nodes.ci_type=='people'][['actual_supply_education_people', 'geometry', 'counts']]
gpd.GeoSeries(shapely.geometry.MultiPoint(
    gdf_ppl_educ[(gdf_ppl_educ.actual_supply_education_people==0)&(gdf_ppl_educ_bs.actual_supply_education_people==1)
    ].geometry.values).buffer(0.1)).to_file(save_path_exps+'/educ_loss_model.shp')

educ_incidents = gpd.GeoSeries(shapely.geometry.Point(value) for value in educ_dict.values())
educ_incidents.to_file(save_path_exps+'/educ_incidents.shp')

cis_graph = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/basestate_graph_flalga.graphml'))
cis_network_bs = cis_graph.return_network()
gdf_ppl_educ_bs = cis_network_bs.nodes[cis_network_bs.nodes.ci_type=='people'][['actual_supply_education_people', 'counts']]

gdf_ppl_educ[(gdf_ppl_educ.actual_supply_education_people==0)&(gdf_ppl_educ_bs.actual_supply_education_people==1)].counts.sum()