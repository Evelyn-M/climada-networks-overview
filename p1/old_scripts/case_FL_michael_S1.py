#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:45:13 2021
@author: evelynm

Sensitivity Analysis:
    Change Vulnerability funcs +/-
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

# RELOAD IF NECESSARY
cis_graph = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/basestate_graph_flalga.graphml'))
cis_network = cis_graph.return_network()
cis_network.nodes['geometry'] = cis_network.nodes.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)
cis_network.edges['geometry'] = cis_network.edges.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)

# =============================================================================
# Service Access & stats - Predisaster
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
p_fail_powerlines_low = p_fail_pl(v_eval, v_crit=30, v_coll=60)
p_fail_powerlines_high = p_fail_pl(v_eval, v_crit=50, v_coll=100)
p_fail_powerlines = p_fail_pl(v_eval, v_crit=40, v_coll=80)

# Step func
impf_prob_low = ImpactFunc() 
impf_prob_low.id = 1
impf_prob_low.tag = 'PL_Prob'
impf_prob_low.haz_type = 'TC'
impf_prob_low.name = 'power line failure prob'
impf_prob_low.intensity_unit = 'm/s'
impf_prob_low.intensity = np.array(v_eval)
impf_prob_low.mdd = np.array(p_fail_powerlines_low)
impf_prob_low.paa = np.sort(np.linspace(1, 1, num=120))
impf_prob_low.check()
impf_prob_low.plot()

# Step func
impf_prob_high = ImpactFunc() 
impf_prob_high.id = 1
impf_prob_high.tag = 'PL_Prob'
impf_prob_high.haz_type = 'TC'
impf_prob_high.name = 'power line failure prob'
impf_prob_high.intensity_unit = 'm/s'
impf_prob_high.intensity = np.array(v_eval)
impf_prob_high.mdd = np.array(p_fail_powerlines_high)
impf_prob_high.paa = np.sort(np.linspace(1, 1, num=120))
impf_prob_high.check()
impf_prob_high.plot()

# adapted from figure H.13 (residential 2-story building) loss function, Hazus TC 2.1 (p.940)
# medium terrain roughness parameter (z_theta = 0.35)
impf_educ_low = ImpactFunc() 
impf_educ_low.id = 2
impf_educ_low.tag = 'TC educ'
impf_educ_low.haz_type = 'TC'
impf_educ_low.name = 'Loss func. residental building z0 = 0.35'
impf_educ_low.intensity_unit = 'm/s'
impf_educ_low.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
impf_educ_low.mdd =       np.array([0, 0,   0, 5,  20,  50,  80,  98,  100, 100, 100, 100, 100]) / 100
impf_educ_low.paa = np.sort(np.linspace(1, 1, num=13))
impf_educ_low.check()
impf_educ_low.plot()

impf_educ_high = ImpactFunc() 
impf_educ_high.id = 2
impf_educ_high.tag = 'TC educ'
impf_educ_high.haz_type = 'TC'
impf_educ_high.name = 'Loss func. residental building z0 = 0.35'
impf_educ_high.intensity_unit = 'm/s'
impf_educ_high.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
impf_educ_high.mdd =       np.array([0, 0,   0,0,0, 5,  20,  50,  80,  98,  100, 100, 100]) / 100
impf_educ_high.paa = np.sort(np.linspace(1, 1, num=13))
impf_educ_high.check()
impf_educ_high.plot()

# adapted from figure N.1 (industrial 2 building) loss function, Hazus TC 2.1 (p.1115)
# medium terrain roughness parameter (z_theta = 0.35)
impf_indus_low = ImpactFunc() 
impf_indus_low.id = 3
impf_indus_low.haz_type = 'TC'
impf_indus_low.name = 'Loss func. industrial building z0 = 0.35'
impf_indus_low.intensity_unit = 'm/s'
impf_indus_low.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
impf_indus_low.mdd =       np.array([0, 0,  0,  5,  15,  70,  98, 100, 100, 100, 100, 100, 100]) / 100
impf_indus_low.paa = np.sort(np.linspace(1, 1, num=13))
impf_indus_low.check()
impf_indus_low.plot()

impf_indus_high = ImpactFunc() 
impf_indus_high.id = 3
impf_indus_high.haz_type = 'TC'
impf_indus_high.name = 'Loss func. industrial building z0 = 0.35'
impf_indus_high.intensity_unit = 'm/s'
impf_indus_high.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
impf_indus_high.mdd =       np.array([0, 0,  0, 0, 0,  5,  15,  70,  98, 100, 100, 100, 100]) / 100
impf_indus_high.paa = np.sort(np.linspace(1, 1, num=13))
impf_indus_high.check()
impf_indus_high.plot()

# adapted from Koks et al. 2019 (tree blowdown on road > 42 m/s)
impf_road_low = ImpactFunc() 
impf_road_low.id = 4
impf_road_low.haz_type = 'TC'
impf_road_low.name = 'Loss func. for roads from tree blowdown'
impf_road_low.intensity_unit = 'm/s'
impf_road_low.intensity = np.array([0, 30, 35, 40, 48, 120])
impf_road_low.mdd =       np.array([0, 0,  50, 100, 100, 100]) / 100
impf_road_low.paa = np.sort(np.linspace(1, 1, num=6))
impf_road_low.check()
impf_road_low.plot()

impf_road_high = ImpactFunc() 
impf_road_high.id = 4
impf_road_high.haz_type = 'TC'
impf_road_high.name = 'Loss func. for roads from tree blowdown'
impf_road_high.intensity_unit = 'm/s'
impf_road_high.intensity = np.array([0, 30, 41, 48.5, 52, 58, 120])
impf_road_high.mdd =       np.array([0, 0,   0, 0,  50, 100, 100]) / 100
impf_road_high.paa = np.sort(np.linspace(1, 1, num=7))
impf_road_high.check()
impf_road_high.plot()

# adapted from newspaper articles ("cell towers to withstand up to 110 mph")
impf_tele_low = ImpactFunc() 
impf_tele_low.id = 5
impf_tele_low.haz_type = 'TC'
impf_tele_low.name = 'Loss func. cell tower'
impf_tele_low.intensity_unit = 'm/s'
impf_tele_low.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
impf_tele_low.mdd =       np.array([0, 0,   0, 0, 100, 100,  100,  100,  100, 100, 100, 100, 100]) / 100
impf_tele_low.paa = np.sort(np.linspace(1, 1, num=13))
impf_tele_low.check()
impf_tele_low.plot()

impf_tele_high = ImpactFunc() 
impf_tele_high.id = 5
impf_tele_high.haz_type = 'TC'
impf_tele_high.name = 'Loss func. cell tower'
impf_tele_high.intensity_unit = 'm/s'
impf_tele_high.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
impf_tele_high.mdd =       np.array([0, 0,   0, 0, 0, 0, 100,  100,  100, 100, 100, 100, 100]) / 100
impf_tele_high.paa = np.sort(np.linspace(1, 1, num=13))
impf_tele_high.check()
impf_tele_high.plot()

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


impfuncSet_low = ImpactFuncSet()
#impfuncSet.append(impfunc)
impfuncSet_low.append(impf_prob_low)
impfuncSet_low.append(impf_educ_low)
impfuncSet_low.append(impf_indus_low)
impfuncSet_low.append(impf_road_low)
impfuncSet_low.append(impf_tele_low)
impfuncSet_low.append(impf_ppl2)

impfuncSet_high = ImpactFuncSet()
#impfuncSet.append(impfunc)
impfuncSet_high.append(impf_prob_high)
impfuncSet_high.append(impf_educ_high)
impfuncSet_high.append(impf_indus_high)
impfuncSet_high.append(impf_road_high)
impfuncSet_high.append(impf_tele_high)
impfuncSet_high.append(impf_ppl2)

# Step func
impf_prob = ImpactFunc() 
impf_prob.id = 1
impf_prob.tag = 'PL_Prob'
impf_prob.haz_type = 'TC'
impf_prob.name = 'Power lines'
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
impf_educ.name = 'Residental buildings' # z0 = 0.35
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
impf_indus.name = 'Industrial buildings ' #z0 = 0.35
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
impf_road.name = 'Roads (tree blowdown)'
impf_road.intensity_unit = 'm/s'
impf_road.intensity = np.array([0, 30, 41, 44.5, 48, 120])
impf_road.mdd =       np.array([0, 0,   0,   50, 100, 100]) / 100
impf_road.paa = np.sort(np.linspace(1, 1, num=6))
impf_road.check()
impf_road.plot()

# adapted from newspaper articles ("cell towers to withstand up to 110 mph")
impf_tele = ImpactFunc() 
impf_tele.id = 5
impf_tele.haz_type = 'TC'
impf_tele.name = 'Cell tower'
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
impfuncSet.append(impf_ppl)
impfuncSet.append(impf_ppl2)


# plot all in one
fig, axes = plt.subplots(3,2, figsize=(20,20))

for ax, i in zip(axes.flatten(), np.arange(1,len(axes.flatten())+1)):
    if i <=5:
        ax.plot(impfuncSet.get_func(fun_id=i)[0].intensity,
                impfuncSet.get_func(fun_id=i)[0].mdd,
                label='base scenario', color='0.8', linewidth=2)
        ax.plot(impfuncSet_high.get_func(fun_id=i)[0].intensity,
                impfuncSet_high.get_func(fun_id=i)[0].mdd,
                label='low vulnerability scenario', linestyle='dashed',  color='k')
        ax.plot(impfuncSet_low.get_func(fun_id=i)[0].intensity,
                impfuncSet_low.get_func(fun_id=i)[0].mdd,
                label='high vulnerability scenario', linestyle='dashdot',  color='k')
        ax.legend(frameon=False, fontsize=18)
        ax.set_xlabel('Wind intensity (m/s)',fontsize=16)
        ax.set_title(f'{impfuncSet.get_func(fun_id=i)[0].name}',fontsize=21)
        if i == 1:
            ax.set_ylabel('Failure Probability',fontsize=18)
        else:
            ax.set_ylabel('Structural Damage Fraction',fontsize=18)
        ax.set_title(f'{impfuncSet.get_func(fun_id=i)[0].name}',fontsize=21)
    else:
        ax.plot(impfuncSet.get_func(fun_id=i)[0].intensity,
                impfuncSet.get_func(fun_id=i)[0].mdd,
                label='Tropical Cyclone Threshold',  color='k')
        ax.plot(impfuncSet.get_func(fun_id=i+1)[0].intensity,
                impfuncSet.get_func(fun_id=i+1)[0].mdd, linestyle='dashed',
                label='Tropical Storm Threshold',  color='k')
        ax.legend(frameon=False, fontsize=16)
        ax.set_xlabel('Wind intensity (m/s)', fontsize=18)
        ax.set_ylabel('Fraction of affected population', fontsize=18)
        ax.set_title('Windfield Mapping onto Population', fontsize=21)

plt.tight_layout()

# =============================================================================
# Exposure (CIs to points)                     
# =============================================================================

"""
see case_FLALGA_michael_preprocessing.py for conversion of lines exposures
 to points
"""
# POWER LINES
# 500m p point, TC func 1
entity_pl_low = Entity()
exp_pl = Exposures()
exp_pl.gdf = gpd.read_file(save_path_exps+'/exp_pl_gdf.shp')
entity_pl_low.exposures.gdf = exp_pl.gdf
entity_pl_low.impact_funcs = impfuncSet_low
entity_pl_low.check()

entity_pl_high = Entity()
entity_pl_high.exposures.gdf = exp_pl.gdf
entity_pl_high.impact_funcs = impfuncSet_high
entity_pl_high.check()


# ROAD
# 500m p point, TC func 4
entity_road_high = Entity()
exp_road = Exposures()
exp_road.gdf = gpd.read_file(save_path_exps+'/exp_road_gdf.shp')
entity_road_high.exposures.gdf = exp_road.gdf
entity_road_high.impact_funcs = impfuncSet_high
entity_road_high.check()

entity_road_low = Entity()
entity_road_low.exposures.gdf = exp_road.gdf
entity_road_low.impact_funcs = impfuncSet_low
entity_road_low.check()

# HEALTHCARE
entity_hc_low = Entity()
exp_hc = Exposures()
exp_hc.gdf = cis_network.nodes[cis_network.nodes.ci_type=='health']
exp_hc.gdf['value'] = 1
exp_hc.gdf['impf_TC'] = 3
entity_hc_low.exposures.gdf = exp_hc.gdf
entity_hc_low.exposures.set_lat_lon()
entity_hc_low.impact_funcs = impfuncSet_low
entity_hc_low.check()

entity_hc_high = Entity()
entity_hc_high.exposures.gdf = exp_hc.gdf
entity_hc_high.exposures.set_lat_lon()
entity_hc_high.impact_funcs = impfuncSet_high
entity_hc_high.check()


# EDUCATION
entity_educ_low = Entity()
exp_educ = Exposures()
exp_educ.gdf = cis_network.nodes[cis_network.nodes.ci_type=='education']
exp_educ.gdf['value'] = 1
exp_educ.gdf['impf_TC'] = 2
entity_educ_low.exposures.gdf = exp_educ.gdf
entity_educ_low.exposures.set_lat_lon()
entity_educ_low.impact_funcs = impfuncSet_low
entity_educ_low.check()

entity_educ_high = Entity()
entity_educ_high.exposures.gdf = exp_educ.gdf
entity_educ_high.exposures.set_lat_lon()
entity_educ_high.impact_funcs = impfuncSet_high
entity_educ_high.check()

# TELECOM
entity_tele_high = Entity()
exp_tele = Exposures()
exp_tele.gdf = cis_network.nodes[cis_network.nodes.ci_type=='celltower']
exp_tele.gdf['value'] = 1
exp_tele.gdf['impf_TC'] = 5
entity_tele_high.exposures.gdf = exp_tele.gdf
entity_tele_high.exposures.set_lat_lon()
entity_tele_high.impact_funcs = impfuncSet_high
entity_tele_high.check()

entity_tele_low = Entity()
entity_tele_low.exposures.gdf = exp_tele.gdf
entity_tele_low.exposures.set_lat_lon()
entity_tele_low.impact_funcs = impfuncSet_low
entity_tele_low.check()

# WASTEWATER
entity_wwater_low = Entity()
exp_wwater = Exposures()
exp_wwater.gdf = cis_network.nodes[cis_network.nodes.ci_type=='wastewater']
exp_wwater.gdf['value'] = 1
exp_wwater.gdf['impf_TC'] = 3
entity_wwater_low.exposures.gdf = exp_wwater.gdf
entity_wwater_low.exposures.set_lat_lon()
entity_wwater_low.impact_funcs = impfuncSet_low
entity_wwater_low.check()

entity_wwater_high = Entity()
entity_wwater_high.exposures.gdf = exp_wwater.gdf
entity_wwater_high.exposures.set_lat_lon()
entity_wwater_high.impact_funcs = impfuncSet_high
entity_wwater_high.check()

# PEOPLE (for reference / validation purposes)
entity_ppl = Entity()
exp_ppl = Exposures()
exp_ppl.gdf = gdf_people.rename({'counts':'value'}, axis=1)
exp_ppl.gdf['impf_TC'] = 6
entity_ppl.exposures.gdf = exp_ppl.gdf
entity_ppl.exposures.set_lat_lon()
entity_ppl.impact_funcs = impfuncSet_losw
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
impact_pl_michael_low = Impact()
impact_pl_michael_low.calc(entity_pl_low.exposures, entity_pl_low.impact_funcs, tc_michael)
impact_pl_michael_low.plot_scatter_eai_exposure()
bin_imp = binary_impact_from_prob(impact_pl_michael_low)
entity_pl_low.exposures.gdf['imp_dir'] = bin_imp*entity_pl_low.exposures.gdf.value
impact_pl_michael_agg_low = entity_pl_low.exposures.gdf.groupby('edge ID').imp_dir.sum()
impact_pl_michael_rel_low = impact_pl_michael_agg_low/(entity_pl_low.exposures.gdf.groupby('edge ID').count()['geometry']*entity_pl_low.exposures.gdf.groupby('edge ID').value.mean())

impact_pl_michael_high = Impact()
impact_pl_michael_high.calc(entity_pl_high.exposures, entity_pl_high.impact_funcs, tc_michael)
impact_pl_michael_high.plot_scatter_eai_exposure()
bin_imp = binary_impact_from_prob(impact_pl_michael_high)
entity_pl_high.exposures.gdf['imp_dir'] = bin_imp*entity_pl_high.exposures.gdf.value
impact_pl_michael_agg_high = entity_pl_high.exposures.gdf.groupby('edge ID').imp_dir.sum()
impact_pl_michael_rel_high = impact_pl_michael_agg_high/(entity_pl_high.exposures.gdf.groupby('edge ID').count()['geometry']*entity_pl_high.exposures.gdf.groupby('edge ID').value.mean())



# ROAD
impact_road_michael_high = Impact()
impact_road_michael_high.calc(entity_road_high.exposures, entity_road_high.impact_funcs, tc_michael)
impact_road_michael_high.plot_scatter_eai_exposure()
entity_road_high.exposures.gdf['imp_dir'] = impact_road_michael_high.eai_exp
impact_road_michael_agg_high = entity_road_high.exposures.gdf.groupby('edge ID').imp_dir.sum()
impact_road_michael_rel_high = impact_road_michael_agg_high/(entity_road_high.exposures.gdf.groupby('edge ID').count()['geometry']*entity_road_high.exposures.gdf.groupby('edge ID').value.mean())

impact_road_michael_low = Impact()
impact_road_michael_low.calc(entity_road_low.exposures, entity_road_low.impact_funcs, tc_michael)
impact_road_michael_low.plot_scatter_eai_exposure()
entity_road_low.exposures.gdf['imp_dir'] = impact_road_michael_low.eai_exp
impact_road_michael_agg_low = entity_road_low.exposures.gdf.groupby('edge ID').imp_dir.sum()
impact_road_michael_rel_low = impact_road_michael_agg_low/(entity_road_low.exposures.gdf.groupby('edge ID').count()['geometry']*entity_road_low.exposures.gdf.groupby('edge ID').value.mean())

# HEALTHCARE
impact_hc_michael_low = Impact()
impact_hc_michael_low.calc(entity_hc_low.exposures, entity_hc_low.impact_funcs, tc_michael)
impact_hc_michael_low.plot_scatter_eai_exposure()

impact_hc_michael_high = Impact()
impact_hc_michael_high.calc(entity_hc_high.exposures, entity_hc_high.impact_funcs, tc_michael)
impact_hc_michael_high.plot_scatter_eai_exposure()

# EDUCATION
impact_educ_michael_low = Impact()
impact_educ_michael_low.calc(entity_educ_low.exposures, entity_educ_low.impact_funcs, tc_michael)
impact_educ_michael_low.plot_scatter_eai_exposure()

impact_educ_michael_high = Impact()
impact_educ_michael_high.calc(entity_educ_high.exposures, entity_educ_high.impact_funcs, tc_michael)
impact_educ_michael_high.plot_scatter_eai_exposure()

# TELECOM
impact_tele_michael_low = Impact()
impact_tele_michael_low.calc(entity_tele_low.exposures, entity_tele_low.impact_funcs, tc_michael)
impact_tele_michael_low.plot_scatter_eai_exposure()

impact_tele_michael_high = Impact()
impact_tele_michael_high.calc(entity_tele_high.exposures, entity_tele_high.impact_funcs, tc_michael)
impact_tele_michael_high.plot_scatter_eai_exposure()

# WASTEWATER
impact_wwater_michael_low = Impact()
impact_wwater_michael_low.calc(entity_wwater_low.exposures, entity_wwater_low.impact_funcs, tc_michael)
impact_wwater_michael_low.plot_scatter_eai_exposure()

impact_wwater_michael_high = Impact()
impact_wwater_michael_high.calc(entity_wwater_high.exposures, entity_wwater_high.impact_funcs, tc_michael)
impact_wwater_michael_high.plot_scatter_eai_exposure()

# PEOPLE
impact_ppl_michael = Impact()
impact_ppl_michael.calc(entity_ppl.exposures, entity_ppl.impact_funcs, tc_michael)
impact_ppl_michael.plot_scatter_eai_exposure()

import copy
cis_network_low = copy.deepcopy(cis_network)
cis_network_high = copy.deepcopy(cis_network)

# Assign Impacts to network
cis_network_low.edges.loc[cis_network_low.edges.ci_type=='power line', 'imp_dir'] = \
    impact_pl_michael_rel_low
cis_network_low.edges.imp_dir[cis_network_low.edges.ci_type=='road'] = \
    impact_road_michael_rel_low
cis_network_low.nodes.imp_dir.loc[cis_network_low.nodes.ci_type=='health'] = \
    impact_hc_michael_low.eai_exp
cis_network_low.nodes.imp_dir.loc[cis_network_low.nodes.ci_type=='wastewater'] = \
    impact_wwater_michael_low.eai_exp
cis_network_low.nodes.imp_dir.loc[cis_network_low.nodes.ci_type=='celltower'] = \
    impact_tele_michael_low.eai_exp
cis_network_low.nodes.imp_dir.loc[cis_network_low.nodes.ci_type=='education'] = \
    impact_educ_michael_low.eai_exp

cis_network_high.edges.loc[cis_network_high.edges.ci_type=='power line', 'imp_dir'] = \
    impact_pl_michael_rel_high
cis_network_high.edges.imp_dir[cis_network_high.edges.ci_type=='road'] = \
    impact_road_michael_rel_high
cis_network_high.nodes.imp_dir.loc[cis_network_high.nodes.ci_type=='health'] = \
    impact_hc_michael_high.eai_exp
cis_network_high.nodes.imp_dir.loc[cis_network_high.nodes.ci_type=='wastewater'] = \
    impact_wwater_michael_high.eai_exp
cis_network_high.nodes.imp_dir.loc[cis_network_high.nodes.ci_type=='celltower'] = \
    impact_tele_michael_high.eai_exp
cis_network_high.nodes.imp_dir.loc[cis_network_high.nodes.ci_type=='education'] = \
    impact_educ_michael_high.eai_exp    

# IMPACT - FUNCTIONALITY LEVEL (INTERNAL)
THRESH_PLINE = 0.3
THRESH_ROAD = 0.3
THRESH_HEALTH = 0.3
THRESH_EDUC = 0.3
THRESH_WW = 0.3
THRESH_CT = 0.3

cond_fail_pline_low = ((cis_network_low.edges.ci_type=='power line') & 
                   (cis_network_low.edges.imp_dir>=THRESH_PLINE))

cond_fail_road_low = ((cis_network_low.edges.ci_type=='road') & 
                   (cis_network_low.edges.imp_dir>=THRESH_ROAD))

cond_fail_health_low = ((cis_network_low.nodes.ci_type=='health') & 
                   (cis_network_low.nodes.imp_dir>THRESH_HEALTH))

cond_fail_educ_low = ((cis_network_low.nodes.ci_type=='educ') & 
                   (cis_network_low.nodes.imp_dir>THRESH_EDUC))

cond_fail_wwater_low = ((cis_network_low.nodes.ci_type=='wastewater') & 
                   (cis_network_low.nodes.imp_dir>THRESH_WW))

cond_fail_tele_low = ((cis_network_low.nodes.ci_type=='celltower') & 
                   (cis_network_low.nodes.imp_dir>THRESH_CT))

for fail_cond in [cond_fail_pline_low, cond_fail_road_low]:
    cis_network_low.edges.func_internal.loc[fail_cond] = 0

for fail_cond in [cond_fail_health_low, cond_fail_educ_low, cond_fail_wwater_low, 
                  cond_fail_tele_low]:
    cis_network_low.nodes.func_internal.loc[fail_cond] = 0

cond_fail_pline_high = ((cis_network_high.edges.ci_type=='power line') & 
                   (cis_network_high.edges.imp_dir>=THRESH_PLINE))

cond_fail_road_high = ((cis_network_high.edges.ci_type=='road') & 
                   (cis_network_high.edges.imp_dir>=THRESH_ROAD))

cond_fail_health_high = ((cis_network_high.nodes.ci_type=='health') & 
                   (cis_network_high.nodes.imp_dir>THRESH_HEALTH))

cond_fail_educ_high = ((cis_network_high.nodes.ci_type=='educ') & 
                   (cis_network_high.nodes.imp_dir>THRESH_EDUC))

cond_fail_wwater_high = ((cis_network_high.nodes.ci_type=='wastewater') & 
                   (cis_network_high.nodes.imp_dir>THRESH_WW))

cond_fail_tele_high = ((cis_network_high.nodes.ci_type=='celltower') & 
                   (cis_network_high.nodes.imp_dir>THRESH_CT))

for fail_cond in [cond_fail_pline_low, cond_fail_road_low]:
    cis_network_low.edges.func_internal.loc[fail_cond] = 0

for fail_cond in [cond_fail_health_low, cond_fail_educ_low, cond_fail_wwater_low, 
                  cond_fail_tele_low]:
    cis_network_low.nodes.func_internal.loc[fail_cond] = 0
    
for fail_cond in [cond_fail_pline_low, cond_fail_road_low]:
    cis_network_low.edges.func_internal.loc[fail_cond] = 0

for fail_cond in [cond_fail_health_high, cond_fail_educ_high, cond_fail_wwater_high, 
                  cond_fail_tele_high]:
    cis_network_high.nodes.func_internal.loc[fail_cond] = 0
# TOTAL FUNC-STATES
cis_network_low.edges['func_tot'] = [np.min([func_internal, func_tot]) 
                                 for func_internal, func_tot in 
                                 zip(cis_network_low.edges.func_internal, 
                                     cis_network_low.edges.func_tot)]

cis_network_low.nodes['func_tot'] = [np.min([func_internal, func_tot]) 
                                 for func_internal, func_tot in 
                                 zip(cis_network_low.nodes.func_internal, 
                                     cis_network_low.nodes.func_tot)]

cis_network_high.edges['func_tot'] = [np.min([func_internal, func_tot]) 
                                 for func_internal, func_tot in 
                                 zip(cis_network_high.edges.func_internal, 
                                     cis_network_high.edges.func_tot)]

cis_network_high.nodes['func_tot'] = [np.min([func_internal, func_tot]) 
                                 for func_internal, func_tot in 
                                 zip(cis_network_high.nodes.func_internal, 
                                     cis_network_high.nodes.func_tot)]

# =============================================================================
# Failure Cascades                  
# =============================================================================
df_dependencies = pd.read_csv(path_deps, sep=',', header=0)
def geoms_to_wkt(geom):
    if not geom:
        return 'GEOMETRYCOLLECTION EMPTY'
    elif not geom.is_empty:
        return geom.wkt
    else:
        return 'GEOMETRYCOLLECTION EMPTY'

cis_network_low.nodes = cis_network_low.nodes.drop('name', axis=1)
cis_graph_low = Graph(cis_network_low, directed=True)
cis_graph_low.cascade(df_dependencies, p_source='power plant', p_sink='power line',
                                        per_cap_cons=PER_CAP_ECONSUMP, source_var='NET_GEN')
cis_network_low = cis_graph_low.return_network()

# SAVE 
cis_network_low.nodes['geom_wkt'] = cis_network_low.nodes.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network_low.edges['geom_wkt'] = cis_network_low.edges.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network_low.nodes = cis_network_low.nodes.drop('name', axis=1)
cis_graph_low = Graph(cis_network_low, directed=True)
cis_graph_low.graph.write(save_path_graphs+'/disrstate_graph_flalga_S1l.graphml', format='graphml')

cis_network_high.nodes = cis_network_high.nodes.drop('name', axis=1)
cis_graph_high = Graph(cis_network_high, directed=True)
cis_graph_high.cascade(df_dependencies, p_source='power plant', p_sink='power line',
                                        per_cap_cons=PER_CAP_ECONSUMP, source_var='NET_GEN')
cis_network_high = cis_graph_high.return_network()

# SAVE 
cis_network_high.nodes['geom_wkt'] = cis_network_high.nodes.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network_high.edges['geom_wkt'] = cis_network_high.edges.apply(lambda row: geoms_to_wkt(row.geometry), axis=1)
cis_network_high.nodes = cis_network_high.nodes.drop('name', axis=1)
cis_graph_high = Graph(cis_network_high, directed=True)
cis_graph_high.graph.write(save_path_graphs+'/disrstate_graph_flalga_S1h.graphml', format='graphml')


# RELOAD IF NECESSARY
cis_graph = GraphCalcs(ig.Graph.Read_GraphML(save_path_graphs+'/disrstate_graph_flalga_S1l.graphml'))
cis_network = cis_graph.return_network()
cis_network.nodes['geometry'] = cis_network.nodes.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)
cis_network.edges['geometry'] = cis_network.edges.apply(lambda row: shapely.wkt.loads(row.geom_wkt), axis=1)

# =============================================================================
# Basic service Access stats - disrupted state       
# =============================================================================

# Selectors
bool_powersupply = cis_network_high.nodes['actual_supply_power line_people']>0
bool_healthsupply = cis_network_high.nodes.actual_supply_health_people>0
bool_educsupply = cis_network_high.nodes.actual_supply_education_people>0
bool_telesupply = cis_network_high.nodes.actual_supply_celltower_people>0
bool_watersupply = cis_network_high.nodes.actual_supply_wastewater_people>0
bool_mobilitysupply = cis_network_high.nodes.actual_supply_road_people>0

# Stats
ppl_served_power2 = cis_network_high.nodes[bool_powersupply].counts.sum() # 
ppl_nonserved_power2 = cis_network_high.nodes[~bool_powersupply].counts.sum() # 619670 / 

ppl_served_health2 = cis_network_high.nodes[bool_healthsupply].counts.sum() # 
ppl_nonserved_health2 = cis_network_high.nodes[~bool_healthsupply].counts.sum() # 161256 / 4236537

ppl_served_education2 = cis_network_high.nodes[bool_educsupply].counts.sum() # 
ppl_nonserved_education2 = cis_network_high.nodes[~bool_educsupply].counts.sum() # 1722529 / 10032216

ppl_served_telecom2 = cis_network_high.nodes[bool_telesupply].counts.sum() # 
ppl_nonserved_telecom2 = cis_network_high.nodes[~bool_telesupply].counts.sum() # 627968 / 3225607

ppl_served_water2 = cis_network_high.nodes[bool_watersupply].counts.sum() # 
ppl_nonserved_water2 = cis_network_high.nodes[~bool_watersupply].counts.sum() # 1133199 / 2877209

ppl_served_road2 = cis_network_high.nodes[bool_mobilitysupply].counts.sum() # 
ppl_nonserved_road2 = cis_network_high.nodes[~bool_mobilitysupply].counts.sum() # 0 / 9441591

