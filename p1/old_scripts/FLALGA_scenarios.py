#!/usr/bin/env python
# coding: utf-8

# # Critical Infrastructure Failures from Hurricane Michael (FL, AL, GA)
# ## Scenario Analysis

# The notebook is structured as follows:
# * 
# * 

# ### Data Loading and Pre-Processing of Networks

# In[1]:


# standard imports
import geopandas as gpd
import numpy as np
import sys
import pandas as pd
import pickle 


# In[2]:


# currently on climada_petals branch feature/networks until it will be merged into the develop & main branch of the repo
from climada_petals.engine.networks.nw_preps import NetworkPreprocess
from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import Graph
from climada_petals.engine.networks import nw_utils as nwu


# In[3]:


# imports from the core part of CLIMADA
from climada.hazard.base import Hazard
from climada.util import lines_polys_handler as u_lp
from climada.entity.entity_def import Entity
from climada.entity.exposures.base import Exposures
from climada.engine import Impact
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet


# In[5]:


# =============================================================================
# Constants & filepaths
# =============================================================================
# general paths & constants
PATH_DATA = '/cluster/work/climate/evelynm/nw_inputs/FLALGA'
PATH_SAVE = '/cluster/work/climate/evelynm/nw_outputs/FLALGA'
PATH_EL_CONS_GLOBAL = PATH_DATA +'/final_consumption_iea_global.csv'

# =============================================================================
# Execution
# =============================================================================
if __name__ == '__main__': 
    mode = sys.argv[1] #'no_cideps' 'no_cideps', 'mod_cideps', 'low_vuln', 'high_vuln', 'low_thresh', 'high_thresh'

    if mode=='no_cideps':
        path_deps = PATH_DATA + '/dependencies_FLALGA_no_cideps.csv'
    elif mode=='long_paths':
        path_deps = PATH_DATA + '/dependencies_FLALGA_long_paths.csv'
    elif mode=='short_paths':
        path_deps = PATH_DATA + '/dependencies_FLALGA_short_paths.csv'
    else:
        path_deps = PATH_DATA + '/dependencies_FLALGA_telemod.csv'
    
    
    # In[6]:
    
    
    # =============================================================================
    # CI Data
    # =============================================================================
    
    
    # In[7]:
    
    
    # PEOPLE 
    gdf_people = gpd.read_file(PATH_DATA+'/people_flalga.shp')
    
    # POWER LINES
    gdf_powerlines = gpd.read_file(PATH_DATA+'/powerlines_flalga.shp')
    gdf_powerlines = gdf_powerlines[['TYPE','geometry','VOLT_CLASS', 'VOLTAGE']]
    gdf_powerlines['osm_id'] = 'n/a'
    
    # POWER PLANTS
    gdf_pplants = gpd.read_file(PATH_DATA+'/powerplants_flalga.shp') 
    gdf_pplants.rename({'NET_GEN':'estimated_generation_gwh_2017'}, axis=1, inplace=True)
    
    gdf_people, gdf_pplants = nwu.PowerFunctionalData().assign_el_prod_consump(
        gdf_people, gdf_pplants, 'USA', PATH_EL_CONS_GLOBAL)
    
    # HEALTH FACILITIES
    gdf_health = gpd.read_file(PATH_DATA+'/healthfacilities_flalga.shp')
    
    # EDUC. FACILITIES
    gdf_educ = gpd.read_file(PATH_DATA+'/educationfacilities_flalga.shp') 
    
    # TELECOM
    gdf_telecom = gpd.read_file(PATH_DATA+'/celltowers_flalga.shp') 
    
    # ROADS
    gdf_mainroads_osm = gpd.read_file(PATH_DATA+'/mainroads_osm_flalga.shp') 
    
    # WASTEWATER
    gdf_wastewater = gpd.read_file(PATH_DATA+'/wastewater_flalga.shp') 
    
    
    # In[9]:
    
    
    # =============================================================================
    # Networks Preprocessing
    # =============================================================================
    
    
    # In[10]:
    
    
    # POWER LINES
    power_network = Network(nodes=gpd.read_feather(PATH_DATA+'/pline_processed_n'),
                            edges=gpd.read_feather(PATH_DATA+'/pline_processed_e'))
    
    
    # In[11]:
    
    
    # ROAD
    road_network = Network(nodes=gpd.read_feather(PATH_DATA+'/roads_processed_n'),
                            edges=gpd.read_feather(PATH_DATA+'/roads_processed_e'))
    
    
    # In[12]:
    
    
    # PEOPLE
    __, gdf_people_nodes = NetworkPreprocess('people').preprocess(
        gdf_nodes=gdf_people)
    people_network = Network(nodes=gdf_people_nodes)
    
    # POWER PLANTS
    __, gdf_pp_nodes = NetworkPreprocess('power_plant').preprocess(
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
    
    
    # ### Creating an interdependent CI Network
    
    # In[13]:
    
    
    # MULTINET
    # add all CI networks into one
    cis_network = Network.from_nws([pplant_network, power_network, wastewater_network,
                                          people_network, health_network, educ_network,
                                          road_network, tele_network])
    cis_network.initialize_funcstates()
    
    
    # In[14]:
    
    
    # remove unnecessary variables
    for col in ['man_made', 'public_tra', 'bus']: #'TYPE','VOLT_CLASS', 'VOLTAGE',
        cis_network.edges.pop(col)
    for col in ['estimated_generation_gwh_2017','name' ]:
        cis_network.nodes.pop(col)
    #del pplant_network, power_network, wastewater_network, people_network, health_network, educ_network, road_network, tele_network
    
    
    # In[15]:
    
    
    # =============================================================================
    # Interdependencies
    # =============================================================================
    
    
    # In[16]:
    
    
    cis_graph = Graph(cis_network, directed=True)
    
    
    # In[17]:
    
    
    df_dependencies = pd.read_csv(path_deps, sep=',', header=0).replace({np.nan:None})
    
    
    # In[18]:
    
    
    # create "missing physical structures" - needed for real world flows
    cis_graph.link_vertices_closest_k('power_line', 'power_plant', link_name='power_line', bidir=True, k=1)
    cis_graph.link_vertices_closest_k('road', 'people',  link_name='road', 
                                      dist_thresh=df_dependencies[
                                          df_dependencies.source=='road'].thresh_dist.values[0],bidir=True, k=5)
    cis_graph.link_vertices_closest_k('road', 'health',  link_name='road',  bidir=True, k=1)
    cis_graph.link_vertices_closest_k('road', 'education', link_name='road',  bidir=True, k=1)
    
    
    # In[20]:
    
    
    # Create dependency edges in graph according to dependency df specs
    for __, row in df_dependencies.iterrows():
        cis_graph.place_dependency(row.source, row.target, 
                                   single_link=row.single_link,
                                   access_cnstr=row.access_cnstr, 
                                   dist_thresh=row.thresh_dist,
                                   preselect=False)
    cis_network = cis_graph.return_network()
    
    
    # In[22]:
    
    
    # =============================================================================
    # Base State
    # =============================================================================
    
    
    # In[23]:
    
    
    # Compute functionality states of all infrastructure components and service supplies for people
    cis_network.initialize_funcstates()
    for __, row in df_dependencies.iterrows():
        cis_network.initialize_capacity(row.source, row.target)
    for __, row in df_dependencies[
            df_dependencies['type_I']=='enduser'].iterrows():
        cis_network.initialize_supply(row.source)
    cis_network.nodes.pop('name')
    
    
    # In[24]:
    
    
    cis_graph = Graph(cis_network, directed=True)
    cis_graph.cascade(df_dependencies, p_source='power_plant', p_sink='power_line', 
                      source_var='el_generation', demand_var='el_consumption',
                      preselect=False,initial=True)
    cis_network = cis_graph.return_network()
    
    
    # In[25]:
    
    
    # Save Base State NW
    cis_network.nodes.to_feather(PATH_SAVE+f'/cis_nw_nodes_base_{mode}')
    
    
    # In[26]:
    
    
    base_stats = nwu.number_noservices(cis_graph,
                             services=['power', 'healthcare', 'education', 
                                       'telecom', 'mobility','water'])
    
    with open(PATH_SAVE +f'/base_stats_scenario_{mode}.pkl', 'wb') as f:
        pickle.dump(base_stats, f) 
    
    
    # In[27]:
    
    
    print(f'Summary of people without access to basic services: {base_stats}')
    
    
    # ### Impact Calculations
    
    # In[29]:
    
    
    # =============================================================================
    # Load Hazard
    # =============================================================================
    tc_michael = Hazard('TC').from_hdf5(PATH_DATA+'/tc_michael_1h.hdf5')
    tc_michael.check()
    
    
    # In[30]:
    
    
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
        
    if mode=='high_vuln':
        impfuncSet = ImpactFuncSet()
        #impfuncSet.append(impfunc)
        impfuncSet.append(impf_prob_high)
        impfuncSet.append(impf_educ_high)
        impfuncSet.append(impf_indus_high)
        impfuncSet.append(impf_road_high)
        impfuncSet.append(impf_tele_high)
        impfuncSet.append(impf_ppl2)    
    elif mode =='low_vuln':
        impfuncSet = ImpactFuncSet()
        #impfuncSet.append(impfunc)
        impfuncSet.append(impf_prob_low)
        impfuncSet.append(impf_educ_low)
        impfuncSet.append(impf_indus_low)
        impfuncSet.append(impf_road_low)
        impfuncSet.append(impf_tele_low)
        impfuncSet.append(impf_ppl2)
    else:
        impfuncSet = ImpactFuncSet()
        impfuncSet.append(impf_prob)
        impfuncSet.append(impf_educ)
        impfuncSet.append(impf_indus)
        impfuncSet.append(impf_road)
        impfuncSet.append(impf_tele)
        impfuncSet.append(impf_ppl2)
        impfuncSet.haz_type = 'TC'
    
    
    # In[ ]:
    
    
    
    
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
    
    
    # In[31]:
    
    
    ################
    # Exposures
    ################
    
    
    # In[32]:
    
    
    # POWER LINES
    # 500m p point, TC func 1
    res_pl = 500
    exp_line = Exposures(cis_network.edges[cis_network.edges.ci_type=='power_line'])
    exp_pnt = u_lp.exp_geom_to_pnt(exp_line, res=res_pl, to_meters=True, 
                                   disagg_met=u_lp.DisaggMethod.FIX, disagg_val=1)  
    exp_pnt.gdf[f'impf_TC'] = 1
    exp_pnt.set_lat_lon()
    exp_pnt.check()
    entity_pl = Entity()
    entity_pl.exposures.gdf = exp_pnt.gdf
    entity_pl.exposures.assign_centroids(tc_michael)
    entity_pl.impact_funcs = impfuncSet
    entity_pl.check()
    
    
    # In[33]:
    
    
    # ROAD
    # 500m p point, TC func 4
    res_rd = 500
    exp_line = Exposures(cis_network.edges[cis_network.edges.ci_type=='road'])
    exp_pnt = u_lp.exp_geom_to_pnt(exp_line, res=res_rd, to_meters=True, 
                                   disagg_met=u_lp.DisaggMethod.FIX, disagg_val=res_rd)  
    exp_pnt.gdf[f'impf_TC'] = 4
    exp_pnt.set_lat_lon()
    exp_pnt.check()
    entity_road = Entity()
    entity_road.exposures.gdf = exp_pnt.gdf
    entity_road.impact_funcs = impfuncSet
    entity_road.exposures.assign_centroids(tc_michael)
    entity_road.check()
    
    
    # In[34]:
    
    
    # HEALTHCARE
    entity_hc = Entity()
    exp_hc = Exposures()
    exp_hc.gdf = cis_network.nodes[cis_network.nodes.ci_type=='health']
    exp_hc.gdf['value'] = 1
    exp_hc.gdf['impf_TC'] = 3
    entity_hc.exposures.gdf = exp_hc.gdf
    entity_hc.exposures.set_lat_lon()
    entity_hc.exposures.assign_centroids(tc_michael)
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
    entity_educ.exposures.assign_centroids(tc_michael)
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
    entity_tele.exposures.assign_centroids(tc_michael)
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
    entity_wwater.exposures.assign_centroids(tc_michael)
    entity_wwater.impact_funcs = impfuncSet
    entity_wwater.check()
    
    # PEOPLE (for reference / validation purposes)
    entity_ppl = Entity()
    exp_ppl = Exposures()
    exp_ppl.gdf = gdf_people.rename({'counts':'value'}, axis=1)
    exp_ppl.gdf['impf_TC'] = 7
    entity_ppl.exposures.gdf = exp_ppl.gdf
    entity_ppl.exposures.set_lat_lon()
    entity_ppl.exposures.assign_centroids(tc_michael)
    entity_ppl.impact_funcs = impfuncSet
    entity_ppl.check()
    
    
    # In[35]:
    
    
    def calc_ci_impacts(hazard, exposures, impfuncSet, res_pl=500):
        """
        hazard: single event
        exposures: list of exposures
        """
        def binary_impact_from_prob(prob_vals, seed=47):
            np.random.seed = seed
            rand = np.random.random(prob_vals.size)
            return np.array([1 if p_fail > rnd else 0 for p_fail, rnd in 
                             zip(prob_vals, rand)])
    
        imp_list = []
        for exp in exposures:
            ci_type = exp.gdf.ci_type.iloc[0]
            imp = Impact()
            if ci_type =='road':
                imp.calc(exp, impfuncSet, hazard, save_mat=True)
                imp = u_lp.impact_pnt_agg(imp,  exp.gdf, u_lp.AggMethod.SUM)
            elif ci_type == 'power_line':
                # first convert failure probs for power line to binary failed / not failed
                imp.calc(exp, impfuncSet, hazard, save_mat=True)
                imp.imp_mat.data = binary_impact_from_prob(imp.imp_mat.data)
                # then to absolute length in metres failed
                imp.imp_mat.data = imp.imp_mat.data*res_pl    
                imp = u_lp.impact_pnt_agg(imp,  exp.gdf, u_lp.AggMethod.SUM)
            else:
                imp.calc(exp, impfuncSet, hazard)
            imp_list.append(imp)
        
        return imp_list
    
    
    # In[36]:
    
    
    imp_list = calc_ci_impacts(tc_michael, [entity.exposures for entity in [
        entity_pl, entity_road, entity_hc, entity_educ, entity_tele, entity_wwater]],
                               impfuncSet, res_pl=500)
    
    
    # In[39]:
    
    
    # Assign Direct Impacts to network
    cis_network.edges.loc[cis_network.edges.ci_type=='power_line', 'imp_dir'] = imp_list[0].eai_exp/(entity_pl.exposures.gdf.groupby(level=0).size()*res_pl) 
    # imp_list[0].eai_exp absolute impacts for Power Lines
    # if converting to fraction: add /(entity_pl.exposures.gdf.groupby(level=0).size()*res_pl) 
    cis_network.edges.loc[cis_network.edges.ci_type=='road', 'imp_dir'] = imp_list[1].eai_exp/(entity_road.exposures.gdf.groupby(level=0).size()*res_rd)
    # imp_list[1].eai_exp absolute impacts for roads
    #if converting to fraction: add  /(entity_road.exposures.gdf.groupby(level=0).size()*res_rd)
    cis_network.nodes.imp_dir.loc[cis_network.nodes.ci_type=='health'] = imp_list[2].eai_exp
    cis_network.nodes.imp_dir.loc[cis_network.nodes.ci_type=='education'] = imp_list[3].eai_exp
    cis_network.nodes.imp_dir.loc[cis_network.nodes.ci_type=='celltower'] = imp_list[4].eai_exp
    cis_network.nodes.imp_dir.loc[cis_network.nodes.ci_type=='wastewater'] = imp_list[5].eai_exp
    
    
    # ### Failure Cascades
    
    # In[40]:
    
    
    # IMPACT - FUNCTIONALITY LEVEL (INTERNAL)
    if mode=='low_thresh':
        THRESH_PLINE = 0.001# if relative, change to 0.01, if absolute change to 500
        THRESH_ROAD = 0.15 # if relative, change to 0.5, if absolute change to 1000
        THRESH_HEALTH = 0.15
        THRESH_EDUC = 0.15
        THRESH_WW = 0.15
        THRESH_CT = 0.15
    
    elif mode=='high_thresh':
        THRESH_PLINE = 0.1# if relative, change to 0.01, if absolute change to 500
        THRESH_ROAD = 0.75 # if relative, change to 0.5, if absolute change to 1000
        THRESH_HEALTH = 0.5
        THRESH_EDUC = 0.5
        THRESH_WW = 0.5
        THRESH_CT = 0.5
    
    else:
        THRESH_PLINE = 0.01# if relative, change to 0.01, if absolute change to 500
        THRESH_ROAD = 0.5 # if relative, change to 0.5, if absolute change to 1000
        THRESH_HEALTH = 0.3
        THRESH_EDUC = 0.3
        THRESH_WW = 0.3
        THRESH_CT = 0.3
    
    # Assign internal functionality level to network components
    cond_fail_pline = ((cis_network.edges.ci_type=='power_line') & 
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
    
    # Update total functional states
    cis_network.edges['func_tot'] = [np.min([func_internal, func_tot]) 
                                     for func_internal, func_tot in 
                                     zip(cis_network.edges.func_internal, 
                                         cis_network.edges.func_tot)]
    
    cis_network.nodes['func_tot'] = [np.min([func_internal, func_tot]) 
                                     for func_internal, func_tot in 
                                     zip(cis_network.nodes.func_internal, 
                                         cis_network.nodes.func_tot)]
    
    
    # In[42]:
    
    
    ###################
    # Impact Cascade
    ###################
    
    
    # In[43]:
    
    
    cis_network.nodes.pop('name')
    graph_disr = Graph(cis_network, directed=True)
    graph_disr.cascade(df_dependencies, p_source='power_plant', p_sink='power_line', 
                              source_var='el_generation', demand_var='el_consumption', 
                              preselect=False)
    
    
    # In[ ]:
    
    
    # save selected impact results as dataframe
    cis_network = graph_disr.return_network()
    vars_to_keep_edges = ['ci_type', 'func_internal', 'func_tot', 'imp_dir','geometry']
    vars_to_keep_nodes = vars_to_keep_edges.copy() 
    vars_to_keep_nodes.extend([colname for colname in cis_network.nodes.columns 
                                       if 'actual_supply_' in colname])
    vars_to_keep_nodes.extend(['counts'])
    df_res = cis_network.nodes[cis_network.nodes.ci_type=='people'][vars_to_keep_nodes]
    for ci_type in ['health', 'education', 'celltower', 'power_plant', 'wastewater']:
        df_res = df_res.append(cis_network.nodes[cis_network.nodes.ci_type==ci_type][vars_to_keep_nodes])
    for ci_type in ['power_line', 'road']:
            df_res = df_res.append(cis_network.edges[cis_network.edges.ci_type==ci_type][vars_to_keep_edges])
    df_res.to_feather(PATH_SAVE+f'/cascade_results_{mode}')
    
    
    # In[45]:
    
    
    # save people's service impacts as dict
    disaster_stats = nwu.disaster_impact_allservices(cis_graph, graph_disr,
                    services=['power', 'healthcare', 'education', 'telecom', 'mobility', 'water'])
    with open(PATH_SAVE+f'/disaster_stats_scen_{mode}.pkl', 'wb') as f:
             pickle.dump(disaster_stats, f)


# In[ ]:




