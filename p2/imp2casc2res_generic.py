#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:55:57 2022

@author: evelynm
---

Parallel calculation per hazard via impact cascade to saving results.

"""

import os
import sys
import geopandas as gpd
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool
import itertools
from datetime import datetime
from scipy.sparse import csr_matrix

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.engine import Impact
from climada.hazard.base import Hazard
from climada.util import lines_polys_handler as u_lp
from climada.util import coordinates as u_coord
from climada.util.api_client import Client

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import Graph
import climada_petals.engine.networks.nw_utils as nwu


START_STR = '01-01-2000'
END_STR = '31-12-2021'

# =============================================================================
# Impact Class Defs
# =============================================================================

class ImpFuncsCIFlood():
    
    def __init__(self):
        self.tag = 'FL'
        self.road = self.step_impf()
        self.residential_build = self.step_impf()
        self.industrial_build = self.step_impf()
        self.power_line = self.step_impf()
        self.power_plant = self.step_impf()
        self.water_plant = self.step_impf()
        self.celltower = self.step_impf()
        self.education = self.residential_build
        self.health = self.industrial_build
        
        
    def step_impf(self):
        step_impf = ImpactFunc() 
        step_impf.id = 1
        step_impf.haz_type = 'FL'
        step_impf.name = 'Step function flood'
        step_impf.intensity_unit = ''
        step_impf.intensity = np.array([0, 0.95,0.955, 1])
        step_impf.mdd =       np.array([0, 0, 1, 1])
        step_impf.paa =       np.sort(np.linspace(1, 1, num=4))
        step_impf.check()
        return step_impf
    
    def no_impf(self):
        no_impf = ImpactFunc() 
        no_impf.id = 2
        no_impf.haz_type = 'FL'
        no_impf.name = 'No impact function flood'
        no_impf.intensity_unit = ''
        no_impf.intensity = np.array([0, 1])
        no_impf.mdd =       np.array([0, 0])
        no_impf.paa =       np.sort(np.linspace(1, 1, num=2))
        no_impf.check()
        return no_impf
    
    def resid_impf(self):
        pass
    def indus_impf(self):
        pass
    def road_impf(self):
        pass
    def tele_impf(self):
        pass
    def pp_impf(self):
        pass
    def pl_impf(self):
        pass


class ImpFuncsCIWind():
    
    def __init__(self):
        self.tag = 'TC'
        self.road = self.road_impf()
        self.residential_build = self.resid_impf()
        self.industrial_build = self.indus_impf()
        self.power_line = self.pl_impf()
        self.power_plant = self.no_impf()
        self.water_plant = self.no_impf()
        self.celltower = self.tele_impf()
        self.people = self.people_impf()
        self.education = self.residential_build
        self.health = self.industrial_build
        
    def road_impf(self):
        # Road adapted from Koks et al. 2019 (tree blowdown on road > 42 m/s)
        impf_road = ImpactFunc() 
        impf_road.id = 2
        impf_road.haz_type = 'TC'
        impf_road.name = 'Loss func. for roads from tree blowdown'
        impf_road.intensity_unit = 'm/s'
        #impf_road.intensity = np.array([0, 30, 35, 42, 48, 120])
        impf_road.intensity = np.array([0, 20, 30, 40, 50, 120])
        impf_road.mdd =       np.array([0, 0,   0, 50, 100, 100]) / 100
        impf_road.paa = np.sort(np.linspace(1, 1, num=6))
        impf_road.check()
        return impf_road
    
    def resid_impf(self):
        # adapted from figure H.13 (residential 2-story building) loss function, Hazus TC 2.1 (p.940)
        # medium terrain roughness parameter (z_theta = 0.35)
        impf_educ = ImpactFunc() 
        impf_educ.id = 5
        impf_educ.tag = 'TC educ'
        impf_educ.haz_type = 'TC'
        impf_educ.name = 'Loss func. residental building z0 = 0.35'
        impf_educ.intensity_unit = 'm/s'
        impf_educ.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
        impf_educ.mdd =       np.array([0, 0,  5,  20,  50,  80,  98,  80,  98, 100, 100, 100, 100]) / 100
        impf_educ.paa = np.sort(np.linspace(1, 1, num=13))
        impf_educ.check()
        return impf_educ


    def indus_impf(self):
        # adapted from figure N.1 (industrial 2 building) loss function, Hazus TC 2.1 (p.1115)
        # medium terrain roughness parameter (z_theta = 0.35)
        impf_indus = ImpactFunc() 
        impf_indus.id = 4
        impf_indus.haz_type = 'TC'
        impf_indus.name = 'Loss func. industrial building z0 = 0.35'
        impf_indus.intensity_unit = 'm/s'
        impf_indus.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
        impf_indus.mdd =       np.array([0, 0,   0,   5,  15,  70,  98, 100, 100, 100, 100, 100, 100]) / 100
        impf_indus.paa = np.sort(np.linspace(1, 1, num=13))
        impf_indus.check()
        return impf_indus
        
    def no_impf(self):
        impf_none = ImpactFunc() 
        impf_none.id = 6
        impf_none.haz_type = 'TC'
        impf_none.name = 'No-impact func'
        impf_none.intensity_unit = 'm/s'
        impf_none.intensity = np.array([0,  140])  
        impf_none.mdd =       np.array([0, 0 ])         
        impf_none.paa = np.sort(np.linspace(1, 1, num=2))
        impf_none.check()
        return impf_none

    def tele_impf(self):
        # adapted from newspaper articles ("cell towers to withstand up to 110 mph")
        impf_tele = ImpactFunc() 
        impf_tele.id = 3
        impf_tele.haz_type = 'TC'
        impf_tele.name = 'Loss func. cell tower'
        impf_tele.intensity_unit = 'm/s'
        impf_tele.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]) / 2.237 #np.linspace(0, 120, num=13)
        impf_tele.mdd =       np.array([0, 0,   0,  0, 100,  100,  100,  100, 100, 100, 100, 100, 100]) / 100
        impf_tele.paa = np.sort(np.linspace(1, 1, num=13))
        impf_tele.check()
        return impf_tele
   
    def p_fail_pl(self, v_eval, v_crit=30, v_coll=60):
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
    
    def pl_impf(self, v_crit=30, v_coll=60):
        # Power line
        v_eval = np.linspace(0, 120, num=120)
        p_fail_powerlines = self.p_fail_pl(v_eval, v_crit=v_crit, v_coll=v_coll)
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
        return impf_prob
    
    def people_impf(self):
        # Mapping of wind field >= hurricane scale 1 (33 m/s)
        impf_ppl = ImpactFunc() 
        impf_ppl.id = 7
        impf_ppl.haz_type = 'TC'
        impf_ppl.name = 'People - Windfield Mapping >= TC'
        impf_ppl.intensity_unit = 'm/s'
        impf_ppl.intensity = np.array([0, 32, 33, 80, 100, 120, 140, 160]) 
        impf_ppl.mdd = np.array([0, 0,   100,  100,   100,  100,  100,  100]) / 100
        impf_ppl.paa = np.sort(np.linspace(1, 1, num=8))
        impf_ppl.check()
        return impf_ppl

    def binary_impact_from_prob(self, impact, seed=47):
        np.random.seed = seed
        rand = np.random.random(impact.eai_exp.size)
        return np.array([1 if p_fail > rnd else 0 for p_fail, rnd in 
                         zip(impact.eai_exp, rand)])

class ImpactThresh():
    def __init__(self):
        self.road = 300
        self.power_line = 500
        self.industrial_build = 0.5
        self.residential_build = 0.5
        self.celltower = 0.5
        self.power_plant = 0.5
        self.water_plant = 0.5
        self.health = 0.5
        self.education = 0.5
        

# =============================================================================
# Impact calc funcs
# =============================================================================

def exposure_from_network_nodes(ci_network, ci_type, imp_class):
    
    exp_pnt = Exposures(ci_network.nodes[ci_network.nodes.ci_type==ci_type])
    exp_pnt.gdf[f'impf_{imp_class.tag}'] = getattr(imp_class, ci_type).id
    exp_pnt.gdf['value'] = 1
    exp_pnt.set_lat_lon()
    exp_pnt.check()
    
    return exp_pnt

        
def exposure_from_network_edges(ci_network, ci_type, imp_class, res, 
                                disagg_met=u_lp.DisaggMethod.FIX, disagg_val=None):
             
    exp_line = Exposures(ci_network.edges[ci_network.edges.ci_type==ci_type])
    if not disagg_val:
        disagg_val = res
    exp_pnt = u_lp.exp_geom_to_pnt(exp_line, res=res, to_meters=True, 
                                   disagg_met=disagg_met, disagg_val=disagg_val)  
    exp_pnt.gdf[f'impf_{imp_class.tag}'] = getattr(imp_class, ci_type).id
    exp_pnt.set_lat_lon()
    exp_pnt.check()
    
    return exp_pnt

def calc_ci_impacts(hazard, exposures, imp_class):
    """
    hazard: single event
    """
    impfuncSet = ImpactFuncSet()
    imp_list = []
    
    for exp in exposures:
        ci_type = exp.gdf.ci_type.iloc[0]
        impfuncSet.append(getattr(imp_class, ci_type))
        imp = Impact()
        if ci_type in ['road', 'power_line']:
            imp.calc(exp, impfuncSet, hazard, save_mat=True)
            if (imp_class.tag=='TC') and (ci_type =='power_line'):
                imp.imp_mat =  csr_matrix(imp.imp_mat/exp.gdf.value.values)
                
                #imp.eai_exp = imp.eai_exp/exp.gdf.value.values
                imp.eai_exp = imp_class.binary_impact_from_prob(imp)*exp.gdf.value.values
                
                exp.gdf['imp_dir']=imp.eai_exp
                exp.gdf.groupby(level=0).imp_dir.sum()
            imp = u_lp.impact_pnt_agg(imp,  exp.gdf, u_lp.AggMethod.SUM)
        else:
            imp.calc(exp, impfuncSet, hazard)
        imp_list.append(imp)
    
    return imp_list


def impacts_to_graph(imp_list, exposures, ci_network):
    
    ci_net = deepcopy(ci_network)
    
    for ix, exp in enumerate(exposures):
        ci_type = exp.gdf.ci_type.iloc[0]            
        func_states = list(
            map(int, imp_list[ix].eai_exp<=getattr(ImpactThresh(), ci_type)))
        if ci_type in ['road', 'power_line']:
            ci_net.edges.loc[ci_net.edges.ci_type==ci_type, 'func_internal'] = func_states
        else:
            ci_net.nodes.loc[ci_net.nodes.ci_type==ci_type, 'func_internal'] = func_states
    ci_net.edges['func_tot'] = [np.min([func_internal, func_tot]) for 
                                func_internal, func_tot in zip(
                                    ci_net.edges.func_internal, 
                                    ci_net.edges.func_tot)]
    ci_net.nodes['func_tot'] = [np.min([func_internal, func_tot]) for 
                                func_internal, func_tot in zip(
                                    ci_net.nodes.func_internal, 
                                    ci_net.nodes.func_tot)]

    return Graph(ci_net, directed=True)


def number_noservice(service, df_res):
    no_service = (1-df_res[df_res.ci_type=='people'][nwu.service_dict()[service]])
    pop = df_res[df_res.ci_type=='people'].counts

    return (no_service*pop).sum()

def number_noservices(df_res, services=['power', 'healthcare', 'education', 
                                        'telecom', 'mobility', 'water']):
    servstats_dict = {}
    for service in services:
        servstats_dict[service] = number_noservice(service, df_res)
    return servstats_dict

def disaster_impact_allservices(pre_graph, df_res, services= ['power', 
                                'healthcare', 'education', 'telecom',
                                'mobility', 'water']):

    dict_pre = nwu.number_noservices(pre_graph,services)
    dict_post = number_noservices(df_res,services)
    dict_delta = {}
    for key, value in dict_post.items():
        dict_delta[key] = value-dict_pre[key]
    
    return dict_delta

def calc_cascade(hazard, exp_list, df_dependencies, graph_base, path_save,
                 imp_class):    
    
    print('Entered function')
    # impact calculation 
    if not os.path.isfile(path_save+f'cascade_results_{hazard.event_name[0]}'):

        imp_list = calc_ci_impacts(hazard, exp_list, imp_class)
        ci_network = graph_base.return_network()
        ci_network.nodes = ci_network.nodes.drop('name', axis=1)
        graph_disr = impacts_to_graph(imp_list, exp_list, ci_network)
        
        # cascade calculation
        # hard-coded kwargs for cascade(). change in future                     
        graph_disr.cascade(df_dependencies, p_source='power_plant', p_sink='power_line', 
                          source_var='el_gen_mw', demand_var='el_load_mw', #'el_generation' 'el_consumption'
                          preselect=False)
        
        # save selected results    
        ci_net = graph_disr.return_network()
        vars_to_keep_edges = ['ci_type', 'func_internal', 'func_tot', 'imp_dir',
                              'geometry']
        vars_to_keep_nodes = vars_to_keep_edges.copy() 
        vars_to_keep_nodes.extend([colname for colname in ci_net.nodes.columns 
                                   if 'actual_supply_' in colname])
        vars_to_keep_nodes.extend(['counts'])
        
        df_res = ci_net.nodes[ci_net.nodes.ci_type=='people'][vars_to_keep_nodes]
        for ci_type in ['health', 'education', 'celltower', 'power_plant']:
            df_res = df_res.append(ci_net.nodes[ci_net.nodes.ci_type==ci_type]
                                   [vars_to_keep_nodes])
        for ci_type in ['power_line', 'road']:
            df_res = df_res.append(ci_net.edges[ci_net.edges.ci_type==ci_type]
                                   [vars_to_keep_edges])
        df_res.to_feather(path_save+f'cascade_results_{hazard.event_name[0]}')
        
        del ci_network
        del df_res
        del ci_net
    
        return nwu.disaster_impact_allservices(
            graph_base, graph_disr, services =['power', 'healthcare', 
                                               'education', 'telecom', 
                                               'mobility'])
    else:
        df_res = gpd.read_feather(path_save+f'cascade_results_{hazard.event_name[0]}')
        return disaster_impact_allservices(
            graph_base, df_res, services =['power', 'healthcare', 
                                           'education', 'telecom', 
                                           'mobility'])
 
def get_selected_tcs_api(iso3, start=START_STR, end=END_STR):
    
    client = Client()
    tc = client.get_hazard('tropical_cyclone', 
                           properties={'country_iso3alpha':iso3, 
                                       'climate_scenario': 'historical',
                                       'spatial_coverage': 'country'})
    # only historic ones                                                     
    tc = tc.select(orig=True)       
    
    # only in between cloud to street DB times                                                      
    startdate_ordinal = datetime.strptime(start, '%d-%m-%Y').date().toordinal()
    enddate_ordinal = datetime.strptime(end, '%d-%m-%Y').date().toordinal()
    date_selectors = (tc.date>=startdate_ordinal)&(tc.date<=enddate_ordinal)
    tc = tc.select(event_id=list(tc.event_id[date_selectors]))
    
    # only nonzeros
    inten_selectors = []
    for event_id in tc.event_id:
        if tc.select(event_id=[event_id]).intensity.nnz>0:
                 inten_selectors.append(event_id)
    tc = tc.select(event_id=inten_selectors)
    
    # only reasonably strong ones
    inten_selectors = []
    for event_id in tc.event_id:
        if np.max(tc.select(event_id=[event_id]).intensity.data)>20:
            inten_selectors.append(event_id)
    tc = tc.select(event_id=inten_selectors)
    
    return tc
           
# =============================================================================
# Execution
# =============================================================================
if __name__ == '__main__': 
    
    cntry = sys.argv[1]
    haz_type = sys.argv[2]
    #parallel = sys.argv[3]
    iso3 = u_coord.country_to_iso(cntry)
    path_root = f'/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/output_multi_cis/{iso3}'
    path_edges  = f'{path_root}/graphs/cis_nw_edges'
    path_nodes = f'{path_root}/graphs/cis_nw_nodes'
    path_save = f'{path_root}/figures'
    path_deps = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/dependency_tables/dependencies_HND.csv'

    # load necessary files
    ci_network = Network(edges=gpd.read_feather(path_edges), 
                         nodes=gpd.read_feather(path_nodes))
    
    ci_network.nodes = ci_network.nodes.drop('name', axis=1)
    graph_base = Graph(ci_network, directed=True)
    df_dependencies = pd.read_csv(path_deps)

    if haz_type=='FL':
        path_haz=path_save+f'flood_{iso3}.hdf5'
        hazards = Hazard('FL').from_hdf5(path_haz)
        
    elif haz_type=='TC':
        hazards = get_selected_tcs_api(iso3)

    impfunc_set = ImpFuncsCIFlood() if haz_type=='FL' else ImpFuncsCIWind()
    # make exposures 
    exp_list = []
    for ci_type in ['power_plant', 'celltower', 'health', 'education']:
        exp_list.append(exposure_from_network_nodes(ci_network, ci_type, impfunc_set))
    
    for ci_type in ['power_line', 'road']:
        exp_list.append(exposure_from_network_edges(ci_network, ci_type, impfunc_set,res=300))
        
    # with Pool() as pool:
    #     exp_list.extend(pool.starmap(exposure_from_network_edges, zip(
    #         itertools.repeat(ci_network, 2),
    #         ['power_line', 'road'],
    #         itertools.repeat(impfunc_set, 2),
    #         [300, 300])))
    
    haz_list = [hazards.select(event_names=[event_name]) for event_name in hazards.event_name]
    n_events = len(haz_list)
    
    # if not parallel:
    dict_list = []
    for haz in haz_list:
        dict_list.append(calc_cascade(haz, exp_list, df_dependencies, graph_base, path_save, impfunc_set))
    # else:
    # # Parallelize
    #     with Pool() as pool:
    #         dict_list = pool.starmap(calc_cascade, zip(
    #                               haz_list, 
    #                               itertools.repeat(exp_list, n_events), 
    #                               itertools.repeat(df_dependencies, n_events), 
    #                               itertools.repeat(graph_base, n_events),
    #                               itertools.repeat(path_save, n_events),
    #                               itertools.repeat(impfunc_set, n_events)))
        

    service_dict = dict(zip(hazards.event_name, dict_list))

    with open(path_save+f'service_stats_{haz_type}_{iso3}.pkl', 'wb') as f:
         pickle.dump(service_dict, f)    
