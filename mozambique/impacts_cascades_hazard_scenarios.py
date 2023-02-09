
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
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pickle
import shapely

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.engine import Impact
from climada.hazard.base import Hazard, Centroids
from climada.hazard import TCTracks, TropCyclone
from climada.util import lines_polys_handler as u_lp
from climada.util import coordinates as u_coord

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import Graph
import climada_petals.engine.networks.nw_utils as nwu


# =============================================================================
# Impact Class Defs
# =============================================================================

class ImpFuncsCIFlood():
    
    def __init__(self):
        self.tag = 'FL'
        self.road = self.step_impf()
        self.health = self.step_impf()
        self.education = self.step_impf()
        self.power_line_e = self.no_impf() # for edges!
        self.power_line_n = self.step_impf() # for nodes! (aka substations)
        self.power_tower = self.no_impf()
        self.power_plant = self.no_impf()
        self.water_plant = self.no_impf()
        self.celltower = self.step_impf()
        self.people = self.step_impf()
        
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


class ImpFuncsCIWind():
    
    def __init__(self):
        self.tag = 'TC'
        self.road = self.road_impf() # for edges!
        self.education = self.resid_impf()
        self.health = self.health_level_1_4_impf()
        self.power_line_e = self.pl_impf() # for edges!
        self.power_line_n = self.no_impf() # for nodes!
        self.power_tower = self.pt_impf()
        self.power_plant = self.no_impf()
        self.water_plant = self.no_impf()
        self.celltower = self.tele_impf()
        self.people = self.people_impf()
        
    def road_impf(self):
        # Road adapted from Koks et al. 2019 (tree blowdown on road > 42 m/s)
        # anecdotal case-study Mühlhofer et al. 2022 (over-estimate) that for 42 m/s all roads sustain 100% damage
        # calibrated with reports from MOZ Idai --> sigmoid function with v 1/2 at 42 m/s and max dmg at 30%
        v_eval = np.linspace(0, 120, num=120)
        L=30
        k=0.3
        x0=42
        mdd = []
        for v in v_eval:
            mdd.append(L/(1 + np.exp(-k * (v-x0)))/100)
        impf_road = ImpactFunc() 
        impf_road.id = 2
        impf_road.haz_type = 'TC'
        impf_road.name = 'Road dmg function for tree blowdown from strong winds'
        impf_road.intensity_unit = 'm/s'
        impf_road.intensity = np.array(v_eval)
        impf_road.mdd = np.array(mdd)
        impf_road.paa = np.sort(np.linspace(1, 1, num=120))
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
        impf_educ.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 260]) / 2.237
        impf_educ.mdd =       np.array([0, 0,  5,  20,  50,  80,  98,  98,  98, 100, 100]) / 100
        impf_educ.paa = np.sort(np.linspace(1, 1, num=11))
        impf_educ.check()
        return impf_educ

    def health_level_1_4_impf(self):
        impf_health_1_4 = ImpactFunc()
        impf_health_1_4.id = 9
        impf_health_1_4.haz_type = 'TC'
        impf_health_1_4.name = 'Step function health'
        impf_health_1_4.intensity_unit = 'm/s'
        impf_health_1_4.intensity = np.array([12, 18, 20, 40, 50, 60, 70, 90])
        impf_health_1_4.mdd = np.array([0, 0.03, 0.3, 0.5, 0.6, 0.61, 0.61, 0.61])
        impf_health_1_4.paa = np.ones(impf_health_1_4.intensity.shape)
        impf_health_1_4.check()
        return impf_health_1_4

    def health_level_1_4_impf_adapt(self):
        impf_health_1_4 = ImpactFunc()
        impf_health_1_4.id = 10
        impf_health_1_4.haz_type = 'TC'
        impf_health_1_4.name = 'Step function health adapted'
        impf_health_1_4.intensity_unit = 'm/s'
        impf_health_1_4.intensity = np.array([21, 26, 29, 48, 60, 65, 70, 90])
        impf_health_1_4.mdd = np.array([0, 0.03, 0.28, 0.45, 0.54, 0.55, 0.55, 0.55])
        impf_health_1_4.paa = np.ones(impf_health_1_4.intensity.shape)
        impf_health_1_4.check()
        return impf_health_1_4

    def indus_impf(self):
        # adapted from figure N.1 (industrial 2 building) loss function, Hazus TC 2.1 (p.1115)
        # medium terrain roughness parameter (z_theta = 0.35)
        impf_indus = ImpactFunc() 
        impf_indus.id = 4
        impf_indus.haz_type = 'TC'
        impf_indus.name = 'Loss func. industrial building z0 = 0.35'
        impf_indus.intensity_unit = 'm/s'
        impf_indus.intensity = np.array([0, 30, 60, 80, 100, 120, 140, 160, 180, 200, 260]) / 2.237 
        impf_indus.mdd =       np.array([0, 0,   0,   5,  15,  70,  98, 100, 100, 100, 100]) / 100
        impf_indus.paa = np.sort(np.linspace(1, 1, num=11))
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
        impf_tele.intensity = np.array([0, 80, 100, 260]) / 2.237 #np.linspace(0, 120, num=13)
        impf_tele.mdd =       np.array([0,  0, 100,  100]) / 100
        impf_tele.paa = np.sort(np.linspace(1, 1, num=4))
        impf_tele.check()
        return impf_tele
   
    def p_fail_pl(self, v_eval, v_crit=30, v_coll=60):
        """
        adapted from  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7801854
         Vulnerability Assessment for Power Transmission Lines under Typhoon 
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
    
    def pl_impf(self, v_crit=40, v_coll=75):
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
    
    def pl_impf_adapt(self, v_crit=50, v_coll=85):
        # Power line
        v_eval = np.linspace(0, 120, num=120)
        p_fail_powerlines = self.p_fail_pl(v_eval, v_crit=v_crit, v_coll=v_coll)
        impf_prob = ImpactFunc() 
        impf_prob.id = 11
        impf_prob.tag = 'PL_Prob'
        impf_prob.haz_type = 'TC'
        impf_prob.name = 'power line failure prob, adapted'
        impf_prob.intensity_unit = 'm/s'
        impf_prob.intensity = np.array(v_eval)
        impf_prob.mdd = np.array(p_fail_powerlines)
        impf_prob.paa = np.sort(np.linspace(1, 1, num=120))
        impf_prob.check()
        return impf_prob
    
    def p_fail_powertower(self, v_eval, L=97.2, x0=77.8, k=0.3):
        """
        L=95, x0=80, k=0.1:
        adapted from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7801854&tag=1,
        based on 'base' class tower
        
        L=97.2, x0=77.8, k=0.3: adapted from https://arxiv.org/abs/2107.06072 Fragility curves for 
        power transmission towers in Odisha, India, 
        based on observed damage during 2019 Cyclone Fani; based on functionality disruption curve
        """
        p_fail = []
        for v in v_eval:
            p_fail.append(L/(1 + np.exp(-k * (v-x0)))/100)        
        return p_fail
    
    def pt_impf(self, L=95, x0=85, k=0.1):
        # Power tower impact function 
        v_eval = np.linspace(0, 140, num=140)
        p_fail_powertower = self.p_fail_powertower(v_eval, L=L, x0=x0, k=k) 
        impf_pt = ImpactFunc() 
        impf_pt.id = 8
        impf_pt.haz_type = 'TC'
        impf_pt.name = 'Disruption func. for power towers from strong winds'
        impf_pt.intensity_unit = 'm/s'
        impf_pt.intensity = np.array(v_eval)
        impf_pt.mdd = np.array(p_fail_powertower)
        impf_pt.paa = np.sort(np.linspace(1, 1, num=140))
        impf_pt.check()
        return impf_pt
    
    def pt_impf_adapt(self, L=95, x0=85, k=0.2):
        # Power tower impact function 
        v_eval = np.linspace(0, 140, num=140)
        p_fail_powertower = self.p_fail_powertower(v_eval, L=L, x0=x0, k=k) 
        impf_pt = ImpactFunc() 
        impf_pt.id = 12
        impf_pt.haz_type = 'TC'
        impf_pt.name = 'Disruption func. for power towers from strong winds, adapted'
        impf_pt.intensity_unit = 'm/s'
        impf_pt.intensity = np.array(v_eval)
        impf_pt.mdd = np.array(p_fail_powertower)
        impf_pt.paa = np.sort(np.linspace(1, 1, num=140))
        impf_pt.check()
        return impf_pt
 
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
    
class ImpactThresh():
    def __init__(self):
        self.road = 1000
        self.power_line_e = 500*0.4
        self.power_line_n = 0.4
        self.power_tower = 500*0.4
        self.celltower = 0.4
        self.power_plant = 0.4
        self.water_plant = 0.4
        self.health = 0.4
        self.education = 0.4
        self.people = np.inf

def ImpfClassDict():
    return {
        'FL' : ImpFuncsCIFlood(),
        'TC' : ImpFuncsCIWind(),
    }

# =============================================================================
# Hazard loading funcs
# =============================================================================

def load_wind_idai(cntry_shape, storm_id='2019063S18038'):
    "re-compute to 10 mins sustained winds for coherence with impact functions"
    tr_idai = TCTracks.from_ibtracs_netcdf(storm_id=storm_id)
    tr_idai.data[0]['max_sustained_wind']*= 0.88
    tr_idai.equal_timestep()
    min_lat, max_lat, min_lon, max_lon = cntry_shape.bounds[1], cntry_shape.bounds[3], cntry_shape.bounds[0], cntry_shape.bounds[2]
    cent_moz = Centroids.from_pnt_bounds((min_lon, min_lat, max_lon, max_lat), res=0.05)
    cent_moz.check()
    tc_idai = TropCyclone.from_tracks(tr_idai, centroids=cent_moz)
    tc_idai.check()
    return tc_idai

def load_flood_idai(path_fl_file):
    fl_idai = Hazard.from_raster([path_fl_file], dst_crs='epsg:4326', haz_type='FL') 
    fl_idai.centroids.set_meta_to_lat_lon() #Hazard('FL').from_raster(path_fl_file)
    fl_idai.event_name = ['Flood_Idai']
    return fl_idai       
        
# =============================================================================
# Impact & cascade calc funcs
# =============================================================================

def gdf_from_network(df_edges_or_nodes, ci_type):
    return df_edges_or_nodes[df_edges_or_nodes['ci_type']==ci_type]

def exposure_from_nodes(gdf, tag=None, value=1):
    exp_pnt = Exposures(gdf)
    exp_pnt.tag = tag if tag is not None else gdf.ci_type.iloc[0]
    exp_pnt.gdf['value'] = value
    exp_pnt.set_lat_lon()
    exp_pnt.check()
    return exp_pnt
      
def exposure_from_edges(gdf, res, tag=None, disagg_met=u_lp.DisaggMethod.FIX, disagg_val=1):
    exp_line = Exposures(gdf)
    if not disagg_val:
        disagg_val = res
    exp_pnt = u_lp.exp_geom_to_pnt(exp_line, res=res, to_meters=True, 
                                   disagg_met=disagg_met, disagg_val=disagg_val)
    exp_pnt.tag = tag if tag is not None else gdf.ci_type.iloc[0]
    exp_pnt.set_lat_lon()
    exp_pnt.check() 
    return exp_pnt

def make_impfset(imp_class):
    impfset = ImpactFuncSet()
    for attribute in set(imp_class.__dict__.keys()).difference({'tag'}):
        impfset.append(getattr(imp_class, attribute))
    return impfset

def assign_impfs(exp, haz_type, impfclass_dict=ImpfClassDict()):
    exp.gdf[f'impf_{haz_type}'] = getattr(impfclass_dict[haz_type], exp.tag).id
    
def calc_point_impacts(haz, exp, impf_set):
    imp = Impact()
    imp.calc(exp, impf_set, haz, save_mat=True)
    return imp

def binary_impact_from_prob(imp, samples=1, seed=47):
    """
    where impact funcs were given as failure probability on y-axis: sample failure states
    e.g. for wind damage to power lines, and wind damage to power towers
    """
    np.random.seed = seed
    data_sampled = np.zeros((samples, imp.imp_mat.data.size))
    for sample in np.arange(samples):
        rand = np.random.random(imp.imp_mat.data.size)
        data_sampled[sample] = np.array([1 if p_fail > rnd else 0 for p_fail, rnd in 
                                     zip(imp.imp_mat.data, rand)])
    imp.imp_mat.data = data_sampled.mean(0,)
    return imp

def binary_to_origres(imp, orig_res):
    """
    where failure states were sampled, assign real underlying damage value
    """
    imp.imp_mat.data = imp.imp_mat.data*orig_res
    return imp

def impacts_to_network(imp, exp_tag, ci_network_disr):          
    func_states = list(
            map(int, imp.imp_mat.toarray().flatten()<=getattr(ImpactThresh(), exp_tag)))
    
    if exp_tag == 'road':
        ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='road',
                                  'func_internal'] = func_states
        ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='road',
                                  'imp_dir'] = imp.imp_mat.toarray().flatten()
    elif exp_tag in ['power_line_e', 'power_tower']:
        # power line edges and power towers are two impacts that are summed on the same exposure
        ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='power_line',
                                  'func_internal'] = [np.min(
            [func_state, func_internal]) for func_state, func_internal in zip(
            func_states, ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='power_line', 'func_internal'])]
        ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='power_line', 'imp_dir'] += imp.imp_mat.toarray().flatten()
        # power lines in one direction need to be impacted also in reverse direction

    elif exp_tag=='power_line_n':
        ci_network_disr.nodes.loc[
                ci_network_disr.nodes.ci_type=='power_line', 'func_internal'] = func_states
        ci_network_disr.nodes.loc[
                ci_network_disr.nodes.ci_type=='power_line', 'imp_dir'] = imp.imp_mat.toarray().flatten()
        
    else:
        ci_network_disr.nodes.loc[
                ci_network_disr.nodes.ci_type==exp_tag, 'func_internal'] = func_states
        ci_network_disr.nodes.loc[
                ci_network_disr.nodes.ci_type==exp_tag, 'imp_dir'] = imp.imp_mat.toarray().flatten()
        
    ci_network_disr.edges['func_tot'] = [np.min([func_internal, func_tot]) for 
                                          func_internal, func_tot in zip(
                                              ci_network_disr.edges.func_internal, 
                                              ci_network_disr.edges.func_tot)]
    ci_network_disr.nodes['func_tot'] = [np.min([func_internal, func_tot]) for 
                                         func_internal, func_tot in zip(
                                             ci_network_disr.nodes.func_internal, 
                                             ci_network_disr.nodes.func_tot)]
        
    return ci_network_disr

def comb_imps_to_network(ci_network, df_res):
    """
    analogous function to impact_to_network, but for combined result datafrane
    from multihazard combined impacts
    """
    ci_network_disr = deepcopy(ci_network)
    # manually assign to network - roads
    ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='road',
                                  'func_internal'] = [np.min(
        [func_state, func_internal]) for func_state, func_internal in zip(
        df_res.loc[df_res.ci_type=='road','func_internal'], 
        ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='road', 'func_internal'])]
    ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='road',
                              'imp_dir'] = df_res.loc[df_res.ci_type=='road','imp_dir'].values
    # manually assign to network - power lines edges
    ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='power_line',
                                  'func_internal'] = [np.min(
        [func_state, func_internal]) for func_state, func_internal in zip(
        df_res.loc[(df_res.ci_type=='power_line')&(df_res.geometry.type=='LineString'),'func_internal'], 
        ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='power_line', 'func_internal'])]
    ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='power_line',
                                  'imp_dir'] = df_res.loc[(df_res.
                                                           ci_type=='power_line')&
                                                          (df_res.geometry.type=='LineString'),'imp_dir']
    # manually assign to network - power lines nodes
    ci_network_disr.nodes.loc[ci_network_disr.nodes.ci_type=='power_line',
                                  'func_internal'] = [np.min(
        [func_state, func_internal]) for func_state, func_internal in zip(
        df_res.loc[(df_res.ci_type=='power_line')&(df_res.geometry.type=='Point'),'func_internal'], 
        ci_network_disr.nodes.loc[ci_network_disr.nodes.ci_type=='power_line', 'func_internal'])]
    ci_network_disr.nodes.loc[ci_network_disr.nodes.ci_type=='power_line',
                                  'imp_dir'] = df_res.loc[(df_res.
                                                           ci_type=='power_line')&
                                                          (df_res.geometry.type=='Point'),'imp_dir']
    # manually assign to network - cell towers, power plants, schools, hospitals
    for ci_type in ['celltower', 'power_plant', 'health', 'education']:
        ci_network_disr.nodes.loc[
            ci_network_disr.nodes.ci_type==ci_type,'func_internal'] = [np.min(
            [func_state, func_internal]) for func_state, func_internal in zip(
            df_res.loc[df_res.ci_type==ci_type,'func_internal'], 
            ci_network_disr.nodes.loc[ci_network_disr.nodes.ci_type==ci_type, 'func_internal'])]
        ci_network_disr.nodes.loc[
            ci_network_disr.nodes.ci_type==ci_type,
            'imp_dir'] = df_res.loc[df_res.ci_type==ci_type,'imp_dir']
    
    ci_network_disr.edges['func_tot'] = [np.min([func_internal, func_tot]) for 
                                          func_internal, func_tot in zip(
                                              ci_network_disr.edges.func_internal, 
                                              ci_network_disr.edges.func_tot)]
    ci_network_disr.nodes['func_tot'] = [np.min([func_internal, func_tot]) for 
                                         func_internal, func_tot in zip(
                                             ci_network_disr.nodes.func_internal, 
                                             ci_network_disr.nodes.func_tot)]
    return ci_network_disr

def load_friction_surf(PATH_FRICTION, cntry_shape):
    friction_surf = Hazard('FRIC').from_raster(
    PATH_FRICTION, geometry=[cntry_shape.convex_hull.buffer(0.1)])
    return friction_surf

# save selected results as feather gdf
def save_resultdf(ci_graph_disr, path_save, event_name):
    ci_network_disr = ci_graph_disr.return_network()
    vars_to_keep_edges = ['ci_type', 'func_internal', 'func_tot', 'imp_dir','geometry']
    vars_to_keep_nodes = vars_to_keep_edges.copy() 
    vars_to_keep_nodes.extend([colname for colname in ci_network_disr.nodes.columns  if 'actual_supply_' in colname])
    vars_to_keep_nodes.extend(['counts'])
        
    df_res = ci_network_disr.nodes[ci_network_disr.nodes.ci_type=='people'][vars_to_keep_nodes]
    for ci_type in ['health', 'education', 'celltower', 'power_plant', 'power_line']:
        df_res = df_res.append(ci_network_disr.nodes[ci_network_disr.nodes.ci_type==ci_type]
                                   [vars_to_keep_nodes])
    for ci_type in ['power_line', 'road']:
        df_res = df_res.append(ci_network_disr.edges[ci_network_disr.edges.ci_type==ci_type]
                                   [vars_to_keep_edges])
    df_res.to_feather(path_save+f'cascade_results_{event_name}')


def wrapper_impacts_cascades_saving(haz, ci_network, exp_list, df_dependencies, friction_surf, res_orig=500, samples=100):
    """
    Parameters
    ----------
    haz: Hazard
    ci_network : Network
    exp_list : [exp_health, exp_educ, exp_celltowers, exp_pplant, exp_pline_n, exp_pline_e, exp_ptower, exp_road]
    df_dependencies : pd.DataFrame
    friction_surf : Hazard
    
    Returns
    -------
    imp_dict : Dict
        Basic service impacts compared to baseline
    """
    # IMPACT CALCS
    ci_network_disr = deepcopy(ci_network)
    for exp in exp_list:
        imp = calc_point_impacts(haz, exp, impf_set)
        if (haz_type in ['TC', 'EQ']) & (exp.tag in ['power_line_e', 'power_tower']):
            imp = binary_impact_from_prob(imp, samples=samples, seed=47)
            imp = binary_to_origres(imp, res_orig)
        if exp.tag in ['road','power_line_e', 'power_tower']:
            imp = u_lp.impact_pnt_agg(imp, exp.gdf, u_lp.AggMethod.SUM)
        ci_network_disr = impacts_to_network(imp, exp.tag, ci_network_disr)

    # IMPACT CASCADES
    ci_graph_disr = Graph(ci_network_disr, directed=True)
    ci_graph_disr.cascade(df_dependencies, p_source='power_plant', p_sink='power_line', 
                              source_var='el_generation', demand_var='el_consumption',
                              preselect=False, initial=False, friction_surf=friction_surf, 
                              dur_thresh=60)

    # SAVE RESULTS
    save_resultdf(ci_graph_disr, path_save, haz.event_name[0])
    # CALC IMPACTSTATS
    ci_network_disr = ci_graph_disr.return_network()
    imp_dict = nwu.disaster_impact_allservices_df(ci_network.nodes, ci_network_disr.nodes, 
                                                      services=['power', 'healthcare', 'education', 
                                                                'telecom', 'mobility'])
    imp_dict['people'] = sum(ci_network_disr.nodes[ci_network_disr.nodes.ci_type=='people'].imp_dir)
    return imp_dict
     

def combine_result_dfs(df_res1, df_res2):
    """
    Combine direct impacts, and assign combined internal functionality
    """
    df_res = deepcopy(df_res_tc)
    # simply sum direct impacts
    df_res['imp_dir'] = df_res_tc['imp_dir']+df_res_fl['imp_dir']
    # allocate failure thresholds
    df_res['func_thresh'] = 0
    for ci_type in ['celltower', 'power_plant', 'health', 'education', 'road']:
        df_res.loc[df_res.ci_type==ci_type, 'func_thresh'] = getattr(ImpactThresh(), ci_type)
    # power lines edges
    df_res.loc[(df_res.ci_type=='power_line')&
               (df_res.geometry.type=='LineString'), 'func_thresh'] = getattr(ImpactThresh(), 'power_line_e')
    # power lines nodes
    df_res.loc[(df_res.ci_type=='power_line')&
               (df_res.geometry.type=='Point'), 'func_thresh'] = getattr(ImpactThresh(), 'power_line_n')
    #assign internal functional state
    df_res['func_internal'] = np.array(list(map(int, df_res.imp_dir.values<=df_res.func_thresh.values)))
    return df_res


# =============================================================================
# Execution
# =============================================================================  
if __name__ == '__main__':  
    cntry = 'Mozambique'
    haz_type = sys.argv[1]
    iso3 = u_coord.country_to_iso(cntry)
    path_root = '/cluster/work/climate/evelynm/nw_outputs'
    path_edges  = f'{path_root}/{iso3}/moz_project/cis_nw_edges'
    path_nodes = f'{path_root}/{iso3}/moz_project/cis_nw_nodes'
    path_save = f'{path_root}/{iso3}/moz_project/'
    path_deps = f'{path_root}/{iso3}/moz_project/dependency_table_{iso3}.csv'
    path_fl_file = '/cluster/work/climate/evelynm/nw_outputs/MOZ/floodmaks_idai_margherita.tif'
    PATH_FRICTION = '/cluster/work/climate/evelynm/nw_inputs/friction/202001_Global_Walking_Only_Friction_Surface_2019.tif'
        
   
    # LOAD PRE_COMPUTED FILES
    ci_network = Network(edges=gpd.read_feather(path_edges), 
                         nodes=gpd.read_feather(path_nodes))
    df_dependencies = pd.read_csv(path_deps)
    
    __, cntry_shape = u_coord.get_admin1_info([cntry])
    cntry_shape = shapely.ops.unary_union([shp for shp in cntry_shape[iso3]])
    
    friction_surf = load_friction_surf(PATH_FRICTION, cntry_shape)
     
    
    # LOAD HAZARD FILES
    if haz_type=='FL':
        haz = load_flood_idai(path_fl_file)
        
    elif haz_type=='TC':
        haz = load_wind_idai(cntry_shape)
   
    if haz_type in ['FL', 'TC']:
    # MAKE EXPOSURES OR LOAD
        if not os.path.isdir(path_save+'exposures'):
            os.mkdir(path_save+'exposures')
            # point exposures 
            exp_health = exposure_from_nodes(gdf_from_network(ci_network.nodes, 'health'))
            exp_educ = exposure_from_nodes(gdf_from_network(ci_network.nodes, 'education'))
            exp_celltowers = exposure_from_nodes(gdf_from_network(ci_network.nodes, 'celltower'))
            exp_pplant = exposure_from_nodes(gdf_from_network(ci_network.nodes, 'power_plant'))
            exp_pline_n = exposure_from_nodes(gdf_from_network(ci_network.nodes, 'power_line'), tag='power_line_n')
            gdf_ppl = gdf_from_network(ci_network.nodes, 'people')
            exp_people = exposure_from_nodes(gdf_ppl, value=gdf_ppl.counts)
        
            # line exposures    
            res_orig = 500
            disagg_val_pline = res_orig
            disagg_val_road = res_orig # damage fraction on y-axis
            
            exp_pline_e = exposure_from_edges(gdf_from_network(ci_network.edges, 'power_line'),
                                              res=res_orig, disagg_val=disagg_val_pline, tag='power_line_e')
            exp_ptower = exposure_from_edges(gdf_from_network(ci_network.edges, 'power_line'),
                                              res=res_orig, disagg_val=disagg_val_pline, tag='power_tower')
            exp_road = exposure_from_edges(gdf_from_network(ci_network.edges, 'road'),
                                           res=res_orig, disagg_val=disagg_val_road)
            for exp in [exp_health, exp_educ, exp_celltowers, exp_pplant, 
                        exp_pline_n, exp_pline_e, exp_ptower, exp_road, exp_people]:
                exp.write_hdf5(path_save+f'exposures/exp_{exp.tag}')
    
        else:
            exp_health = Exposures.from_hdf5(path_save+'exposures/exp_health')
            exp_educ =  Exposures.from_hdf5(path_save+'exposures/exp_education')
            exp_celltowers =  Exposures.from_hdf5(path_save+'exposures/exp_celltower')
            exp_pplant =  Exposures.from_hdf5(path_save+'exposures/exp_power_plant')
            exp_pline_n =  Exposures.from_hdf5(path_save+'exposures/exp_power_line_n')
            exp_people =  Exposures.from_hdf5(path_save+'exposures/exp_people')
            exp_pline_e = Exposures.from_hdf5(path_save+'exposures/exp_power_line_e')
            exp_ptower = Exposures.from_hdf5(path_save+'exposures/exp_power_tower')
            exp_road = Exposures.from_hdf5(path_save+'exposures/exp_road')
        

        exp_pline_e.gdf['value'] = 500 if haz_type in ['FL', 'RF'] else 1 # damage fraction on y-axis for FL, failure prob on y-axis for EQ and TC
        exp_ptower.gdf['value'] = 500 if haz_type in ['FL', 'RF'] else 1 
        exp_list = [exp_health, exp_educ, exp_celltowers, exp_pplant, 
                    exp_pline_n, exp_pline_e, exp_ptower, exp_road, exp_people]
       
        for exp in exp_list:
                assign_impfs(exp, haz_type)
    
        impf_set = make_impfset(ImpfClassDict()[haz_type])
        
        # CASCADE
        res_orig = 500
        samples = 1000
        imp_dict = wrapper_impacts_cascades_saving(haz, ci_network, exp_list, df_dependencies, friction_surf, res_orig, samples)

    elif haz_type=='TCFL':
        # combine direct flood & wind impacts from previous runs
        df_res_tc = gpd.read_feather(path_save+'cascade_results_2019063S18038')
        df_res_fl = gpd.read_feather(path_save+'cascade_results_Flood_Idai')
        df_res = combine_result_dfs(df_res_tc, df_res_fl)
    
        ci_network_disr = comb_imps_to_network(ci_network, df_res)
        # IMPACT CASCADES
        ci_graph_disr = Graph(ci_network_disr, directed=True)
        ci_graph_disr.cascade(df_dependencies, p_source='power_plant', p_sink='power_line', 
                              source_var='el_generation', demand_var='el_consumption',
                              preselect=False, initial=False, friction_surf=friction_surf, 
                              dur_thresh=60)

        # SAVE RESULTS
        save_resultdf(ci_graph_disr, path_save, 'TCFL_base')
        # CALC IMPACTSTATS
        ci_network_disr = ci_graph_disr.return_network()
        imp_dict = nwu.disaster_impact_allservices_df(ci_network.nodes, ci_network_disr.nodes, 
                                                      services=['power', 'healthcare', 'education', 
                                                                'telecom', 'mobility'])
        imp_dict['people'] = sum(ci_network_disr.nodes[ci_network_disr.nodes.ci_type=='people'].imp_dir)
      
    with open(path_save+f'service_stats_idai_{haz_type}_base.pkl', 'wb') as f:
        pickle.dump(imp_dict, f)         
    
