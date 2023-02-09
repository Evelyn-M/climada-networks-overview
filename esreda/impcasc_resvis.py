#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:55:57 2022

@author: evelynm
---
Compound event calculation for flood and TC wind hazards (ESREDA seminar)
"""

import geopandas as gpd
from copy import deepcopy, copy
import numpy as np
import pandas as pd
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.engine import Impact
from climada.hazard.base import Hazard
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.centroids import Centroids
from climada.hazard.trop_cyclone import TropCyclone
from climada.util import lines_polys_handler as u_lp

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import Graph
import climada_petals.engine.networks.nw_utils as nwu

import numpy as np
import mplotutils as mpu

from matplotlib.colors import ListedColormap

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


def calc_ci_impacts(hazard, exposures, ImpfClass, res_pl=500):
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
        impfset = ImpactFuncSet()
        impfset.append(getattr(ImpfClass,ci_type))
        
        if ci_type =='road':
            imp.calc(exp, impfset, hazard, save_mat=True)
            imp = u_lp.impact_pnt_agg(imp,  exp.gdf, u_lp.AggMethod.SUM)
        elif ci_type == 'power_line':
            # first convert failure probs for power line to binary failed / not failed
            imp.calc(exp, impfset, hazard, save_mat=True)
            imp.imp_mat.data = binary_impact_from_prob(imp.imp_mat.data)
            # then to absolute length in metres failed
            imp.imp_mat.data = imp.imp_mat.data*res_pl    
            imp = u_lp.impact_pnt_agg(imp,  exp.gdf, u_lp.AggMethod.SUM)
        else:
            imp.calc(exp, impfset, hazard)
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


def calc_cascade(imp_list, exp_list, df_dependencies, graph_base, path_save,
                 name_save):    
    
    ci_network = graph_base.return_network()
    ci_network.nodes = ci_network.nodes.drop('name', axis=1)
    graph_disr = impacts_to_graph(imp_list, exp_list, ci_network)
    
    # cascade calculation
    # hard-coded kwargs for cascade(). change in future                     
    graph_disr.cascade(df_dependencies, p_source='power_plant', p_sink='power_line', 
                      source_var='el_generation', demand_var='el_consumption', #'el_generation' 'el_consumption'
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
    df_res.to_feather(path_save+f'cascade_results_{name_save}')
    
    del ci_network
    del df_res
    del ci_net

    return nwu.disaster_impact_allservices(
        graph_base, graph_disr, services =['power', 'healthcare', 
                                           'education', 'telecom', 
                                           'mobility'])

# =============================================================================
# Results Processing
# =============================================================================
           
def get_cascstate(gdf):
    casc_state = [0]* len(gdf)
    for i in range(len(gdf)):
        if ((gdf.func_tot.iloc[i]==0) & (gdf.func_internal.iloc[i]==0)):
            casc_state[i] = 1
        elif ((gdf.func_tot.iloc[i] ==0) & (gdf.func_internal.iloc[i] >0)):
            casc_state[i] = 2
    return casc_state                                   

def get_accessstates(gdf, node_gdf_orig):
    """
    1 - accessible, 0 - inaccessible from beginning on, -1 - disrupted due to 
    disaster
    
    Changes gdf entries.
    """
    
    services = [colname for colname in gdf.columns if 'actual_supply_' in colname] 
    for service in services:
        serv_level = gdf[gdf.ci_type=='people'][service].values
        serv_level_orig = node_gdf_orig[node_gdf_orig.ci_type=='people'][service].values
        serv_level[(serv_level==0.) & (serv_level_orig==1.)]= -1.
        gdf.loc[gdf.ci_type=='people', service] = serv_level
    return gdf

class InfraColorMaps:
    def __init__(self):
        self.service_col_dict = {-1. : '#FF5733', 0. : 'grey', 1. : 'green'}
        self.service_col_map = ListedColormap(['#FF5733', 'grey', 'green'])
        self.servicecum_col_dict = {-9. : '#581845', -8. : '#581845',
                                -7. : '#581845', -6. : '#581845',  
                                -5. : '#581845', -4. : '#581845',
                                -3. : '#900C3F', -2. : '#C70039', 
                                -1. : '#FF5733', 0. : 'grey', 
                                 1. : 'green'}
        self.servicecum_col_map = ListedColormap(['#581845',  '#581845',
                                '#581845',  '#581845',  '#581845', '#581845',
                               '#900C3F', '#C70039',  '#FF5733', 'grey', 'green'])
        self.casc_col_dict = {0. : 'blue', 1. : 'magenta', 2. : 'yellow'}
        self.casc_col_map = ListedColormap(['blue','magenta','yellow'])

def _get_extent(gdf):
    buffer_deg = 0.1
    sub_gdf = gdf[gdf.geometry.type == 'Point']
    return (min(sub_gdf.geometry.x)-buffer_deg, max(sub_gdf.geometry.x)+buffer_deg,
                     min(sub_gdf.geometry.y)-buffer_deg, max(sub_gdf.geometry.y)+buffer_deg)

def service_impact_plot(gdf, save_path=None, event_name=None):
    """
    per basic service, people cluster with and without access to that service
    """
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    services = [colname for colname in gdf.columns if 'actual_supply_' in colname] 
    
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    f, axes = plt.subplots(3, int(np.ceil(len(services)/3)), 
                           subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=(16,16*1.258))

    for service, ax in zip(services, axes.flatten()[:len(services)]):
        ax.set_extent((88.0844222351, 92.6727209818, 20.670883287, 26.4465255803), ccrs.PlateCarree()) #_get_extent(gdf)
        ax.add_feature(border, facecolor='none', edgecolor='0.5')
        h_serv = ax.scatter(gdf[gdf.ci_type=='people'].geometry.x, 
                            gdf[gdf.ci_type=='people'].geometry.y, 
                            c=gdf[gdf.ci_type=='people'][service],
                            cmap=InfraColorMaps().service_col_map, 
                            transform=ccrs.PlateCarree(), 
                            vmin=-1., vmax=1., s=0.1)
        ax.set_title(f'Disruptions in access to {service[14:-7]}', 
                     weight='bold', fontsize=17)         
    
    if len(services)%2>0:
        f.delaxes(axes.flatten()[-1])
            
    cbar = mpu.colorbar(
        h_serv, axes.flatten()[-2], size=0.05, pad=0.05, orientation='horizontal')
    cbar.set_ticks([-0.66, 0., .66])
    cbar.set_ticklabels(['Disr.', 'Inavail.', 'Avail.'])
    
    #f.suptitle(f'Service Disruptions from event {event_name}', weight='bold', fontsize=24)
    #f.tight_layout()
    f.subplots_adjust(bottom=0.05, top=0.95)  
                                       
    if save_path:
        plt.savefig(f'{save_path}'+f'service_disruptions_{event_name}.pdf', 
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
        
def _get_roadmask(gdf):
    """
    don't plot road access links which were generated during network construction
    in road plots
    """
    access_links = 0
    for ci_type in ['health', 'education', 'people']:
        access_links+=sum(gdf.ci_type==ci_type)
    return sum(gdf.ci_type=='road')-int(access_links*2)

def infra_plot(gdf, save_path=None, ):
    """ all infrastructures in one plot"""
    
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    ci_types = set(gdf.ci_type).difference({'people'})
    f, axes = plt.subplots(3, int(np.ceil(len(ci_types)/3)), 
                          subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=(16,16*1.258))

    for ci_type, ax in zip(ci_types, axes.flatten()):
        ax.set_extent( (88.0844222351, 92.6727209818, 20.670883287,  26.4465255803), ccrs.PlateCarree()) #_get_extent(gdf)
        ax.add_feature(border, facecolor='none', edgecolor='0.5')
        
        if ci_type=='road': 
            gdf[gdf.ci_type==ci_type][:_get_roadmask(gdf)
                    ].plot(ax=ax, markersize=1, linewidth=0.5, 
                           transform=ccrs.PlateCarree(), color='k')
        else:
            gdf[gdf.ci_type==ci_type].plot(
                ax=ax, markersize=1, linewidth=0.5, 
                transform=ccrs.PlateCarree(), color='k')
            
        ax.set_title(f'{ci_type}', weight='bold', fontsize=17)
    f.suptitle('Infrastructures', weight='bold', fontsize=24)
    f.tight_layout()
    f.subplots_adjust(bottom=0.05, top=0.95)   
    
    if save_path:
        plt.savefig(f'{save_path}'+'infrastructures.pdf', 
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    
# =============================================================================
# Execution
# =============================================================================
  
# Case 1: consecutive hazrds TC Giri 2010 (wind) 2010280N17085 & flood DFO_3713 2 weeks later
# Case 2: sub-hazards TC Wind and Storm surge / Flood TC Sidr (2007)
# 	2007314N10093 and DFO_3226
  
iso3 = 'BGD'
path_root = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/climada-networks-overview/esreda/BGD/'
path_edges  = f'{path_root}cis_nw_edges'
path_nodes = f'{path_root}cis_nw_nodes'
path_deps = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/dependency_tables/dependencies_default.csv'

# load necessary files
ci_network = Network(edges=gpd.read_feather(path_edges), 
                     nodes=gpd.read_feather(path_nodes))
ci_network.nodes = ci_network.nodes.drop('name', axis=1)
graph_base = Graph(ci_network, directed=True)
df_dependencies = pd.read_csv(path_deps)

# make exposures 
exp_list = []
for ci_type in ['power_plant', 'celltower', 'health', 'education']:
    exp_list.append(exposure_from_network_nodes(ci_network, ci_type, ImpFuncsCIFlood()))
for ci_type in ['power_line', 'road']:
    exp_list.append(exposure_from_network_edges(ci_network, ci_type, ImpFuncsCIFlood(),res=500))    
for exp in exp_list:
    exp.gdf[f'impf_{ImpFuncsCIWind().tag}'] = getattr(ImpFuncsCIWind(), ci_type).id

# load hazards
path_haz_fl =path_root+f'flood_{iso3}.hdf5'
haz_fl = Hazard('FL').from_hdf5(path_haz_fl)
haz_fl = haz_fl.select(event_names=['DFO_3226'])
#haz_fl.plot_intensity(event=[70],smooth=False) #82 for #1

tr_track = TCTracks.from_ibtracs_netcdf(storm_id='2007314N10093') 
tr_track.equal_timestep()
cent_bgd = Centroids().from_pnt_bounds(
    (88.0844222351, 20.670883287, 92.6727209818, 26.4465255803), res=0.05) 

haz_tc = TropCyclone.from_tracks(tr_track, centroids=cent_bgd)
# cent_bgd.check()
# cent_bgd.plot()
# ax = tr_giri.plot()
# ax.set_title('TC Giri 2010')
# haz_tc.plot_intensity(0)  
 
# calculate single event impacts and failure cascades
for hazard, ImpfClass in zip([haz_fl, haz_tc], [ImpFuncsCIFlood(), ImpFuncsCIWind()]):
    imp_list = calc_ci_impacts(hazard, exp_list, ImpfClass, res_pl=500)
    event_name = hazard.event_name[0]
    # failure cascade
    service_dict = calc_cascade(imp_list, exp_list, df_dependencies, graph_base, path_root,
                     event_name)
    # save stats
    with open(path_root+f'service_stats_{event_name}.pkl', 'wb') as f:
         pickle.dump(service_dict, f) 
    
# calculate compound event impact and faulure cascade
imp_list_fl = calc_ci_impacts(haz_fl, exp_list, ImpFuncsCIFlood(), res_pl=500)
imp_list_tc = calc_ci_impacts(haz_tc, exp_list, ImpFuncsCIWind(), res_pl=500)
imp_list = []
for imp_fl, imp_tc in zip(imp_list_fl, imp_list_tc):
    imp = copy(imp_fl)
    imp.eai_exp = imp_fl.eai_exp + imp_tc.eai_exp
    imp_list.append(imp)
event_name = 'tcfl'
service_dict = calc_cascade(imp_list, exp_list, df_dependencies, graph_base, path_root,
                 event_name)
with open(path_root+f'service_stats_{event_name}.pkl', 'wb') as f:
     pickle.dump(service_dict, f)
    

# load saved stats
with open(path_root+'event_sidr_subhazards/service_stats_tcfl_2007.pkl', 'rb') as f:
     service_dict = pickle.load(f)  
service_dict[event_name]
    
# compare individual sums vs. compound
df_cascres_fl = gpd.read_feather(path_root+f'event_sidr_subhazards/cascade_results_DFO_3226') #3713
df_cascres_fl = get_accessstates(df_cascres_fl, ci_network.nodes)
df_cascres_tc = gpd.read_feather(path_root+f'event_sidr_subhazards/cascade_results_2007314N10093') #2010280N17085
df_cascres_tc = get_accessstates(df_cascres_tc, ci_network.nodes)
df_cascres_comp = gpd.read_feather(path_root+f'event_sidr_subhazards/cascade_results_tcfl_2007')
df_cascres_comp = get_accessstates(df_cascres_comp, ci_network.nodes)
df_cascres_sum = gpd.GeoDataFrame(geometry=df_cascres_fl[df_cascres_fl.ci_type=='people'].geometry)
df_cascres_sum = get_accessstates(df_cascres_sum, ci_network.nodes)

for column in df_cascres_fl.columns[5:]:
    df_cascres_sum[column] = np.minimum(df_cascres_fl[df_cascres_fl.ci_type=='people'][column], 
           df_cascres_tc[df_cascres_tc.ci_type=='people'][column])
service_dict = {}
for service in df_cascres_sum.columns[1:]:
    service_dict[service[14:-7]] = df_cascres_sum[df_cascres_sum[service]==-1].counts.sum()

service_impact_plot(df_cascres_comp, save_path=path_root, event_name='TCFL_sidr')

infra_plot(df_cascres_comp)