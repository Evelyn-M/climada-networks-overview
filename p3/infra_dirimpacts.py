#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:04:56 2022

@author: evelynm

End-to-end pipeline to compute infrastructure damages from floods and TC winds
"""


import geopandas as gpd
import numpy as np
from pathlib import Path
import shapely
import os
from scipy.sparse import csr_matrix
from datetime import datetime

# on climada_petals branch feature/networks until merged!
from climada_petals.engine.networks import nw_utils as nwu
from climada_petals.entity.exposures.openstreetmap import osm_dataloader as osm
from climada_petals.util.constants import DICT_GEOFABRIK

# on climada_python develop branch
from climada.util import coordinates as u_coords
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.engine import Impact
from climada.hazard.base import Hazard
from climada.util import lines_polys_handler as u_lp
from climada.util.api_client import Client

# general paths & constants
PATH_DATA = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data/'
PATH_DATA_OSM = PATH_DATA +'osm_countries/' # path to search for osm.pbf files and to download to if not existing
PATH_DATA_HVMV = PATH_DATA +'power_global/hvmv_global.gpkg' # path of this file from gridfinder. needs to be downloaded.
PATH_DATA_PP = PATH_DATA +'power_global/global_power_plant_database.csv' # path of this file from WRI global power plant db. needs to be downloaded.
PATH_DATA_CT = PATH_DATA +'cell_towers/opencellid_global_1km_int.tif' # path of this file from worldbank open data gridded celltowers. needs to be downloaded.
PATH_DATA_POP = PATH_DATA + 'population/' # path to search for population files and to download to if not existing
PATH_SAVE = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/p3/'

# start and end date for api hazard query
START_STR = '01-01-2000'
END_STR = '31-12-2021'

cntry = 'cntryname'

# =============================================================================
# Load Infra Data
# =============================================================================

iso3 = u_coords.country_to_iso(cntry)
path_osm_cntry = PATH_DATA_OSM+DICT_GEOFABRIK[iso3][-1]+'-latest.osm.pbf'
path_worldpop_cntry = PATH_DATA_POP + f'{iso3.lower()}_ppp_2020_1km_Aggregated_UNadj.tif'
path_save_cntry = PATH_SAVE + f'{iso3}/'

if not os.path.isdir(path_save_cntry):
    os.mkdir(path_save_cntry)
    
__, cntry_shape = u_coords.get_admin1_info([cntry])
cntry_shape = shapely.ops.unary_union([shp for shp in cntry_shape[iso3]])
osm.OSMRaw().get_data_geofabrik(iso3, file_format='pbf', save_path=PATH_DATA_OSM)
CntryFileQuery = osm.OSMFileQuery(path_osm_cntry)

# POWER LINES
gdf_powerlines = gpd.read_file(PATH_DATA_HVMV, mask=cntry_shape)
gdf_powerlines['osm_id'] = 'n/a'
gdf_powerlines['ci_type'] = 'power_line' 
gdf_powerlines = gdf_powerlines[['osm_id', 'geometry', 'ci_type']]

# POWER PLANTS
# try WRI pp database, then OSM
gdf_pp_world = gpd.read_file(PATH_DATA_PP, crs='EPSG:4326')
gdf_pp = gdf_pp_world[gdf_pp_world.country==f'{iso3}'][
    ['estimated_generation_gwh_2017','latitude','longitude',
     'name']]
del gdf_pp_world
if not gdf_pp.empty:
    gdf_pp = gpd.GeoDataFrame(
        gdf_pp, geometry=gpd.points_from_xy(gdf_pp.longitude, gdf_pp.latitude))
else:
    gdf_pp = CntryFileQuery.retrieve_cis('power')
    if len(gdf_pp[gdf_pp.power=='plant'])>1:
        gdf_pp = gdf_pp[gdf_pp.power=='plant']
    else:
        # last 'resort': take generators frmom OSM
        gdf_pp = gdf_pp[gdf_pp.power=='generator']
    gdf_pp['geometry'] = gdf_pp.geometry.apply(lambda geom: geom.centroid)
    gdf_pp = gdf_pp[['name', 'power', 'geometry']]
gdf_pp['ci_type'] = 'power_plant'

# PEOPLE
nwu.get_worldpop_data(iso3, PATH_DATA_POP)
gdf_people = nwu.load_resampled_raster(path_worldpop_cntry, 1)
gdf_people = gdf_people[gdf_people.counts>=10].reset_index(drop=True)
gdf_people['ci_type'] = 'people'

# HEALTH FACILITIES
# from osm
gdf_health = CntryFileQuery.retrieve_cis('healthcare') 
gdf_health['geometry'] = gdf_health.geometry.apply(lambda geom: geom.centroid)
gdf_health = gdf_health[['name', 'geometry']]
gdf_health = gdf_health[gdf_health.geometry.within(cntry_shape)]
gdf_health['ci_type'] = 'health'


# EDUC. FACILITIES
# from osm
gdf_educ = CntryFileQuery.retrieve_cis('education')
gdf_educ['geometry'] = gdf_educ.geometry.apply(lambda geom: geom.centroid)
gdf_educ = gdf_educ[['name', 'geometry']]
gdf_educ = gdf_educ[gdf_educ.geometry.within(cntry_shape)]
gdf_educ['ci_type'] = 'education'


# TELECOM
# cells from rasterized opencellID (via WB)
path_ct_cntry = path_save_cntry+'celltowers.tif'
if not Path(path_ct_cntry).is_file():
    if cntry_shape.type=='Polygon':
        geo_mask = [cntry_shape]
    else:
        geo_mask = [mp for mp in cntry_shape]
    meta_ct, arr_ct = u_coords.read_raster(PATH_DATA_CT, src_crs={'epsg':'4326'},
                                           geometry=geo_mask)
    u_coords.write_raster(path_ct_cntry, arr_ct, meta_ct)
gdf_cells = nwu.load_resampled_raster(path_ct_cntry, 1/5)
gdf_cells['ci_type'] = 'celltower'


# ROADS
# from osm
gdf_roads = CntryFileQuery.retrieve_cis('main_road') 
gdf_roads = gdf_roads[['osm_id','highway', 'geometry']]
gdf_roads = gdf_roads[gdf_roads.within(cntry_shape)]
gdf_roads['ci_type'] = 'road'


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

# =============================================================================
# Impact calc funcs
# =============================================================================

def exposure_from_points(gdf, imp_class):
    exp_pnt = Exposures(gdf)
    exp_pnt.gdf[f'impf_{imp_class.tag}'] = getattr(imp_class, gdf['ci_type'].iloc[0]).id
    exp_pnt.gdf['value'] = 1
    exp_pnt.set_lat_lon()
    exp_pnt.check()
    return exp_pnt
      
def exposure_from_lines(gdf, imp_class, res, 
                                disagg_met=u_lp.DisaggMethod.FIX, disagg_val=None):
    exp_line = Exposures(gdf)
    if not disagg_val:
        disagg_val = res
    exp_pnt = u_lp.exp_geom_to_pnt(exp_line, res=res, to_meters=True, 
                                   disagg_met=disagg_met, disagg_val=disagg_val)  
    exp_pnt.gdf[f'impf_{imp_class.tag}'] = getattr(imp_class, gdf['ci_type'].iloc[0]).id
    exp_pnt.set_lat_lon()
    exp_pnt.check()
    
    return exp_pnt

def calc_ci_impacts(hazard, exposures, imp_class):
    """
    Parameters
    ----------
    hazard: single event
    exposures: list of exp objects
    imp_class: ImpFuncsCIFlood or ImpFuncsCIWind
    
    Returns
    -------
    list of impacts
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

def get_selected_floods_api(iso3, start=START_STR, end=END_STR):
    """ 
    once cloud to street hazards uploaded to data api.
    For now, get hdf5 files directly from folder on cluster
    """
    pass


# =============================================================================
# Direct Impact calculations
# =============================================================================
haz_type= 'specify haz_tag'

if haz_type=='FL':
    #hazards = get_selected_floods_api(iso3)
    path_haz=f'insert_path_floods_cluster/{iso3}/'+f'flood_{iso3}.hdf5'
    hazards = Hazard('FL').from_hdf5(path_haz)
    imp_class = ImpFuncsCIFlood()
    
elif haz_type=='TC':
    hazards = get_selected_tcs_api(iso3)
    imp_class = ImpFuncsCIWind()

exp_list = []
for gdf in [gdf_educ, gdf_health, gdf_cells, gdf_pp]:
    exp_list.append(exposure_from_points(gdf, imp_class))
for gdf in [gdf_powerlines, gdf_roads]:
    exp_list.append(exposure_from_lines(gdf, imp_class, res=500))

imp_list = calc_ci_impacts(hazards, exp_list, imp_class)