#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script handling all the CI data preprocessing for TC Michael
in Florida, Alabama, Georgia

@author: evelynm
"""


import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from cartopy.io import shapereader

# on climada_petals branch feature/networks until merged!!!
from climada_petals.engine.networks.nw_preps import (PowerlinePreprocess,
                                                     RoadPreprocess)
                                                     #UtilFunctionalData)
from climada_petals.engine.networks.base import Network
from climada_petals.engine.networks.nw_calcs import Graph
from climada_petals.entity.exposures.black_marble import country_iso_geom
from climada_petals.entity.exposures.openstreetmap.osm_dataloader import OSMFileQuery

# on climada_python branch feature/lines_polygons_exp until merged into develop!!
from climada.util.constants import ONE_LAT_KM
from climada.entity.exposures.base import Exposures
from climada.hazard import TCTracks, Centroids, TropCyclone


# =============================================================================
# Constants
# =============================================================================
path_input_data =  '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data'
path_output_data = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/output_multi_cis/FLALGA'

path_plines = path_input_data+'/power_global/Electric_Power_Transmission_Lines_US.geojson'
path_pp = path_input_data+'/power_global/Power_Plants_US.geojson'
path_ct = path_input_data+'/cell_towers/USA_Cellular_Towers1.geojson'
path_worldpop_fl = path_input_data+'/population/50_US_states_1km_2020_UNadj/US-FL_ppp_2020_1km_UNadj.tif'
path_worldpop_al = path_input_data+'/population/50_US_states_1km_2020_UNadj/US-AL_ppp_2020_1km_UNadj.tif'
path_worldpop_ga = path_input_data+'/population/50_US_states_1km_2020_UNadj/US-GA_ppp_2020_1km_UNadj.tif'
path_roads = path_input_data+'/roads/Transportation_US.geojson'
path_osm_al = path_input_data+'/osm_countries/alabama-latest.osm.pbf'
path_osm_fl = path_input_data+'/osm_countries/florida-latest.osm.pbf'
path_osm_ga = path_input_data+'/osm_countries/georgia-latest.osm.pbf'
path_schools = path_input_data+'/education/Public_Schools_USA/PublicSchools.shp'
path_hospitals = path_input_data+'/health/Hospitals_USA/Hospitals.shp'
path_wastewater = path_input_data+'/water/Wastewater_Treatment_Plants_USA.geojson'
path_el_consump_usa = path_input_data+'/power_global/Electricity consumption by sector - United States.csv'
path_el_imp_exp_usa = path_input_data+'/power_global/Electricity imports vs. exports - United States.csv'
path_deps = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/dependencies_FLALGA.csv'

save_path_cis = path_input_data+'/multi_cis_extracts/FLALGA'
save_path_graphs = path_output_data+'/graphs'
save_path_exps = path_output_data+'/exps'
save_path_figs = path_output_data+'/figures'

# =============================================================================
# CI Data Curation
# =============================================================================

# get shape file for Florida, Alabama, Georgia
shp_file = shapereader.natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
shp_file = shapereader.Reader(shp_file)
prov_names = {'United States Of America': ['Florida', 'Alabama','Georgia']}
polygon_usa, polygons_fl_ge_al = country_iso_geom(prov_names, shp_file)
shape_flalga = shapely.ops.unary_union(polygons_fl_ge_al['USA'])
buffer_km = 20
shape_flalga_buffered = shape_flalga.buffer(buffer_km/ONE_LAT_KM)

# PEOPLE 
# from worldpop, reproject 1km2 to 10x10km grid:
gdf_people_fl = UtilFunctionalData().load_resampled_raster(path_worldpop_fl, upscale_factor=1/10)
gdf_people_al = UtilFunctionalData().load_resampled_raster(path_worldpop_al, upscale_factor=1/10)
gdf_people_ga = UtilFunctionalData().load_resampled_raster(path_worldpop_ga, upscale_factor=1/10)
gdf_people = pd.concat([gdf_people_fl, gdf_people_al, gdf_people_ga],ignore_index=True)
gdf_people.to_file(save_path_cis+'/people_flalga.shp')


# POWER LINES
# from HIFLD
gdf_powerlines = gpd.read_file(path_plines, mask=shape_flalga_buffered)
gdf_powerlines['osm_id'] = 'n/a'
gdf_powerlines = gdf_powerlines[['osm_id','TYPE','geometry','VOLT_CLASS', 'VOLTAGE']]
gdf_powerlines.to_file(save_path_cis+'/powerlines_flalga.shp')

# POWER PLANTS
# from HIFLD
gdf_pplants = gpd.read_file(path_pp, mask=shape_flalga_buffered) 
gdf_pplants = gdf_pplants[['NAME','geometry','OPER_CAP', 'NET_GEN']]
gdf_pplants[gdf_pplants.NET_GEN==-999999] = np.nan
gdf_pplants = gdf_pplants[gdf_pplants.geometry!=None]
PER_CAP_ECONSUMP = 13787828*277.778/332475723
to_assign = PER_CAP_ECONSUMP*gdf_people.counts.sum()-gdf_pplants.NET_GEN.sum()
gdf_pplants = gdf_pplants.append(
    gpd.GeoSeries({'NAME':'imp_exp_balance', 'NET_GEN':to_assign,
                   'geometry':shapely.geometry.Point([min(gdf_pplants.geometry.x), min(gdf_pplants.geometry.y)])}),
                   ignore_index=True)
gdf_pplants.to_file(save_path_cis+'/powerplants_flalga.shp')

# HEALTH FACILITIES
gdf_health = gpd.read_file(path_hospitals).to_crs(epsg=4326)
gdf_health = gdf_health[gdf_health.within(shape_flalga_buffered)]
gdf_health = gdf_health[['NAME','geometry','BEDS']]
gdf_health.to_file(save_path_cis+'/healthfacilities_flalga.shp')

# EDUC. FACILITIES
# from HIFLD
gdf_educ = gpd.read_file(path_schools).to_crs(epsg=4326)
gdf_educ = gdf_educ[gdf_educ.within(shape_flalga_buffered)]
gdf_educ = gdf_educ[['NAME','geometry', 'ENROLLMENT']]
gdf_educ.to_file(save_path_cis+'/educationfacilities_flalga.shp')

# TELECOM
# from HIFLD
gdf_telecom = gpd.read_file(path_ct, mask=shape_flalga_buffered)
gdf_telecom = gdf_telecom[['geometry', 'StrucType']]
gdf_telecom.to_file(save_path_cis+'/celltowers_flalga.shp')


# ROADS
# from HIFLD - not recommendeable for processing algorithm

#from OSM
FlFileQuery = OSMFileQuery(path_osm_fl)
AlFileQuery = OSMFileQuery(path_osm_al)
GaFileQuery = OSMFileQuery(path_osm_ga)

# all roads
gdf_roads_osm = FlFileQuery.retrieve_cis('road') 
gdf_roads_osm = gdf_roads_osm.append(AlFileQuery.retrieve_cis('road'))
gdf_roads_osm = gdf_roads_osm.append(GaFileQuery.retrieve_cis('road'))
gdf_roads_osm = gdf_roads_osm[gdf_roads_osm.geometry.type=='LineString']
gdf_roads_osm.to_file(save_path_cis+'/roads_osm_flalga.shp')
gdf_roads_osm = gpd.read_file(save_path_cis+'/roads_osm_flalga.shp') 

# main roads
gdf_mainroads_osm = FlFileQuery.retrieve_cis('main_road') 
gdf_mainroads_osm = gdf_mainroads_osm.append(AlFileQuery.retrieve_cis('main_road'))
gdf_mainroads_osm = gdf_mainroads_osm.append(GaFileQuery.retrieve_cis('main_road'))
gdf_mainroads_osm = gdf_mainroads_osm[gdf_mainroads_osm.geometry.type=='LineString']
gdf_mainroads_osm.to_file(save_path_cis+'/mainroads_osm_flalga.shp')

# WASTEWATER (FROM HIFLD)
gdf_wastewater = gpd.read_file(path_wastewater, mask=shape_flalga_buffered)
gdf_wastewater = gdf_wastewater[['geometry', 'CWP_NAME']]
gdf_wastewater.to_file(save_path_cis+'/wastewater_flalga.shp')

# =============================================================================
# Networks Preprocessing
# =============================================================================

# POWER LINES
gdf_power_edges, gdf_power_nodes = PowerlinePreprocess().preprocess(
    gdf_edges=gdf_powerlines)
power_network = Network(gdf_power_edges, gdf_power_nodes)
power_network.edges = power_network.edges[['from_id', 'to_id', 'osm_id','geometry','distance', 'ci_type']]
power_graph = Graph(power_network, directed=False)
power_graph.link_clusters()
power_network = Network().from_graphs([power_graph.graph.as_directed()])
power_network.nodes = power_network.nodes.drop('name', axis=1)
power_network.nodes.to_file(save_path_cis+'/powerlines_flalga_processedn.shp') 
power_network.edges.to_file(save_path_cis+'/powerlines_flalga_processede.shp')


# ROAD
gdf_road_edges, gdf_road_nodes = RoadPreprocess().preprocess(
    gdf_edges=gdf_mainroads_osm)
road_network = Network(gdf_road_edges, gdf_road_nodes)
# # easy workaround for doubling edges
road_graph = Graph(road_network, directed=False)
road_graph.link_clusters()
road_network = Network().from_graphs([road_graph.graph.as_directed()])
road_network.nodes = road_network.nodes.drop('name', axis=1)
road_network.edges.to_file(save_path_cis+'/mainroads_osm_flalga_processede.shp')
road_network.nodes.to_file(save_path_cis+'/mainroads_osm_flalga_processedn.shp')
# road_graph.graph.clusters().summary(): 
# 'Clustering with 51920 elements and 1 clusters'

# =============================================================================
# Exposure Preprocessing
# =============================================================================

tr_michael = TCTracks.from_ibtracs_netcdf(provider='usa', storm_id='2018280N18273') # Michael 2018
tr_michael.equal_timestep()
ax = tr_michael.plot()
ax.set_title('TC Michael 2018')

# construct centroids
cent_flalga = Centroids().from_pnt_bounds((-88, 24.5, -79.4, 35), res=0.1) 
cent_flalga.check()
cent_flalga.plot()

tc_michael = TropCyclone.from_tracks(tr_michael, centroids=cent_flalga)


# =============================================================================
# Exposure Preprocessing
# =============================================================================

# POWER LINES
exp_pl = Exposures()
exp_pl.set_from_lines(
    gpd.GeoDataFrame(cis_network.edges[cis_network.edges.ci_type=='power line'], 
                      crs='EPSG:4326'), m_per_point=500, disagg_values='cnst',
                      m_value=1)
exp_pl.gdf['impf_TC'] = 1
exp_pl.gdf.to_file(save_path_exps+'/exp_pl_gdf.shp')

# ROAD
exp_road = Exposures()
exp_road.set_from_lines(
    gpd.GeoDataFrame(cis_network.edges[cis_network.edges.ci_type=='road'], 
                      crs='EPSG:4326'), m_per_point=500, disagg_values='cnst',
                      m_value=1)
exp_road.gdf['impf_TC'] = 4
exp_road.gdf.to_file(save_path_exps+'/exp_road_gdf.shp')

# HEALTHCARE
exp_hc = Exposures()
exp_hc.gdf = cis_network.nodes[cis_network.nodes.ci_type=='health']
exp_hc.gdf['value'] = 1
exp_hc.gdf['impf_TC'] = 3
exp_hc.write_hdf5(save_path_exps+'/exp_hc.h5')


# EDUCATION
exp_educ = Exposures()
exp_educ.gdf = cis_network.nodes[cis_network.nodes.ci_type=='education']
exp_educ.gdf['value'] = 1
exp_educ.gdf['impf_TC'] = 2
exp_educ.write_hdf5(save_path_exps+'/exp_educ.h5')


# TELECOM
exp_tele = Exposures()
exp_tele.gdf = cis_network.nodes[cis_network.nodes.ci_type=='celltower']
exp_tele.gdf['value'] = 1
exp_tele.gdf['impf_TC'] = 5
exp_tele.write_hdf5(save_path_exps+'/exp_tele.h5')

# WASTEWATER
exp_wwater = Exposures()
exp_wwater.gdf = cis_network.nodes[cis_network.nodes.ci_type=='wastewater']
exp_wwater.gdf['value'] = 1
exp_wwater.gdf['impf_TC'] = 3
exp_wwater.write_hdf5(save_path_exps+'/exp_wwater.h5')
