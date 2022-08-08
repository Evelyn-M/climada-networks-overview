#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:47:39 2022

@author: evelynm
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import mplotutils as mpu
import geopandas as gpd
import sys
import climada.util.coordinates as u_coords

import os
import glob

from matplotlib.colors import ListedColormap

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
                          figsize=(16,20*1.4142))

    for ci_type, ax in zip(ci_types, axes.flatten()):
        ax.set_extent(_get_extent(gdf), ccrs.PlateCarree())
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
        
    
def infra_func_plot(gdf, save_path=None, event_name=None):
    """ 
    per infrastructure, a plot of functional, destroyed-dysfunctional, 
    cascaded-dysfunctional
    """
    
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    ci_types = set(gdf.ci_type).difference({'people'})
    f, axes = plt.subplots(3, int(np.ceil(len(ci_types)/3)), 
                           subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=(16,20*1.4142))

    for ci_type, ax in zip(ci_types, axes.flatten()):
        ax.set_extent(_get_extent(gdf), ccrs.PlateCarree())
        ax.add_feature(border, facecolor='none', edgecolor='0.5')
        if ci_type=='road':
            gdf[gdf.ci_type==ci_type][:_get_roadmask(gdf)
                    ].plot(ax=ax, markersize=1, linewidth=0.5, 
                           transform=ccrs.PlateCarree(), 
                           color=gdf[gdf.ci_type==ci_type].casc_state.map(
                               InfraColorMaps().casc_col_dict).values.tolist())
        elif ci_type=='power_line':
            gdf[gdf.ci_type==ci_type].plot(ax=ax, markersize=1, linewidth=0.5, 
                           transform=ccrs.PlateCarree(), 
                           color=gdf[gdf.ci_type==ci_type].casc_state.map(
                               InfraColorMaps().casc_col_dict).values.tolist())
        else:
            h_casc = ax.scatter(gdf[gdf.ci_type==ci_type].geometry.x, 
                                gdf[gdf.ci_type==ci_type].geometry.y, 
                                c=gdf[gdf.ci_type==ci_type].casc_state,
                                cmap=InfraColorMaps().casc_col_map, 
                                transform=ccrs.PlateCarree(), 
                                s=1.5, vmin=0, vmax=2)
            
        ax.set_title(f'Functional Failures {ci_type}', weight='bold', fontsize=17)
    
    cbar = mpu.colorbar(
        h_casc, axes.flatten()[-2], size=0.05, pad=0.05, orientation='horizontal')
    cbar.set_ticks([0.33, 1., 1.66])
    cbar.set_ticklabels(['Func.', 'Dysfunc.', 'Casc.'])
    
    f.suptitle(f'Failure states from event {event_name}', weight='bold', fontsize=24)
    #f.tight_layout()
    f.subplots_adjust(bottom=0.05, top=0.95)  
    
    if save_path:
        plt.savefig(f'{save_path}'+f'failure_states_{event_name}.pdf', 
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
        

def infra_impact_plot(gdf, save_path=None, event_name=None):
    """ 
    per infrastructure, a plot of structural damages
    """
    
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    ci_types = set(gdf.ci_type).difference({'people'})
    
    f, axes = plt.subplots(3, int(np.ceil(len(ci_types)/3)), 
                           subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=(16,20*1.4142))

    for ci_type, ax in zip(ci_types, axes.flatten()):
        ax.set_extent(_get_extent(gdf), ccrs.PlateCarree())
        ax.add_feature(border, facecolor='none', edgecolor='0.5')
        
        if ci_type=='power_line':
            gdf[gdf.ci_type==ci_type].plot('imp_dir', ax=ax, markersize=1, linewidth=0.5, 
                                           transform=ccrs.PlateCarree(), vmin=0., vmax=300.)
        # don't plot pple-road connections
        elif ci_type=='road':
            gdf[gdf.ci_type==ci_type][
                :(len(gdf[gdf.ci_type=='road'])-len(gdf[gdf.ci_type=='people'])*2)
                ].plot('imp_dir', ax=ax, markersize=1, linewidth=0.5, 
                       transform=ccrs.PlateCarree(), vmin=0., vmax=300.)
        else:
            h_imp = ax.scatter(gdf[gdf.ci_type==ci_type].geometry.x, 
                               gdf[gdf.ci_type==ci_type].geometry.y, 
                        c=gdf[gdf.ci_type==ci_type].imp_dir,
                        transform=ccrs.PlateCarree(), s=1.5, vmin=0., vmax=1.)
        ax.set_title(f'Structural damages {ci_type}', weight='bold', fontsize=17)

    cbar = mpu.colorbar(h_imp, axes.flatten()[-2], size=0.05, pad=0.15, 
                        orientation='horizontal')
    cbar.set_label('Structural Damage Frac.')
    f.suptitle(f'Event nr. {event_name}', fontsize=24)
    
    if save_path:
        plt.savefig(f'{save_path}'+f'structural_impacts_{event_name}.pdf', 
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
        
def service_impact_plot(gdf, save_path=None, event_name=None):
    """
    per basic service, people cluster with and without access to that service
    """
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    services = [colname for colname in gdf.columns if 'actual_supply_' in colname] 
    
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    ci_types = set(gdf.ci_type).difference({'people'})
    f, axes = plt.subplots(3, int(np.ceil(len(ci_types)/3)), 
                           subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=(16,20*1.4142))

    for service, ax in zip(services, axes.flatten()[:len(services)]):
        ax.set_extent(_get_extent(gdf), ccrs.PlateCarree())
        ax.add_feature(border, facecolor='none', edgecolor='0.5')
        h_serv = ax.scatter(gdf[gdf.ci_type=='people'].geometry.x, 
                            gdf[gdf.ci_type=='people'].geometry.y, 
                            c=gdf[gdf.ci_type=='people'][service],
                            cmap=InfraColorMaps().service_col_map, 
                            transform=ccrs.PlateCarree(), 
                            vmin=-1., vmax=1., s=0.1)
        ax.set_title(f'Disruptions in access to {service[14:-7]}', 
                     weight='bold', fontsize=17)         
                
    cbar = mpu.colorbar(
        h_serv, axes.flatten()[-2], size=0.05, pad=0.05, orientation='horizontal')
    cbar.set_ticks([-0.66, 0., .66])
    cbar.set_ticklabels(['Disr.', 'Inavail.', 'Avail.'])
    
    f.suptitle(f'Service Disruptions from event {event_name}', weight='bold', fontsize=24)
    #f.tight_layout()
    f.subplots_adjust(bottom=0.05, top=0.95)  
                                       
    if save_path:
        plt.savefig(f'{save_path}'+f'service_disruptions_{event_name}.pdf', 
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
   
        
def service_cumimpact_plot(gdf_services, save_path=None):
    """
    per basic service, people cluster with and without access to that service
    """
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    services = [colname for colname in gdf_services.columns if 'actual_supply_' in colname] 
    
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    ci_types = set(gdf.ci_type).difference({'people'})
    f, axes = plt.subplots(3, int(np.ceil(len(ci_types)/3)), 
                           subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=(16,20*1.4142))

    for service, ax in zip(services, axes.flatten()[:len(services)]):
        ax.set_extent(_get_extent(gdf_services), ccrs.PlateCarree())
        ax.add_feature(border, facecolor='none', edgecolor='0.5')
        h_servcum = ax.scatter(gdf_services.geometry.x, gdf_services.geometry.y, 
                            c=gdf_services[service],
                            cmap=InfraColorMaps().servicecum_col_map, 
                            transform=ccrs.PlateCarree(), 
                            vmin=-9., vmax=1., s=0.1)
        ax.set_title(f'Cumulativ disr`s in access to {service[14:-7]}', 
                     weight='bold', fontsize=17)         
                
    cbar = mpu.colorbar(
        h_servcum, axes.flatten()[-2], size=0.05, pad=0.05, orientation='horizontal')
    cbar.set_ticks(np.arange(-9,1.5).tolist())
    cbar.set_ticklabels(['(9x)', '(8x)', '(7x)', '(6x)',
                         '(5x)', '(4x)' ,'(3x)', '(2x)',
                         '(1x)','Inavail.', 'Avail.'])
    
    f.suptitle('Cumulative disruptions from all flood events', weight='bold', fontsize=24)
    #f.tight_layout()
    f.subplots_adjust(bottom=0.05, top=0.95)  
                                       
    if save_path:
        plt.savefig(f'{save_path}'+'service_disruptions_cum.pdf', 
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

                
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

def sum_impacts(gdf_list):
    services = [colname for colname in gdf_list[0].columns if 'actual_supply_' in colname]
    gdf_services = gdf_list[0][gdf_list[0].ci_type=='people'][['counts', 'geometry']]
    
    for service in services:
        service_counts = np.array(gdf_list[0][gdf_list[0].ci_type=='people'][service].values)
        for gdf in gdf_list[1:]:
            service_counts = np.vstack([service_counts, np.array(gdf[gdf.ci_type=='people'][service].values)])
        failures = np.ma.masked_greater_equal(service_counts, 0).sum(axis=0).filled(np.nan)
        inavails = np.ma.masked_not_equal(service_counts, 0).sum(axis=0).filled(np.nan)
        failures[~np.isnan(inavails)] = 0.
        failures[np.isnan(failures)] = 1.
        gdf_services[service] = failures
        
    return gdf_services
    
# =============================================================================
# Execution
# =============================================================================

cntry = sys.argv[1:]

PATH_SAVE = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/network_stuff/p2/'
iso3 = u_coords.country_to_iso(cntry)
folder_path = PATH_SAVE + f'{iso3}/'
file_paths = glob.glob(folder_path + 'cascade_results_*')
save_path = folder_path+'plots/'
node_gdf_orig = gpd.read_feather(folder_path+'cis_nw_nodes')

gdf_list= []
for file_path in file_paths:
    gdf_list.append(gpd.read_feather(file_path))
    
  
for i, gdf in enumerate(gdf_list):
    gdf_list[i] = get_accessstates(gdf, node_gdf_orig)
    gdf_list[i]['casc_state'] = get_cascstate(gdf)

# checking results
for i, gdf in enumerate(gdf_list):   
    services = [colname for colname in gdf.columns if 'actual_supply_' in colname]
    event_name = file_paths[i][-8:]
    for service in services:
        serv_level = gdf_list[i][gdf_list[i].ci_type=='people'][service].values
        print(event_name, service, np.unique(serv_level))

service_impact_plot(gdf_list[i], save_path=None, event_name=file_paths[i][-8:]) 
infra_func_plot(gdf_list[i], event_name=file_paths[i][-8:])
infra_impact_plot(gdf_list[i], file_paths[i][-8:])
    
gdf_services = sum_impacts(gdf_list)
service_cumimpact_plot(gdf_services)


if not os.path.isdir(save_path):
    os.mkdir(save_path)




    