#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:38:08 2022

@author: evelynm
"""

from climada_petals.entity.exposures.openstreetmap.osm_dataloader import OSMRaw


# Dominica
shape = [-61.492087, 15.194246, -61.227562, 15.666495]
path_extract =  '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data/osm_countries/dominica-latest.osm.pbf'

# Antigua and Barbuda
shape = [ -61.933336, 16.997665, -61.616106, 17.756504]
path_extract =  '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data/osm_countries/antigua-and-barbuda-latest.osm.pbf'

# Puerto Rico
shape = [ -67.323206, 17.899527, -65.578342, 18.550618]
path_extract =  '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data/osm_countries/puerto-rico-latest.osm.pbf'


path_parentfile = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data/osm_countries/central-america-latest.osm.pbf'
OSMRaw().get_data_fileextract(shape, path_extract, path_parentfile,
                              overwrite=False)


