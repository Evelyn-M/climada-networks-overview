#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:39:21 2022

@author: evelynm
"""

from datetime import datetime
import numpy as np
import sys

from climada.util import coordinates as u_coords
from climada.util.api_client import Client

START_STR = '01-01-2000'
END_STR = '31-12-2018'


cntry = sys.argv[1:]
iso3 = u_coords.country_to_iso(cntry)

client = Client()
tc = client.get_hazard('tropical_cyclone', 
                       properties={'country_iso3alpha':iso3, 
                                   'climate_scenario': 'historical',
                                   'spatial_coverage': 'country'})
# only historic ones                                                     
tc = tc.select(orig=True)       

# only in between cloud to street DB times                                                      
startdate_ordinal = datetime.strptime(START_STR, '%d-%m-%Y').date().toordinal()
enddate_ordinal = datetime.strptime(END_STR, '%d-%m-%Y').date().toordinal()
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

tc.event_name