#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:24:55 2017

@author: gfleishman
"""

import pyrpl.models.elastic as elastic
import pyrpl.models.fluid as fluid
import pyrpl.models.lddmm as lddmm
import pyrpl.models.gsid as gsid
import pyrpl.models.sigsid as sigsid
import pyrpl.models.grid as grid

def modeler(input_dictionary):
    if input_dictionary['model'] == 'elastic':
        return elastic.elastic(input_dictionary)
    if input_dictionary['model'] == 'fluid':
        return fluid.fluid(input_dictionary)
    if input_dictionary['model'] == 'lddmm':
        return lddmm.lddmm(input_dictionary)
    if input_dictionary['model'] == 'gsid':
        return gsid.gsid(input_dictionary)
    if input_dictionary['model'] == 'sigsid':
        return sigsid.sigsid(input_dictionary)
    if input_dictionary['model'] == 'grid':
        return grid.grid(input_dictionary)
