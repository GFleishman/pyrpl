#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:00:52 2017

@author: gfleishman
"""

import pyrpl.optimization.static as static
import pyrpl.optimization.secant as secant
import pyrpl.optimization.bb as bb

import pyrpl.optimization.static_local as static_local

def optimize(input_dictionary):
        if input_dictionary['optimizer'] == 'static':
            return static.optimize(input_dictionary)
        elif input_dictionary['optimizer'] == 'secant':
            return secant.optimize(input_dictionary)
        elif input_dictionary['optimizer'] == 'bb':
            return bb.optimize(input_dictionary)

        elif input_dictionary['optimizer'] == 'static_local':
            return static_local.optimize(input_dictionary)
