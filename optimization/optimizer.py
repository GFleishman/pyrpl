#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:00:52 2017
@author: gfleishman
"""


# this function just routes the optimization to the correct optimizer
def optimize(input_dictionary):
    """Identify the correct optimizer and call it"""
    if input_dictionary['optimizer'] == 'static':
        import pyrpl.optimization.static as opt
    elif input_dictionary['optimizer'] == 'secant':
        import pyrpl.optimization.secant as opt
    elif input_dictionary['optimizer'] == 'bb':
        import pyrpl.optimization.bb as opt

    return opt.optimize(input_dictionary)
