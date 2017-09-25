#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:24:55 2017

@author: gfleishman
"""


# TODO: write getter methods for all models, get elements of data container
# TODO: rewrite model constructors to load image data (make J, and T)
# this function just routes the model selection to the correct model
def model(input_dictionary):
    """Identify the correct model and return it"""
    if input_dictionary['model'] == 'elastic':
        import pyrpl.models.elastic as mod
    elif input_dictionary['model'] == 'fluid':
        import pyrpl.models.fluid as mod
    elif input_dictionary['model'] == 'lddmm':
        import pyrpl.models.lddmm as mod
    elif input_dictionary['model'] == 'gsid':
        import pyrpl.models.gsid as mod
    elif input_dictionary['model'] == 'sigsid':
        import pyrpl.models.sigsid as mod
    elif input_dictionary['model'] == 'grid':
        import pyrpl.models.grid as mod

    return mod.model(input_dictionary)

"""
Optimizers make the following assumptions about models:
X    input_dictionary['subset'] needs to be a slice object
X    modeler.model(input_dictionary)  # constructor
X        returns model object with following functions
X    model.resample(resolution)       # resample all items to resolution
X        returns nothing
X    model.evaluate()                 # evaluate objective funtion
X        returns list: [regu_match, [first_data_match, second_data_match, ...]]
X    model.get_gradient()             # evaluate gradient of obj. func.
X        returns list: [the gradient, its magnitude]
X    mode.take_step(step)             # update deformation
X        returns nothing
        
Models require the following getter methods:
X    model.get_original_image(i)  # gives integer, returns numpy array of ith image
X    model.get_warped_image(i)  # gives integer, returns numpy array of ith image
X    model.get_warp()          # returns numpy array of warp
X    model.get_current_vox()  # return current voxel size
"""
