#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:18:33 2017

@author: gfleishman
"""

# TODO: not so great interactions between some of the parameter specs
# e.g. subset and resolutions and their default values
# also, singleton res and iter could cause problems
# consider reformulating input specifications w.r.t. 'x' and ':'
# a method to initialize dictionaries to a default value
def default_dict(keys, def_value):
    dd = {}
    for k in keys:
        dd[k] = def_value
    return dd


# the name of each input parameter
input_params = ['inputs',
                'model',
                'optimizer',
                'matcher',
                'regularizer',
                'times',
                'timesteps',
                'subset',
                'sigma',
                'step',
                'stddev',
                'abcd',
                'window',
                'iterations',
                'resolutions',
                'voxel',
                'feedback']

# a dictionary for the input parameter flags used on the command line
input_flags = default_dict(input_params, '')
input_flags.update( {'inputs':          '-i',
                     'model':           '-dm',
                     'optimizer':       '-o',
                     'matcher':         '-m',
                     'regularizer':     '-r',
                     'times':           '-t',
                     'timesteps':       '-ts',
                     'subset':          '-subset',
                     'sigma':           '-sigma',
                     'step':            '-step',
                     'stddev':          '-stddev',
                     'abcd':            '-abcd',
                     'window':          '-w',
                     'iterations':      '-iter',
                     'resolutions':     '-res',
                     'voxel':           '-vox',
                     'feedback':        '-fb'} )

# a dictionary for the input parameter valid values
input_valids = default_dict(input_params, [])
input_valids.update( {'model':            ['elastic',
                                           'fluid',
                                           'lddmm',
                                           'syn',
                                           'gsid',
                                           'sigsid',
                                           'grid'],
                      'optimizer':        ['static',
                                           'secant',
                                           'bb'],
                      'matcher':          ['ssd',
                                           'gcc',
                                           'lcc',
                                           'mi'],
                      'regularizer':      ['gaussian',
                                           'differential']} )

# a dictionary for the input parameter default values
# 0:0:0 or 0x0x0 is a placeholder, value will be taken directly from inputs
input_defaults = default_dict(input_params, '')
input_defaults.update( {'model':           'elastic',
                        'optimizer':       'static',
                        'matcher':         'gcc',
                        'regularizer':     'gaussian',
                        'times':           '0:1',
                        'timesteps':       '8',
                        'subset':          '0:0:0',
                        'sigma':           '1',
                        'step':            '.01',
                        'stddev':          '2.5:2.5:2.5',
                        'abcd':            '1:0:0.1:2',
                        'window':          '11',
                        'iterations':      '100:50:10',
                        'resolutions':     '64x64x64:128x128x128:0x0x0',
                        'voxel':           '0:0:0',
                        'feedback':        '0'} )

# a dictionary to indicate which inputs are mandatory
input_mandatory = default_dict(input_params, [])
input_mandatory.update( {'inputs':          ['always'],
                         'times':           ['model:gsid',
                                             'model:grid',
                                             'model:sigsid']} )

# a dictionary to indicate if and what type of number a param is
input_nums = default_dict(input_params, '')
input_nums.update( {'times':           float,
                    'timesteps':       int,
                    'subset':          int,
                    'sigma':           float,
                    'step':            float,
                    'stddev':          float,
                    'abcd':            float,
                    'window':          int,
                    'iterations':      int,
                    'resolutions':     int,
                    'voxel':           float,
                    'feedback':        int} )

# a dictionary for the input parameter error messages
input_errors = default_dict(input_params, '')
input_errors.update( {
                     'inputs':
"""
ERROR: Something is wrong with the input specifications. You must specify
at least two images in nifti format.
Example:
-i /path/to/template.nii.gz /path/to/target.nii.gz
""",
                     'model':
"""
ERROR: \'%s\' is not a valid deformation model.
Valid deformation models are:
""" + str(input_valids['model']),

                     'optimizer':
"""
ERROR: \'%s\' is not a valid optimizer.
Valid optimizers are:
""" + str(input_valids['optimizer']),

                     'matcher':
"""
ERROR: \'%s\' is not a valid matching functional.
Valid matching functionals are:
""" + str(input_valids['matcher']),

                     'regularizer':
"""
ERROR: \'%s\' is not a valid regularizer.
Valid regularizers are:
""" + str(input_valids['regularizer']),

                     'times':
"""
ERROR: if deformation model is \'gsid\', \'sigsid\',
or \'grid\' then you must specify the image
acquisition times, in years, colon separated
with -t. Example:
-t 0:0.5:1.0
""",
                     'timesteps':
"""
ERROR: Timesteps must be an integer
""",

                     'subset':
"""
ERROR: subset format incorrect.
Example:
-subset 0x128:0x128:64x65
""",

                     'sigma':
"""
ERROR: -sigma must be a real number
""",

                     'step':
"""
ERROR: -step must be a real number
""",

                     'stddev':
"""
ERROR: -stddev must specify d real numbers colon separated
where d is the spatial dimensionality of the input images
Example:
-stddev 2.5:2.5:2.5
""",

                     'abcd':
"""
ERROR: -abcd must specify 4 real numbers colon separated
Example:
-abcd 1:0:0.1:2
""",

                     'iterations':
"""
ERROR: -iter must specify integers separated by colons.
The list can be arbitrarily long.
Example:
-iter 100:50:10
""",

                     'resolutions':
"""
ERROR: -res must specify resolutions, each dimension separated by x.
Resolutions themselves must be separated by colons. There must be
the same number of resolutions as there are integers to the
iterations (-iter) argument. Use 'full' to specify the full resolution
of the input images.
Example:
-res 64x64x64:128x128x128:full
""",

                     'voxel':
"""
ERROR: -vox must specify d real numbers, colon separated, where
d is the spatial dimension of the images.
Example:
-vox 1:1:1.125
""",
                     'physical_space':
"""
ERROR: Images do not occupy the same physical space.
""",
                     'feedback':
"""
ERROR: feedback error, example: -fb 1
"""} )

# the name of each output parameter
output_params = ['warp',
                'invwarp',
                'momentum',
                'objective']

# a dictionary for the output parameter flags used on the command line
output_flags = default_dict(output_params, '')
output_flags.update( {'warp':           '-warp',
                      'invwarp':        '-invwarp',
                      'momentum':       '-mom',
                      'objective':      '-obj'} )

# a dictionary for the output parameter default values
output_errors = default_dict(output_params, '')
output_errors.update( {
        'warp':
"""
ERROR: output directory for warp does not exist
""",
        'invwarp':
"""
ERROR: output directory for inverse warp does not exist
""",
        'momentum':
"""
ERROR: output directory for momentum does not exist
""",
        'objective':
"""
ERROR: output directory for objective does not exist
"""} )

# TODO: -step is incomplete for secant and bb methods
# TODO: update this message, make sure everything is correct
help_message = """

 pyrpl: version 0.1
 Python Registration Prototyping Library
 Author: Greg M. Fleishman, PhD

MANDATORY ARGUMENTS
-i:     			 Input image filepaths.
        			 The first filepath will be the moving image.
        			 For most deformation models, there should be only two file paths.
        			 For grid, there can be an arbitrary number of filepaths.
        			 -----------------------
OPTIONAL ARGUMENTS
-dm:    			 The deformation model.
        			 options include: elastic, fluid, lddmm, syn, gsid, sigsid, grid
        			 default: -dm elastic
        			 -----------------------
	-t:     		 required/relevant if -dm == gsid, sigsid, or grid
        			 specifies the image acquisition times
                    example: -t 0:1.0:2.5
        			 -----------------------
	-ts:     		 number of timesteps along the geodesic to sample
        			 only relevant for gsid, sigsid, or grid models
                    example: -ts 8
        			 -----------------------
-o:     			 The optimizer model.
        			 options include: static, secant, bb
        			 default: -o static
        			 -----------------------
-m:     			 The image matching functional.
        			 options include: ssd, gcc, lcc, mi
        			 default: -m gcc
        			 -----------------------
-r:     			 The regularizer.
        			 options include: gaussian, differential
        			 default: -r gaussian
        			 -----------------------
	-stddev:		 relevant if -r == gaussian
        			 the standard deviations (for each spatial dimension) of the gaussian
        			 units are in millimeters
        			 default: -stddev 2.5:2.5:2.5
        			 -----------------------
	-abcd:  		 relevant if -r == differential
        			 the parameters to the differential operator
        			 units are in millimeters
        			 default: -abcd 1.0:0.0:0.1:2.0
        			 -----------------------
-subset:     	      the subset of the image grid that you want to register
                    useful if you want a 2D registration
                    example: -subset 0x128:0x128:64:65
        			 -----------------------
-step:   			 the gradient descent step size
        			 has different meanings for different values of -o
        			 for -o == static, specifies the static gradient descent step size
        			 for -o == secant, specifies the 
        			 for -o === bb, specifies the
        			 default: -step 0.01
        			 -----------------------
-iter:  			 The number of iterations for each resolution
        			 must be equal in length to argument of -res
        			 default: -iter 100:50:10
        			 -----------------------
-res:    			 The resolutions at which to register
        			 must be equal in length to argument of -iter
        			 default: -res 64x64x64:128x128x128:full (full == full image resolution)
        			 -----------------------
-sigma: 			 compromise parameter between image matching and regularization
        			 default: -sigma 1e-6
        			 -----------------------
-vox:      		 the voxel size in millimeters
        			 format example: 1.0x1.0x1.0
        			 default: determined from moving image header
        			 -----------------------

"""
