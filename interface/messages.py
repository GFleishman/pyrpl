#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:05:13 2017

@author: gfleishman
"""

help_message = """\n pyrpl: version 0.1
 Python Registration Prototyping Library
 Author: Greg M. Fleishman\n\n"""

help_message += "MANDATORY ARGUMENTS\n"

help_message += """-i:     \t\t\t Input image filepaths.
        \t\t\t The first filepath will be the moving image.
        \t\t\t For most deformation models, there should be only two file paths.
        \t\t\t For grid, there can be an arbitrary number of filepaths.
        \t\t\t -----------------------\n"""

help_message += "OPTIONAL ARGUMENTS\n"

help_strings = []
help_strings.append("""-dm:    \t\t\t The deformation model.
        \t\t\t options include: elastic, fluid, lddmm, syn, gsid, sigsid, grid
        \t\t\t default: elastic\n""")
help_strings.append("""\t-n:     \t\t required/relevant if -dm == grid
        \t\t\t specifies the number of images in the time series\n""")
help_strings.append("""\t-t:     \t\t required/relevant if -dm == gsid, sigsid, or grid
        \t\t\t specifies the image acquisition times\n""")
help_strings.append("""-o:     \t\t\t The optimizer model.
        \t\t\t options include: static, secant, bb
        \t\t\t default: static\n""")
help_strings.append("""-m:     \t\t\t The image matching functional.
        \t\t\t options include: ssd, gcc, lcc, mi
        \t\t\t default: gcc\n""")
help_strings.append("""-r:     \t\t\t The regularizer.
        \t\t\t options include: gaussian, differential
        \t\t\t default: gaussian\n""")
help_strings.append("""\t-stddev:\t\t required/relevant if -r == gaussian
        \t\t\t the standard deviations (for each spatial dimension) of the gaussian
        \t\t\t example format: 2.0x2.0x2.0 for -d == 3, or 2.0x2.0 for -d == 2
        \t\t\t units are in millimeters
        \t\t\t default: 3.0x3.0x3.0\n""")
help_strings.append("""\t-abcd:  \t\t required/relevant if -r == differential
        \t\t\t the parameters to the differential operator
        \t\t\t example format: 1.0x1.0x0.1x2.0
        \t\t\t units are in millimeters
        \t\t\t default: 1.0x0.0x0.1x2.0\n""")
help_strings.append("""-d:     \t\t\t the dimension of the registration
        \t\t\t default: 3, if -d == 2, you must specify -axis and -slice\n""")
help_strings.append("""\t-axis:  \t\t required/relevant if -d == 2
        \t\t\t the axis along which to slice for 2d registration\n""")
help_strings.append("""\t-slice:  \t\t required/relevant if -d == 2
        \t\t\t the specific slice to take for 2d registration\n""")
help_strings.append("""-step:   \t\t\t the gradient descent step size
        \t\t\t has different meanings for different values of -o
        \t\t\t for -o == static, specifies the static gradient descent step size
        \t\t\t for -o == secant, specifies the TODO: look this up and fix
        \t\t\t for -o === bb, specifies the TODO: look this up and fix
        \t\t\t default: 0.01\n""")
help_strings.append("""-iter:  \t\t\t The number of iterations for each resolution
        \t\t\t must be equal in length to argument of -res
        \t\t\t format example: 100x50x10
        \t\t\t default: 100x50x10\n""")
help_strings.append("""-res:    \t\t\t The resolutions at which to register
        \t\t\t must be equal in length to argument of -iter
        \t\t\t format example: 32x32x32|64x64x64|128x128x128
        \t\t\t default: 64^d|128^d|full, where full is the full image resolution\n""")
help_strings.append("""-sigma: \t\t\t compromise parameter between image matching and regularization
        \t\t\t default: 1e-6\n""")
help_strings.append("""-v:      \t\t\t the voxel size in millimeters
        \t\t\t format example: 1.0x1.0x1.0
        \t\t\t default: determined from moving image header\n""")

for s in help_strings:
    help_message += s
    help_message += '        \t\t\t -----------------------\n'





deformation_error = ('\n\nERROR: \'%s\' is not a valid deformation model; '
                       'Valid deformation models are:')

optimizer_error = ('\n\nERROR: \'%s\' is not a valid optimizer; Valid optimizers are: ')

matcher_error = ('\n\nERROR: \'%s\' is not a valid matching functional; '
                   'Valid matching functionals are: ')

regularizer_error = ('\n\nERROR: \'%s\' is not a valid regularizer; '
                       'Valid regularizers are: ')

number_error1 = ('\n\nERROR: -n must specify the number of images in your '
                 'time series as an integer')

number_error2 = ('\n\nERROR: if deformation model is \'grid\' then you must '
                 'specify the number of images in your time series '
                 'as an integer with -n')

times_error1 = ('\n\nERROR: -t must specify the acquisition time, in '
                'years, for each image in your time series; '
                'it must be a list of floats whose length is '
                'equal to the number of images you have input')

times_error2 = ('\n\nERROR: if deformation model is \'gsid\', \'sigsid\', '
                'or \'grid\' then you must specify the image '
                'acquisition times for your images as '
                'arguments to -t')

dimension_error1 = ('\n\nERROR: -d must be an integer')

dimension_warning = ('assuming default spatial dimension: 3')

axis_error1 = ('\n\nERROR: -axis must be an integer')

axis_error2 = ('\n\nERROR: if -d is 2, then you must specify -axis')

slice_error1 = ('\n\nERROR: -slice must be an integer')

slice_error2 = ('\n\nERROR: if -d is 2, then you must specify -slice')

step_error = ('\n\nERROR: -step must be a real number')

step_warning = ('using default step: 0.01')

stddev_error = ('\n\nERROR: -stddev must specify d floating point numbers '
                'where d is the spatial dimensionality of the '
                'input images')

stddev_warning = ('using default stddev: 3x3x3 millimeters')

abcd_error = ('\n\nERROR: -abcd must specify 4 floating point numbers')

abcd_warning = ('using default abcd: 1.0, 0.0, 0.1, 2.0')

iterations_error = ('\n\nERROR: -iter must specify integers delimited by x; '
                    'e.g. 100x50x10x... where ... indicates the list '
                    'can be arbitrarily long')

iterations_warning = ('using default iterations scheme: 100x50x10')

resolutions_error = ('\n\nERROR: -res must specify resolutions delimited by |; '
                     'each resolution must be integers delimited by x; '
                     'e.g. 64x64x64|128x128x128|220x220x220')

resolutions_warning = ('using default resolution scheme: 64^d|128^d|full')

sigma_error = ('\n\nERROR: -sigma must be a real number')

sigma_warning = ('using default sigma: 1e-6')

input_error1 = ('\n\nERROR: one or more of your filepaths does not exist or is not '
                'a nifti image')

input_error2 = ('\n\nERROR: you must specify at least two input image paths with -i')

voxel_error1 = ('-v must be d floating point numbers separated by x, where '
                'd is the dimension passed in with -d (default is 3); e.g. '
                '1.0x1.0x1.0')

voxel_warning = ('-v omitted, determining voxel dimensions from header of '
                 'first input image')

voxel_error2 = ('\n\nERROR: could not determine voxel dimensions from first input image '
                'header')

dir_error = ('\n\nERROR: write directory %s does not exist')

deform_error1 = ('\n\nERROR: you must specify a write directory for the deformation '
             'with -def')

inv_error1 = ('\n\nERROR: to obtain inverse transformation with -inv flag, model '
             'must be diffeomorphic; this includes: lddmm, syn, gsid, sigsid, '
             'and grid')

mom_error1 = ('\n\nERROR: you must specify a momentum output path with -mom when '
              'using a diffeomorphic deformation model (lddmm, syn, gsid, sigsid, '
              'grid)')