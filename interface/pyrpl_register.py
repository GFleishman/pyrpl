# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:28:04 2015

@author: gfleishman
"""

import os
import sys
import numpy as np
import nibabel as nib
import pyrpl.interface.valids as valids
import pyrpl.interface.messages as messages
import pyrpl.image_tools.preprocessing as preprocessing
import pyrpl.optimization.optimizer as optimizer


# helper function for reading in some input parameters that have defaults
def default_parameter(flag, name, default, valids,
                       error_string, input_dictionary):
    if flag in sys.argv:
        param = sys.argv[sys.argv.index(flag)+1]
        if param in valids:
            input_dictionary[name] = param
        else:
            print error_string % (param)
            print valids
            sys.exit()
    else:
        print 'using default %s: %s' % (name, default)
        input_dictionary[name] = default


# read in and double check validity of input parameters
def parse_input_commands():
    # provide help documentation to the user
    if '-help' in sys.argv:
        print messages.help_message
        sys.exit()

    # a master dictionary to hold all input specifications
    input_dictionary = {}

    # params with defaults: model, optimizer, matcher, regularizer
    default_parameter('-dm', 'model', 'elastic', valids.models,
                       messages.deformation_error, input_dictionary)
    default_parameter('-o', 'optimizer', 'static', valids.optimizers,
                       messages.optimizer_error, input_dictionary)
    default_parameter('-m', 'matcher', 'gcc', valids.matchers,
                       messages.matcher_error, input_dictionary)
    default_parameter('-r', 'regularizer', 'gaussian', valids.regularizers,
                       messages.regularizer_error, input_dictionary)

    # length of time series, mandatory co-parameter for grid
    if input_dictionary['model'] == 'grid':
        if '-n' in sys.argv:
            n = sys.argv[sys.argv.index('-n')+1]
            try:
                n = int(n)
            except:
                print messages.number_error1
                sys.exit()
            input_dictionary['number'] = n
        else:
            print messages.number_error2
            sys.exit()
    if 'number' not in input_dictionary.keys():
        input_dictionary['number'] = 2

    # acquisition times, mandatory co-parameter for gsid, sigsid, or grid
    if (input_dictionary['model'] == 'gsid' or
    input_dictionary['model'] == 'sigsid' or
    input_dictionary['model'] == 'grid'):
        if '-t' in sys.argv:
            t = [sys.argv[sys.argv.index('-t')+i+1] for i
                  in range(input_dictionary['number'])]
            try:
                t = np.array(t).astype(np.float)
            except:
                print messages.times_error1
                sys.exit()
            # only care about follow up times
            input_dictionary['times'] = t - t[0]
        else:
            print messages.times_error2
            sys.exit()

    # determine dimensionality of registration
    if '-d' in sys.argv:
        d = sys.argv[sys.argv.index('-d')+1]
        try:
            d = int(d)
        except:
            print messages.dimension_error
            sys.exit()
        input_dictionary['dimension'] = d
    else:
        print messages.dimension_warning
        input_dictionary['dimension'] = 3

    # mandatory co-parameters if d == 2
    if input_dictionary['dimension'] == 2:
        # determine which axis to slice from
        if '-axis' in sys.argv:
            axis = sys.argv[sys.argv.index('-axis')+1]
            try:
                axis = int(axis)
            except:
                print messages.axis_error1
                sys.exit()
            input_dictionary['axis'] = axis
        else:
            print messages.axis_error2
            sys.exit()

        # determine which slice to register
        if '-slice' in sys.argv:
            slc = sys.argv[sys.argv.index('-slice')+1]
            try:
                slc = int(slc)
            except:
                print messages.slice_error1
                sys.exit()
            input_dictionary['slice'] = slc
        else:
            print messages.slice_error2
            sys.exit()

    # determine gradient descent step size
    if '-step' in sys.argv:
        step = sys.argv[sys.argv.index('-step')+1]
        try:
            step = float(step)
        except:
            print messages.step_error
            sys.exit()
        input_dictionary['step'] = step
    else:
        print messages.step_warning
        input_dictionary['step'] = 0.01

    # determine gaussian regularizer parameters
    if input_dictionary['regularizer'] == 'gaussian':
        if '-stddev' in sys.argv:
            stddev = [sys.argv[sys.argv.index('-stddev')+i+1] for i
                      in range(input_dictionary['dimension'])]
            try:
                stddev = np.array(stddev).astype(np.float)
            except:
                print messages.stddev_error
                sys.exit()
            input_dictionary['stddev'] = stddev
        else:
            print messages.stddev_warning
            input_dictionary['stddev'] = np.array([3., 3., 3.])

    # determine differential regularizer parameters
    if input_dictionary['regularizer'] == 'differential':
        if '-abcd' in sys.argv:
            abcd = [sys.argv[sys.argv.index('-abcd')+i+1] for i in range(4)]
            try:
                abcd = np.array(abcd).astype(np.float)
            except:
                print messages.abcd_error
                sys.exit()
            input_dictionary['abcd'] = abcd
        else:
            print messages.abcd_warning
            input_dictionary['abcd'] = np.array([1., 0., .1, 2.])

    # determine multi-resolution iteration scheme
    if '-iter' in sys.argv:
        iterations = sys.argv[sys.argv.index('-iter')+1]
        iterations = iterations.split('x')
        try:
            iterations = np.array(iterations).astype(np.int)
        except:
            print messages.iterations_error
            sys.exit()
        input_dictionary['iterations'] = iterations
    else:
        print messages.iterations_warning
        input_dictionary['iterations'] = np.array([100, 50, 10])

    # determine resolutions for multi-resolution iteration scheme
    if '-res' in sys.argv:
        resolutions = sys.argv[sys.argv.index('-res')+1]
        resolutions = resolutions.split('|')
        resolutions = [res.split('x') for res in resolutions]
        try:
            resolutions = [np.array(res).astype(np.int) for res in resolutions]
        except:
            print messages.resolutions_error
            sys.exit()
        input_dictionary['resolutions'] = resolutions
    else:
        print messages.resolutions_warning
        d = input_dictionary['dimension']
        input_dictionary['resolutions'] = [np.array([64,]*d),
                                           np.array([128,]*d)]
        # ^ Will add full resolution after images are loaded

    # determine regularizer/data match compromise parameter
    if '-sigma' in sys.argv:
        sigma = sys.argv[sys.argv.index('-sigma')+1]
        try:
            sigma = float(sigma)
        except:
            print messages.sigma_error
            sys.exit()
        input_dictionary['sigma'] = sigma
    else:
        print messages.sigma_warning
        input_dictionary['sigma'] = 1e-6

    # determine image paths and read in images
    if '-i' in sys.argv:
        n = input_dictionary['number']
        inputs = [sys.argv[sys.argv.index('-i')+i+1] for i in range(n)]
        input_dictionary['inputs'] = inputs
        try:
            images = [nib.load(path) for path in inputs]
        except:
            print messages.input_error1
            sys.exit()
        input_dictionary['images'] = images
    else:
        print messages.input_error2
        sys.exit()

    # if necessary, update resolutions to contain full resolution
    if (len(input_dictionary['resolutions']) ==
    len(input_dictionary['iterations']) - 1):
        full = input_dictionary['images'][0].header.get_data_shape()
        full = np.array(full).astype(np.int)
        if 'axis' in input_dictionary.keys():
            np.delete(full, input_dictionary['axis'])
        input_dictionary['resolutions'].append(full)

    # determine voxel dimensions
    if '-v' in sys.argv:
        voxel = sys.argv[sys.argv.index('-v')+1]
        voxel = voxel.split('x')
        try:
            voxel = np.array(iterations).astype(np.float)
        except:
            print messages.voxel_error1
            sys.exit()
        input_dictionary['voxel'] = voxel
    else:
        print messages.voxel_warning
        voxel = input_dictionary['images'][0].header.get_zooms()
        try:
            voxel = np.array(voxel).astype(np.float)
        except:
            print messages.voxel_error2
            sys.exit()
        if 'axis' in input_dictionary.keys():
            np.delete(voxel, input_dictionary['axis'])
        input_dictionary['voxel'] = voxel

    return input_dictionary


# helper function for reading in some optional output parameters
def optional_parameter(flag, name, output_dictionary):
    if flag in sys.argv:
        param = sys.argv[sys.argv.index(flag) + 1]
        param_directory = os.path.dirname(param)
        if not os.path.isdir(param_directory):
            print messages.dir_error % (param_directory)
            sys.exit()
        else:
            output_dictionary[name] = param


# read in and double check validity of output parameters
def parse_output_commands(input_dictionary):
    # master dictionary for all output related specifications
    output_dictionary = {}

    # determine deformation output filepath
    if '-def' in sys.argv:
        deform = sys.argv[sys.argv.index('-def') + 1]
        deform_directory = os.path.dirname(deform)
        if not os.path.isdir(deform_directory):
            print messages.dir_error % (deform_directory)
            sys.exit()
        else:
            output_dictionary['deformation'] = deform
    else:
        print messages.deform_error1
        sys.exit()

    # optional: determine forward deformation filepath
    if '-inv' in sys.argv:
        if not input_dictionary['model'] in valids.diffeo_models:
            print messages.inv_error1
            sys.exit()
        else:
            inv = sys.argv[sys.argv.index('-inv') + 1]
            inv_directory = os.path.dirname(inv)
            if not os.path.isdir(inv_directory):
                print messages.dir_error % (inv_directory)
                sys.exit()
            else:
                output_dictionary['inverse'] = inv

    # determine momenta output filepath (mandatory for diffeo models)
    if input_dictionary['model'] in valids.diffeo_models:
        if not '-mom' in sys.argv:
            print messages.mom_error1
            sys.exit()
        else:
            mom = sys.argv[sys.argv.index('-mom') + 1]
            mom_directory = os.path.dirname(mom)
            if not os.path.isdir(mom_directory):
                print messages.dir_error % (mom_directory)
                sys.exit()
            else:
                output_dictionary['momenta'] = mom

    # optional params: warped moving, warped target, optimization array
    optional_parameter('-warped', 'warped', output_dictionary)
    optional_parameter('-invwarped', 'invwarped', output_dictionary)
    optional_parameter('-dist', 'distance', output_dictionary)

    return output_dictionary


# read in and preprocess image data
def read_and_preprocess_data(input_dictionary):
    # read in image data
    n = input_dictionary['number']
    sh = input_dictionary['images'][0].header.get_data_shape()
    input_dictionary['imgdata'] = np.empty((n,) + sh)
    for i in range(n):
        input_dictionary['imgdata'][i] = (
            input_dictionary['images'][i].get_data().squeeze())

    # check for shave
    if '--shave' in sys.argv:
        bounds = np.empty((input_dictionary['dimension'], 2))
        bounds[:, 0] = np.iinfo(np.int).max
        bounds[:, 1] = 0
        for i in range(n):
            b = preprocessing.shave(input_dictionary['imgdata'][i])
            bounds[:, 0] = np.minimum(bounds[:, 0], b[:, 0])
            bounds[:, 1] = np.maximum(bounds[:, 1], b[:, 1])
        # pad a little bit
        bounds[:, 0] -= 5
        bounds[:, 1] += 5
        bounds[bounds < 0] = 0
        # ^ THIS CODE SHOULD REALLY BE IN preprocessing.py ITSELF
            

    # check for cube
    if '--cube' in sys.argv:
        pass  # TODO:

    # check for obtain_foreground_mask
    if '--foreground' in sys.argv:
        pass  # TODO:
    
    # check for compress_intensity_range
    if '--range' in sys.argv:
        pass  # TODO:
    
    # check for histogram_match
    if '--hm' in sys.argv:
        pass  # TODO:
    
    # check for scale_intensity
    if '--scale' in sys.argv:
        pass  # TODO:


def write_output(output_dictionary, result):

    pass
    #TODO: will have to fill this in when I get models and optimizers sorted


def main():

    # Get registration specifications
    input_dictionary = parse_input_commands()
    output_dictionary = parse_output_commands(input_dictionary)

    # read in and preprocess image data
    read_and_preprocess_data(input_dictionary)
    
    # report final registration specifications
    print '\nREGISTRATION SPECIFICATIONS:'
    for key in input_dictionary.keys():
        print key + ":\t\t\t" + str(input_dictionary[key])
    print '\nWRITE SPECIFICATIONS:'
    for key in output_dictionary.keys():
        print key + ":\t\t\t" + str(output_dictionary[key])

    # Fit geodesic
    result = optimizer.optimize(input_dictionary)

    # Write the output
    write_output(output_dictionary, result)

main()
