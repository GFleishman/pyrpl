# -*- coding: utf-8 -*-
"""
@author: gfleishman
"""

import os
import sys
import numpy as np
import nibabel as nib
import pyrpl.interface.params as params
import pickle as pkl


# a helper function for brevity
def _input_error(error_string):
    """short hand for input errors"""
    print error_string
    sys.exit()


# a helper function to determine if a parameter is mandatory
def _mandatory_check(mandatory_list, input_dictionary):
    """Check if a parameter is mandatory given other arguments"""
    if mandatory_list:
        for s in mandatory_list:
            name, value = s.split(':')
            if input_dictionary[name] == value:
                return 1
    return 0


# a helper function to parse multi-arguments separated by 'x'
def _parse_multi_args_x(param, num, error_string):
    """Split up multiple arguments separated by x"""
    if 'x' not in param and not num:
        return param
    elif 'x' not in param and num:
        try:
            param = num(param)
        except:
            _input_error(error_string)
    else:
        param = param.split('x')
        try:
            param = [num(p) for p in param]
        except:
            _input_error(error_string)
    return param


# a helper function to parse multi-argument separated by ':'
def _parse_multi_args_col(param, num, error_string):
    """Split up multiple arguments separated by colons (:)"""
    if ':' not in param and not num:
        return param
    elif ':' not in param and 'x' not in param and num:
        try:
            param = num(param)
        except:
            _input_error(error_string)
        return param
    else:
        param = param.split(':')
        es = error_string  # just for brevity in next line
        proc_param = [_parse_multi_args_x(p, num, es) for p in param]
        return proc_param


# read in a parameter
def _read_parameter(name, flag, input_dictionary, error_string,
                    mandatory_list, default, valids, num):
    """Parse a parameter from the input string."""
    mandatory = _mandatory_check(mandatory_list, input_dictionary)
    if flag not in sys.argv and mandatory:
        _input_error(error_string)
    elif flag not in sys.argv and not mandatory and default:
        input_dictionary[name] = _parse_multi_args_col(default, num,
                                                       error_string)
    else:
        param = sys.argv[sys.argv.index(flag)+1]
        param = _parse_multi_args_col(param, num, error_string)
        if valids and param not in valids:
            _input_error(error_string)
        input_dictionary[name] = param


# a special method to read the unique inputs(-i) parameter
# the only parameter that has more than one white space separated argument
def _read_inputs_param(name, flag, input_dictionary, error_string):
    """Parse the arguments to the input (-i) flag"""
    if flag not in sys.argv:
        _input_error(error_string)
    file_path_indx = sys.argv.index(flag) + 1
    file_paths = []
    while sys.argv[file_path_indx][0] != '-':
        file_path = sys.argv[file_path_indx]
        if os.path.isfile(file_path):
            file_paths.append(file_path)
            file_path_indx += 1
        else:
            _input_error(error_string)
    input_dictionary[name] = file_paths


# a special method to load the images and update default params
# TODO: consider checks on res and vox when subset indicates 2D
def _load_input_images(input_dictionary, placeholder, error_message):
    """Loads the images and checks for physical space consistency.
    Also updates default parameters for max resolution, subset,
    and voxel size."""
    input_dictionary['images'] = [nib.load(p) for p in
                                  input_dictionary['inputs']]

    affine = [img.affine for img in
              input_dictionary['images']]
    voxels = [img.header['pixdim'][1:4] for img in
              input_dictionary['images']]

    if ((not all(np.allclose(x, affine[0]) for x in affine)) or
        (not all(np.allclose(x, voxels[0]) for x in voxels))):
        _input_error(error_message)

    shape = list(input_dictionary['images'][0].header.get_data_shape())
    if input_dictionary['subset'] == placeholder:
        input_dictionary['subset'] = [[0, x] for x in shape]
    if input_dictionary['resolutions'][-1] == placeholder:
        input_dictionary['resolutions'][-1] = shape
    if input_dictionary['voxel'] == placeholder:
        input_dictionary['voxel'] = voxels[0]
    
    # load the actual image data and slice out the appropriate subset
    J = [img.get_data().squeeze() for img in input_dictionary['images']]
    s = [slice(x[0], x[1]) for x in input_dictionary['subset']]
    input_dictionary['image_data'] = np.array([img[s].squeeze() for img in J])


# a special method to read the unique output parameters
def _read_output_params(name, flag, output_dictionary, error_string):
    """Parse an output parameter from the input string"""
    if flag in sys.argv:
        file_path = sys.argv[sys.argv.index(flag)+1]
        directory = os.path.abspath(os.path.join(file_path, os.pardir))
        if not os.path.isdir(directory):
            _input_error(error_string)
        else:
            output_dictionary[name] = file_path


# parse the entire input string
def parse_input_commands():
    """Parse the input string; check for mandatory and valid arguments.
    Not robust to every conceivable pathological input. Just don't do
    silly things like have file paths that begin with a dash '-'"""

    # provide help documentation to the user
    if '-help' in sys.argv:
        print params.help_message
        sys.exit()

    # a master dictionary to hold all input specifications
    input_dictionary = {}

    # for brevity, nickname the input dictionaries from params module
    input_params = params.input_params
    input_flags = params.input_flags
    input_valids = params.input_valids
    input_defaults = params.input_defaults
    input_mandatory = params.input_mandatory
    input_nums = params.input_nums
    input_errors = params.input_errors

    # parse the input files
    # the 'inputs' param is always first in the list
    inputs = input_params.pop(0)
    _read_inputs_param(inputs, input_flags[inputs],
                       input_dictionary, input_errors[inputs])

    # parse all other input commands
    for param in input_params:
        _read_parameter(param, input_flags[param], input_dictionary,
                        input_errors[param], input_mandatory[param],
                        input_defaults[param], input_valids[param],
                        input_nums[param])
    # recall, 0:0:0 is the placeholder for implicit defaults
    _load_input_images(input_dictionary, [0, 0, 0],
                       input_errors['physical_space'])

    return input_dictionary


# parse the entire output string
def parse_output_commands():
    """Parse the ouptut string; check for mandatory and valid arguments.
    Not robust to every conceivable pathological input. Just don't do
    silly things like have file paths that begin with a dash '-'"""

    # a master dictionary to hold all output specifications
    output_dictionary = {}

    # for brevity, nickname the output dictionaries from params module
    output_params = params.output_params
    output_flags = params.output_flags
    output_errors = params.output_errors

    # parse the output commands
    for param in output_params:
        _read_output_params(param, output_flags[param],
                            output_dictionary,
                            output_errors[param])

    return output_dictionary


# write the output; assumes results are already stored as Nifti imgs
def write_output(input_dictionary, output_dictionary, result):
    """Write the results of the optimization. Good job!"""
    
    affine = input_dictionary['images'][0].affine
    # write out results
    for param in output_dictionary.keys():
        if param in ['momentum', 'warp', 'invwarp']:
            img = nib.Nifti1Image(result[param], affine)
            nib.save(img, output_dictionary[param])
        elif param == 'objective':
            # TODO: save custom csv file instead
            # TODO: look up how to pickle out something
            pass
