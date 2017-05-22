# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:28:04 2015

@author: gfleishman
"""
import os
import sys
import time
import numpy as np
import nibabel as nib

import pyrpl.optimization.static as optimizer
import pyrpl.image_tools.preprocessing as preproc
import pyrpl.image_tools.transformer as transformer


def main():

    # grab the input parameters
    reference_path = sys.argv[1]
    write_path_root = sys.argv[2]
    outer_iterations = int(sys.argv[3])

    # get the reference image, make an id string for it
    ref = nib.load(reference_path).get_data().squeeze()
    #ref = preproc.rescale_intensity(ref, mean=1.0)
    # currently assumes filenames are unique between inputs
    # consider generating a unique hash here, common suffix between references
    ref_str = reference_path.split('/')[-1].split('.')[0]

    # create transformer for later
    _t = transformer.transformer()

    # establish parameters for registration of template to reference
    params = {
            'vox': np.array([1.0, 1.0, 1.0]),
            'oIts': outer_iterations,
            'pStep': 0.025,
            'iStep': 0.0,
            'tStep': 1.0,
            'rat': 0.01,
            'its': [5],
            'res': [ref.shape],
            'h': 6,
            'a': 1.0,
            'b': 0.0,
            'c': 0.1,
            'd': 2.0,
            'mType': 'SSD',
            'rType': 'differential'
            }

    # number of template updates
    for o in range(outer_iterations):

        # keep cycling until the newest template is done
        tPath = write_path_root + '/template' + str(o) + '.nii.gz'
        tFound = False
        while not tFound:
            if os.path.exists(tPath):
                time.sleep(5)
                tFound = True
                tmp = nib.load(tPath).get_data().squeeze()
                tmp = preproc.rescale_intensity(tmp, mean=1.0)
            else:
                time.sleep(2)

        # Register the template to the reference image
        J = np.array([tmp, ref])
        T = np.array([0.0, 1.0])

        # Fit geodesic
        grdc, P0_mag, data_match, grad_mag = optimizer.optimize(J, T, params)
        print P0_mag
        print data_match
        print grad_mag

        # write out the momentum and transformations
        wPath = write_path_root + '/' + ref_str + '_momentum.npy'
        np.save(wPath, grdc.P[0])
        I0 = _t.applyTransform(ref, grdc.full_vox, grdc.uf[-1])
        wPath = write_path_root + '/' + ref_str + '_I0.npy'
        np.save(wPath, I0)

main()
