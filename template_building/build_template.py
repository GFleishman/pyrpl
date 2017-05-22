# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:14:20 2015

@author: gfleishman
"""

import os
import sys
import time
import glob
import numpy as np
import nibabel as nib

import pyrpl.models.gsid as model
import pyrpl.image_tools.vcalc as vcalc


def main():

    # grab the input parameters
    initial_template_path = sys.argv[1]
    write_path_root = sys.argv[2]
    outer_iterations = int(sys.argv[3])
    N = int(sys.argv[4])

    # get the current template image, write it out
    T = nib.load(initial_template_path).get_data().squeeze()
    Tout = nib.Nifti1Image(T, np.eye(4))
    nib.save(Tout, write_path_root + '/template0.nii.gz')

    # a place to store the full momentum for the warp of the initial template
    P0_full = np.zeros_like(T)

    # establish particulars for shooting template
    params = {
            'vox': np.array([1.0, 1.0, 1.0]),
            'oIts': outer_iterations,
            'pStep': 1e-8,
            'iStep': 0.0,
            'tStep': 1.0,
            'rat': 0.01,
            'its': [5],
            'res': [T.shape],
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

        # cycle until all momenta are finished
        allMomentaFound = False
        while not allMomentaFound:
            mPaths = glob.glob(write_path_root + '/*_momentum.npy')
            iPaths = glob.glob(write_path_root + '/*_I0.npy')
            if len(mPaths) != N or len(iPaths) != N:
                time.sleep(2)
            else:
                time.sleep(5)
                allMomentaFound = True

        # average the momenta
        avgP0 = 0
        for i in range(N):
            avgP0 += np.load(mPaths[i])
        avgP0 *= 1.0/N
        # increment total momentum
        P0_full = P0_full + avgP0

        # average the images
        avgI0 = 0
        for i in range(N):
            avgI0 += np.load(iPaths[i])
        avgI0 *= 1.0/N

        # clear the momenta and images files
        map(os.remove, mPaths)
        map(os.remove, iPaths)

        # geodesic shoot the sharp template with cumulative average momentum
        J = np.array([T, T])
        tm = np.array([0.0, 1.0])
        # TODO: consider creating this only once outside outer_its loop
        gr = model.gsid(J, tm, params)
        gr.resample(params['res'][-1])
        gr.dc.P[0] = P0_full

        # ensure CFL condition is satisfied
        dI = vcalc.gradient(T, params['vox'])
        v = - gr._r.regularize(dI * P0_full)
        gr.dc.cfl_nums[0] = abs(v * gr.dc.T[-1]/params['vox']).max()
        gr.dc.satisfy_cfl()

        # integrate and write out results
        gr.evaluate()
        Tout = nib.Nifti1Image(gr.dc.I[-1], np.eye(4))
        nib.save(Tout, write_path_root + '/template' + str(o+1) + '.nii.gz')

        # geodesic shoot the averaged template with the average momentum
        params['h'] = 6
        J = np.array([avgI0, avgI0])
        tm = np.array([0.0, 1.0])
        gr = model.geodesic_regression_in_diffeomorphisms(J, tm, params)
        gr.resample(params['res'][-1])
        gr.dc.P[0] = avgP0

        # ensure CFL condition is satisfied
        dI = vcalc.gradient(avgI0, params['vox'])
        v = - gr._r.regularize(dI * avgP0)
        gr.dc.cfl_nums[0] = abs(v * gr.dc.T[-1]/params['vox']).max()
        gr.dc.satisfy_cfl()

        # integrate and write out results
        gr.evaluate()
        Tout = nib.Nifti1Image(gr.dc.I[-1], np.eye(4))
        nib.save(Tout, write_path_root + '/templateIA' + str(o+1) + '.nii.gz')

main()
