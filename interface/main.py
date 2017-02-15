# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:28:04 2015

@author: gfleishman
"""

# file i/o imports
import sys
import nibabel as nib

# numerical imports
import numpy as np

# optimizer import
import pyrpl.optimization.static_grad_desc_lcl as optimizer

help_string = """

PYRT: version 0.1

Geodesic Regression in Diffeomorphisms for Neuroimage Data Analysis

This code will fit a geodesic of diffeomorphisms through an image
time series. The output will be the initial scalar momentum distribution
which fully parameterizes the geodesic.

USAGE

-help: \t\t display this message

Mandatory arguments:

-N: \t\t The number of images in the time series
-J: \t\t The file paths for the images in chronological order, separated
by a single space
-T: \t\t The time values for the corresponding images in units of years,
separated by a single space
-out: \t\t The directory where results will be written out (no backslash
on the end please)

Optional arguments:

-vox: \t\t The voxel dimensions of the images in meters, each dimension
separated by a single space. Default is '0.001 0.001 0.001'.
-pStep: \t\t The gradient descent step size for momentum optimization.
Default is '1.0e-13'.
-iStep: \t\t The gradient descent step size for the initial image
appearance. Default is '0.0' meaning no optimization.
-rat: \t\t The scalar weight on the image matching term. Default is
'1e5'.
-h: \t\t The number of time discretization steps. Default is 8.
-mew: \t\t The first Lame parameter to the differential operator, weights
the Laplacian term. Default is '1.5e-6'.
-lam: \t\t The second Lame parameter to the differential operator, weights
the divergence or identity term depending on choice of regularizer. Default
is '0.12'.
-k: \t\t The third differential operator parameter. Default is '2.0'.
-mType: \t\t The matching funcational used. Default is Local Correlation
Coefficient 'CCL'
-rType: \t\t The differential operator/regularizer used. Default is the
Laplacian plus Identity 'LaplacianPlusIdentity'.

-uf1: \t\t Use to write out the forward transformation at the endpoint
of the geodesic
-ub1: \t\t Use to write out the backward transformation at the initial
point of the geodesic
-dist: \t\t Use to write out the optimization values
-I0: Use to write out the initial image
-I1: Use to write out the final image at the end of the geodesic

"""


def parseInputArgs():

    # Help string
    if '-help' in sys.argv:
        print help_string
        sys.exit()

    # Mandatory arguments: # of images, image paths, time point values
    if '-N' in sys.argv:
        N = int(sys.argv[sys.argv.index('-N')+1])
    else:
        print "Must specify number of images in time series with -N"
        sys.exit()
    if '-J' in sys.argv:
        p = sys.argv[sys.argv.index('-J')+1]
        J1 = nib.load(p).get_data().squeeze()
        J = np.empty((N,) + J1.shape)
        J[0] = J1
        for i in range(1, N):
            p = sys.argv[sys.argv.index('-J')+i+1]
            J[i] = nib.load(p).get_data().squeeze()

    else:
        print "Must specify image paths with -J"
        sys.exit()
    T = np.empty((N,))
    if '-T' in sys.argv:
        for i in range(N):
            T[i] = float(sys.argv[sys.argv.index('-T')+i+1])
    else:
        print "Must specify image times with -T"
        sys.exit()
    T = T - T[0]
    for i in range(N):
        J[i] = J[i]*(1.0/np.mean(J[i][J[i] != 0]))

    # The Default parameter values (default for longitudinal)
    # cross sectional defaults: pStep -> 0.001
    #                           its -> [100, 1]
    params = {
            'vox': np.array([1., 1., 1.]),
            'pStep': 0.001,
            'iStep': 0.0,
            'rat': 5.0,
            'its': [1000, 1000],
            'res': [(128, 128, 128), J[0].shape],
            'h': 8,
            'a': 1.0,
            'b': 0.0,
            'c': 0.1,
            'd': 2.0,
            'mType': 'MI',
            'rType': 'differential'
            }

    # Check for optional parameters
    if '-2D' in sys.argv:
        if '-vox' not in sys.argv:
            print "To use -2D, you must specify a voxel size with -vox"
            sys.exit()
        axis = int(sys.argv[sys.argv.index('-2D')+1])
        slc = int(sys.argv[sys.argv.index('-2D')+2])

        sh = list(J[0].shape)
        del sh[axis]
        sh = tuple(sh)
        J2D = np.empty((N,) + sh)
        for i in range(N):
            if axis is 0:
                J2D[i] = J[i, slc, :, :]
            elif axis is 1:
                J2D[i] = J[i, :, slc, :]
            elif axis is 2:
                J2D[i] = J[i, :, :, slc]
        J = J2D
        params['res'] = [(128, 128), J[0].shape]
    if '-vox' in sys.argv:
        vox = []
        vox.append(float(sys.argv[sys.argv.index('-vox')+1]))
        vox.append(float(sys.argv[sys.argv.index('-vox')+2]))
        if '-2D' not in sys.argv:
            vox.append(float(sys.argv[sys.argv.index('-vox')+3]))
        params['vox'] = np.array(vox)
    if '-cs' in sys.argv:
        params['pStep'] = 0.1
        params['its'] = [100, 1]
    if '-pStep' in sys.argv:
        params['pStep'] = float(sys.argv[sys.argv.index('-pStep')+1])
    if '-iStep' in sys.argv:
        params['iStep'] = float(sys.argv[sys.argv.index('-iStep')+1])
    if '-rat' in sys.argv:
        params['rat'] = float(sys.argv[sys.argv.index('-rat')+1])
    if '-h' in sys.argv:
        params['h'] = int(sys.argv[sys.argv.index('-h')+1])
    if '-a' in sys.argv:
        params['a'] = float(sys.argv[sys.argv.index('-a')+1])
    if '-b' in sys.argv:
        params['b'] = float(sys.argv[sys.argv.index('-b')+1])
    if '-c' in sys.argv:
        params['c'] = float(sys.argv[sys.argv.index('-c')+1])
    if '-d' in sys.argv:
        params['d'] = float(sys.argv[sys.argv.index('-d')+1])
    if '-mType' in sys.argv:
        params['mType'] = sys.argv[sys.argv.index('-mType')+1]
    if '-rType' in sys.argv:
        params['rType'] = sys.argv[sys.argv.index('-rType')+1]

    return J, T, params


def parseOutputArgs():

    # Only mandatory argument
    if '-out' in sys.argv:
        outdir = sys.argv[sys.argv.index('-out') + 1]
    else:
        print "Must indicate a directory for output with -out"
        sys.exit()

    # Optional output flags
    output_flags = []
    if '-uf1' in sys.argv:
        output_flags.append('uf1')
    if '-ub1' in sys.argv:
        output_flags.append('ub1')
    if '-dist' in sys.argv:
        output_flags.append('dist')
    if '-I0' in sys.argv:
        output_flags.append('I0')
    if '-I1' in sys.argv:
        output_flags.append('I1')

    return outdir, output_flags


def writeOutput(outdir, output_flags, grdc):

    img = nib.Nifti1Image(grdc.P[0], np.eye(4))
    nib.save(img, outdir + '/P0.nii.gz')

    if 'uf1' in output_flags:
        img = nib.Nifti1Image(grdc.uf[-1, ..., 0], np.eye(4))
        nib.save(img, outdir + '/uf1_0.nii.gz')
        img = nib.Nifti1Image(grdc.uf[-1, ..., 1], np.eye(4))
        nib.save(img, outdir + '/uf1_1.nii.gz')
        if grdc.d == 3:
            img = nib.Nifti1Image(grdc.uf[-1, ..., 2], np.eye(4))
            nib.save(img, outdir + '/uf1_2.nii.gz')
    if 'ub1' in output_flags:
        img = nib.Nifti1Image(grdc.ub[-1, ..., 0], np.eye(4))
        nib.save(img, outdir + '/ub1_0.nii.gz')
        img = nib.Nifti1Image(grdc.ub[-1, ..., 1], np.eye(4))
        nib.save(img, outdir + '/ub1_1.nii.gz')
        if grdc.d == 3:
            img = nib.Nifti1Image(grdc.ub[-1, ..., 2], np.eye(4))
            nib.save(img, outdir + '/ub1_2.nii.gz')
    if 'I0' in output_flags:
        img = nib.Nifti1Image(grdc.Ifr[0], np.eye(4))
        nib.save(img, outdir + '/I0.nii.gz')
    if 'I1' in output_flags:
        img = nib.Nifti1Image(grdc.Ifr[-1], np.eye(4))
        nib.save(img, outdir + '/I1.nii.gz')


def main():

    # Get input arguments
    J, T, params = parseInputArgs()

    # Get output arguments
    outdir, output_flags = parseOutputArgs()

    # Fit geodesic
    grdc, P0_mag, data_match, grad_mag = optimizer.optimize(J, T, params)

    # save optimization information
    np.save(outdir + '/P0_mag.npy', P0_mag)
    np.save(outdir + '/data_match.npy', data_match)
    np.save(outdir + '/grad_mag.npy', grad_mag)

    # Write the output
    writeOutput(outdir, output_flags, grdc)

main()
