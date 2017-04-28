# -*- coding: utf-8 -*-
"""
Tools for preparing images for nonlinear registration. These are optional
preprocessing methods typically applied just after file input.

Author: Greg M. Fleishman
"""

import numpy as np
import scipy.ndimage as ndi


def shave(img):
    """Remove all boundary slices containing only background

    Assumes background voxels are 0, i.e. for masked images"""

    # spatial dimension of image
    dimension = len(img.shape)
    # place to store results
    bounds = np.empty((dimension, 2))

    # for each spatial dimension
    for i in range(dimension):
        # shave from the left
        j = -1
        s = 0
        while s == 0:
            j += 1
            slc = [slice(None, None, None),] * dimension
            slc[i] = slice(j, j+1, None)
            s = np.sum(img[slc])
        bounds[i, 0] = j
        # shave from the right
        j = img.shape[i]
        s = 0
        while s == 0:
            j -= 1
            slc = [slice(None, None, None),] * dimension
            slc[i] = slice(j, j+1, None)
            s = np.sum(img[slc])
        bounds[i, 1] = j
        
    return bounds


def cube(img, vox):
    """Adjust dimensions such that field of view is a cube

    Cube is with respect to the physical units, not number of voxels"""

    # compute on image copy
    imgc = np.copy(img)

    # field of view physical dimensions
    sh = imgc.shape
    d = len(sh)
    phys_dims = sh * vox

    # pad the array along the longest dimension
    if pad:
        i = np.argmax(phys_dims)
        p = [(0, 0)]*d
        p[i] = (pad, pad)
        imgc = np.pad(imgc, p, mode='constant')
        # update shape and physical dimensions
        sh = imgc.shape
        phys_dims = sh * vox

    # determine number of voxels to add to each side
    mx = phys_dims.max()
    diff = (mx - phys_dims) / (vox * 2.)
    diff = np.round(diff).astype(np.int)

    # pad the image to make cube
    pad = [(df, df) for df in diff]
    return np.pad(imgc, pad, mode='constant')


def obtain_foreground_mask(img):
    """Estimate a binary mask for foreground voxels

    This is a very rudimentary method based on heuristic thresholding and
    morphological operations. It will generally produce conservative masks
    that cut off some foreground on the boundary of the object. Also,
    the mask is not guaranteed to be free from holes, though quite a bit
    of morphological closing is done."""

    # obtain pdf over non-zero values
    mn = np.min(img[img > 0])
    mx = np.max(img)
    h, e = np.histogram(img, bins=np.ceil(mx-mn)+1, range=(mn, mx))
    h = h.astype(np.float)
    h *= (1.0/np.sum(h))
    # cdf will be useful later
    h_cdf = np.cumsum(h)

    # obtain first local minimum of pdf above 35th percentile
    # 35th percentile is a rough heuristic based on the relative volume
    # occumpied by the foreground normalized to the icbm template and
    # the background
    i = 0
    found = False
    while not found:
        if h[i+1] - h[i] > 0 and h_cdf[i] > 0.35:
            found = True
        i += 1

    # obtain intial background mask below threshold
    bg = np.zeros_like(img)
    bg[img < e[i+1]] = 1

    # clean up holes in mask
    # iterations=10 is also a heuristic
    bg = ndi.morphology.binary_fill_holes(bg)
    fg = ndi.morphology.binary_closing(~bg, iterations=10)
    return ndi.morphology.binary_fill_holes(fg)


def compress_intensity_range(img):
    """Compress the intensity range to eliminate intensity outliers

    Outliers are defined as intensities below the 2nd or above the 98th
    percentiles."""

    # compute on image copy
    imgc = np.copy(img)

    # obtain histogram cdf
    mn = np.min(imgc[imgc > 0])
    mx = np.max(imgc)
    h, e = np.histogram(imgc, bins=np.ceil(mx-mn)+1, range=(mn, mx))
    h = h.astype(np.float)
    h *= (1.0/np.sum(h))
    h_cdf = np.cumsum(h)

    # find 2nd and 98th percentiles and create outliers masks
    mn = np.argmax(h_cdf > 0.02)
    mx = np.argmax(h_cdf > 0.98)
    low_outliers = (imgc < e[mn])
    high_outliers = (imgc >= e[mx])
    zeros = (imgc == 0)

    # compress outliers
    imgc[low_outliers] = e[mn]
    imgc[high_outliers] = e[mx-1]

    # create mean filtered version of image
    fltr = np.ones((3, 3, 3)) * (1./27)
    filtered_img = ndi.filters.convolve(imgc, fltr, mode='constant')

    # replace outliers
    outliers = low_outliers + high_outliers
    imgc[outliers] = filtered_img[outliers]
    imgc[zeros] = 0
    return imgc


def histogram_match(ref, img, bins=2048):
    """Match the histogram of img to ref"""

    # Get reference histogram, and cdf
    refbg = ref[[slice(0, 1, None),]*len(ref.shape)]
    refhist, refe = np.histogram(ref[ref > refbg], bins=bins)
    refhist = refhist.astype(np.float)
    refhist = refhist*(1.0/np.sum(refhist))
    refcdf = np.cumsum(refhist)

    # Get image histogram, and cdf
    imgbg = img[[slice(0, 1, None),]*len(img.shape)]
    imghist, imge = np.histogram(img[img > imgbg], bins=bins)
    imghist = imghist.astype(np.float)
    imghist = imghist*(1.0/np.sum(imghist))
    imgcdf = np.cumsum(imghist)

    # Get transfer function and compute new image
    transfer_func = np.interp(imgcdf, refcdf, refe[:-1])
    out = np.interp(img, imge[:-1], transfer_func)
    out[img <= imgbg] = refbg
    out[out <= refbg] = refbg
    return out


def scale_intensity(img, mn=0., mx=0., mean=0.):
    """rescale image intensities"""

    # compute on image copy
    imgc = np.copy(img)

    # rescale appropriately
    if mn:
        return imgc * (mn / np.min(imgc[imgc != 0]))
    elif mx:
        return imgc * (mx / np.max(imgc[imgc != 0]))
    elif mean:
        return imgc * (mean / np.mean(imgc[imgc != 0]))
