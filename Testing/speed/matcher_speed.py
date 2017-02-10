import numpy as np
import timeit



setup = """
import matcher
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

pPath = '/Users/gfleishman/Desktop/Preprocess4'
tmpPath = '/sc/FSLMathsnii.gz_3.OutputFile-04.nii.gz'
#refPath = '/sc/FSLMathsnii.gz_3.OutputFile-04.nii.gz'
#tmpPath = '/y2/FSLMathsnii.gz_4.OutputFile-04.nii.gz'
refPath = '/y2/FSLMathsnii.gz_4.OutputFile-04.nii.gz'

tmp = nib.load(pPath+tmpPath).get_data().squeeze()
ref = nib.load(pPath+refPath).get_data().squeeze()

#tmp = tmp[...,90]
#ref = ref[...,90]

tmp = tmp*(1.0/tmp.max())
ref = ref*(1.0/ref.max())

#m = matcher.matcher('SSD')
#m = matcher.matcher('CC')
m = matcher.matcher('CCL')
"""


#time distance
tr = (timeit.Timer("m.dist(ref, tmp)", setup).repeat(2,5))
print np.mean(tr)

#time force
tr = (timeit.Timer("m.force(ref, tmp)", setup).repeat(2,5))
print np.mean(tr)