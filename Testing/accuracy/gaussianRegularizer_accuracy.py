import numpy as np
import regularizer
# import regularizer_fftw
from matplotlib import pyplot as plt

# Create test img/vector field
res = np.array([128, 128])
resf = res.astype(np.float)

rngx = np.arange(0, res[0])/resf[0]
rngy = np.arange(0, res[1])/resf[1]
vox = np.ones(2)/resf

x, y = np.meshgrid(rngx, rngy, indexing='ij')
R = 2*np.pi
sn = np.sin
cs = np.cos

img = sn(R*x) + cs(R*y) - sn(R*y)*cs(R*x)
img = np.reshape(img, img.shape + (1,))
img = np.concatenate((img, img), axis=-1)

# Initialize regularizer
sig = 50.0
trn = 4.0
_r = regularizer._gaussian()
# _r = regularizer_fftw.gaussian()
_r._initialize(sig, trn, vox)

# regularize
img_li_rr = _r.regularize(img)

# display original image and smoothed version
fig = plt.figure('Gaussian regularizer results', figsize=(16, 10))
fig.add_subplot(2, 2, 1)
plt.imshow(img[..., 0])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 2, 2)
plt.imshow(img_li_rr[..., 0])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 2, 3)
plt.imshow(img[..., 1])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 2, 4)
plt.imshow(img_li_rr[..., 1])
plt.axis('off')
plt.colorbar()
plt.show()
