import numpy as np
import vcalc
import regularizer_fftw_b
from matplotlib import pyplot as plt
import scipy.ndimage.filters as ndif


def computeDifferentialOperator(a, b, c, img, vox):
    img_lap = np.empty_like(img)
    img_lap[..., 0] = vcalc.divergence(vcalc.gradient(img[..., 0], vox), vox)
    img_lap[..., 1] = vcalc.divergence(vcalc.gradient(img[..., 1], vox), vox)

    img_gradiv = vcalc.gradient(vcalc.divergence(img, vox), vox)

    return -a*img_lap - b*img_gradiv + c*img

# Initialize regularizer
vox = np.array([1.0/128, 1.0/128])
a = 0.022
b = 0.0
c = 0.1
d = 2.0
_r = regularizer_fftw_b._differential()
_r._initialize(a, b, c, d, vox, (128, 128))

# Create test img/vector field
res = np.array([128, 128])
rngx = np.arange(0, res[0])/128.0
rngy = np.arange(0, res[1])/128.0

x, y = np.meshgrid(rngx, rngy, indexing='ij')
R = 2*np.pi
sn = np.sin
cs = np.cos

img = sn(R*x) + cs(R*y) - sn(R*y)*cs(R*x)
img = np.reshape(img, img.shape + (1,))
img = np.concatenate((img, img), axis=-1)

# Compute numerical differential operator
img_li_n = np.copy(img)
for i in range(int(d)):
    img_li_n = computeDifferentialOperator(a, b, c, img_li_n, vox)


# Compute gaussian smoothed image
img_g = np.empty_like(img)
img_g[..., 0] = ndif.gaussian_filter(img[..., 0], (30.0, 30.0), mode='wrap')
img_g[..., 1] = ndif.gaussian_filter(img[..., 1], (30.0, 30.0), mode='wrap')

# Compute NavierLame operator with regularizer
img_li_rc = _r.convolve(img)
img_li_rr = _r.regularize(img)

# display
fig = plt.figure('NL compare', figsize=(16, 10))
fig.add_subplot(2, 5, 1)
plt.imshow(img[..., 0])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 5, 2)
plt.imshow(img_li_n[..., 0])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 5, 3)
plt.imshow(img_li_rc[..., 0])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 5, 4)
plt.imshow(img_g[..., 0])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 5, 5)
plt.imshow(img_li_rr[..., 0])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 5, 6)
plt.imshow(img[..., 1])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 5, 7)
plt.imshow(img_li_n[..., 1])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 5, 8)
plt.imshow(img_li_rc[..., 1])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 5, 9)
plt.imshow(img_g[..., 1])
plt.axis('off')
plt.colorbar()
fig.add_subplot(2, 5, 10)
plt.imshow(img_li_rr[..., 1])
plt.axis('off')
plt.colorbar()

plt.show()
