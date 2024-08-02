import os
import tomopy
import dxchange
import matplotlib.pyplot as plt
import matplotlib
import logging
import cupy as cp
import sys
from cupy.fft import fft2, ifft2
import numpy as np

matplotlib.use('Agg') 

def bin_3d(array, binning_factorA, binning_factor):
    shape = array.shape
    new_shape = (shape[0] // binning_factorA, binning_factorA,
                 shape[1] // binning_factor, binning_factor,
                 shape[2] // binning_factor, binning_factor)
    binned_array = array.reshape(new_shape).sum(axis=(1, 3, 5))
    return binned_array

def bin_1d(array, binning_factor):
    shape = array.shape[0]
    new_shape = (shape // binning_factor, binning_factor)
    binned_array = array.reshape(new_shape).sum(axis=1)
    return binned_array


logging.basicConfig(level=logging.INFO)

# Paganin phase retrieval function and supporting functions
BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]

def _wavelength(energy):
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy

def paganin_filter(data, pixel_size=1e-4, dist=50, energy=20, db=1000, W=2e-4, pad=True):
    py, pz, val = _calc_pad(data, pixel_size, dist, energy, pad)
    dx, dy, dz = data.shape
    kf = _reciprocal_gridG(pixel_size, dy + 2 * py, dz + 2 * pz)
    phase_filter = cp.fft.fftshift(_paganin_filter_factorG(energy, dist, kf, pixel_size, db, W))

    prj = cp.full((dy + 2 * py, dz + 2 * pz), val, dtype=data.dtype)
    _retrieve_phase(data, phase_filter, py, pz, prj, pad)

    return data

def _retrieve_phase(data, phase_filter, px, py, prj, pad):
    dx, dy, dz = data.shape
    num_jobs = data.shape[0]
    normalized_phase_filter = phase_filter / phase_filter.max()

    for m in range(num_jobs):
        prj[px:dy + px, py:dz + py] = data[m]
        prj[:px] = prj[px]
        prj[-px:] = prj[-px-1]
        prj[:, :py] = prj[:, py][:, cp.newaxis]
        prj[:, -py:] = prj[:, -py-1][:, cp.newaxis]
        fproj = fft2(prj)
        fproj *= normalized_phase_filter
        proj = cp.real(ifft2(fproj))
        if pad:
            proj = proj[px:dy + px, py:dz + py]
        data[m] = proj

def _calc_pad(data, pixel_size, dist, energy, pad):
    dx, dy, dz = data.shape
    wavelength = _wavelength(energy)
    py, pz, val = 0, 0, 0
    if pad:
        val = _calc_pad_val(data)
        py = _calc_pad_width(dy, pixel_size, wavelength, dist)
        pz = _calc_pad_width(dz, pixel_size, wavelength, dist)

    return py, pz, val

def _paganin_filter_factorG(energy, dist, kf, pixel_size, db, W):
    aph = db * (dist * _wavelength(energy)) / (4 * PI)
    return 1 / (1.0 - (2 * aph / (W ** 2)) * (kf - 2))

def _calc_pad_width(dim, pixel_size, wavelength, dist):
    pad_pix = cp.ceil(PI * wavelength * dist / pixel_size ** 2)
    return int((pow(2, cp.ceil(cp.log2(dim + pad_pix))) - dim) * 0.5)

def _calc_pad_val(data):
    return cp.mean((data[..., 0] + data[..., -1]) * 0.5)

def _reciprocal_gridG(pixel_size, nx, ny):
    indx = cp.cos(_reciprocal_coord(pixel_size, nx) * 2 * PI * pixel_size)
    indy = cp.cos(_reciprocal_coord(pixel_size, ny) * 2 * PI * pixel_size)
    idx, idy = cp.meshgrid(indy, indx)
    return idx + idy

def _reciprocal_coord(pixel_size, num_grid):
    n = num_grid - 1
    rc = cp.arange(-n, num_grid, 2, dtype=cp.float32)
    rc *= 0.5 / (n * pixel_size)
    return rc



def pad360(cor, data):
	"""Pad data with 0 to handle 360 degrees scan"""

	if (cor < data.shape[1]//2):
	    # if rotation center is on the left side of the ROI
	    data[:] = data[:, :, ::-1]
	w = max(1, int(2*(data.shape[1]//2-cor)))

	# smooth transition at the border
	v = np.linspace(1, 0, w, endpoint=False)
	v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)
	data[:, :, -w:] *= v



# Main script
filename = '/local/data/alberto/DXC_/ESRF_Breast/AK176334_BIN2_4ms_1440p_2_/AK176334_BIN2_4ms_1440p_2_.h5'
proj, flat, dark, theta = dxchange.read_aps_32id(fname=filename)

print(proj.shape)
#proj = proj[0:1500,:,:]
#print(theta.shape)
#theta = theta[0:1500]

#print(theta.shape)
binning_factor = 1  # for example, change as needed

# Apply binning
#proj = bin_3d(proj, 1, binning_factor)
#flat = bin_3d(flat, 1, binning_factor)
#dark = bin_3d(dark, 1, binning_factor)
#theta = bin_1d(theta, binning_factor)




#proj = proj[:,300:500,:]
#flat = flat[:,300:500,:]
#dark = dark[:,300:500,:]
#print(proj.shape)
#sys.exit(0)
#Simulate wrong Flat

#filename = '/data/DXC_Alberto/FOAM/tomo_00094/mosaic_test_001.h5'
#_, flat, _, _ = dxchange.read_aps_32id(fname=filename)

#flat = flat[:,300:500,:]
# Display the first projection image
plt.imshow(proj[:, 0, :])
plt.show()

theta = None
# Generate theta if it is None
if theta is None:
    theta = tomopy.angles(proj.shape[0],360)

# Normalize and log the projections
proj = tomopy.normalize(proj, flat, dark)
proj = tomopy.minus_log(proj)

proj = tomopy.prep.stripe.remove_stripe_fw(
    proj,
    level=7,        # Number of discrete wavelet transform levels
    wname='db5',    # Type of the wavelet filter
    sigma=2,        # Damping parameter in Fourier space
    pad=True,       # Extend the size of the sinogram by padding with zeros
    ncore=4,        # Number of cores that will be assigned to jobs
    nchunk=None     # Chunk size for each core
)


# Apply Paganin phase retrieval
#proj = paganin_filter(cp.array(proj), pixel_size=1.75e-4, dist=10, energy=25.51, db=1100, W=3e-4, pad=True).get()


# Set the rotation center
#rot_center = 1224 tomo78
rot_center = 94//binning_factor #round robin
proj = pad360(rot_center, proj)
# Perform the reconstruction
recon = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec', sinogram_order=False)

# Apply a circular mask to the reconstructed images
recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)

# Display the first reconstructed slice
#plt.imshow(recon[0, :, :], cmap='gray')
#plt.show()

# Save the reconstructed images as TIFF files
data_radix = os.path.basename(filename).replace('.h5', '')
output_dir = os.path.join('./_rec', data_radix)
os.makedirs(output_dir, exist_ok=True)
output_dir = '/local/data/alberto/DXC_/ESRF_Breast/AK176334_BIN2_4ms_1440p_2_/_rec'
for i in range(recon.shape[0]):
    output_file = os.path.join(output_dir, f'recon_{i:05d}.tiff')
    print(output_file)
    dxchange.write_tiff(recon[i], fname=output_file, overwrite=True)

