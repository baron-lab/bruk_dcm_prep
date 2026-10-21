import numpy as np
from scipy.ndimage import convolve
from scipy.linalg import cholesky, solve_triangular
from scipy import fft
from scipy.signal import windows
import matplotlib.pyplot as plt

def findNoiseMask(im):
    # Find threshold for masking noise. 
    # Primary assumption: the bin with the most counts represents the peak of the noise histogram. This assumption is usually pretty safe for mri.
    # Complex sum over all dimensions beyond 3 to improve SNR and thus make separation of noise and signal easier
    if im.ndim>3:
        axes_to_sum = tuple(range(3, im.ndim))
        im_a = np.sum(im, axis=axes_to_sum)
    else:
        im_a = im
        
    # Take magnitude, converting histogram of noise to gamma distribution
    im_a = np.abs(im_a)

    # Compute histogram
    num_bins = int(min(10000, max(1, im_a.size // 200)))
    counts, edges = np.histogram(im_a, bins=num_bins)
    i_max = int(np.argmax(counts))
    n_max = counts[i_max]

    # MATLAB's i_max < 10 (1-based) corresponds to i_max < 9 (0-based)
    if i_max < 9 or n_max < 50:
        raise RuntimeError(
            f"Cannot find noise threshold (i_max={i_max}, n_max={n_max})."
        )

    thresh = edges[i_max] + edges[i_max + 1]

    # Enforce that local magnitude is less than threshold (3x3 for 2D, 3x3x3 for >=3D)
    n_spatial = min(3, im_a.ndim)
    kern_shape = (3,) * n_spatial + (1,) * (im_a.ndim - n_spatial)
    kern = np.ones(kern_shape, dtype=im_a.dtype) / (3 ** n_spatial)
    mask = convolve(im_a, kern, mode="constant", cval=0.0) < thresh

    return mask, thresh

def est_quad_coil_params(im, mask, make_qa_fig=True):
    """
    TODO: rewrite this description
    Estimates the pre-whitening matrix L and quadrature phase offset p_off
    from a two-channel complex image volume.

    Parameters
    ----------
    im_vol : np.ndarray (complex64 or complex128)
        Complex image array of shape (x, y, ..., 2), where the last axis
        represents the 2 quadrature coil channels. Usually a single 2D slice
        or 3D volume (x, y, slice, 2).
    make_qa_fig : bool, default=True
        If True, creates and returns a matplotlib Figure for QA inspection.

    Returns
    -------
    L : np.ndarray, shape (2, 2)
        Lower-triangular Cholesky pre-whitening matrix (normalized to L[0, 0]).
    p_off : float
        Estimated phase offset in radians between Channel 1 and Channel 2.
    w : np.ndarray, shape (2,)
        Fused complex weight vector `w = [1, exp(1j * p_off)] @ inv(L)` so the
        caller can combine pre-whitening and phase combination in a single step:
        `im_combined = w[0] * ch1 + w[1] * ch2`.
    fig : matplotlib.figure.Figure or None
        QA figure handle (or None if make_qa_fig is False).
    """
    # First dim of im should be receivers, followed by normal spatial dims then any extra dims after that
    if im.shape[0] != 2:
        raise ValueError(f"Expected first dimension of size 2 (channels), got {im.shape}")

    # Find voxel-wise phase difference between channels. 
    # For dims>3, complex averaging is done before angle operation (basically a signal weighted average of phase)
    p_off_all = im[0,...] * np.conj(im[1,...])
    if p_off_all.ndim>3:
        axes_to_sum = tuple(range(3, p_off_all.ndim))
        p_off_all = np.sum(p_off_all, axis=axes_to_sum)
    p_off_all = np.angle(p_off_all)
    p_valid = p_off_all[mask] 

    # Find the max bin of a histogram to get close to mean phase difference
    num_bins_p = int(np.round(np.sqrt(p_valid.size)))
    counts_p, edges_p = np.histogram(p_valid, bins=num_bins_p)
    maxI = int(np.argmax(counts_p))
    bin_width = edges_p[1] - edges_p[0]

    # Center the histogram to put phase wraps as far as possible from the peak
    shft = (maxI + 1) - int(np.round(num_bins_p / 2.0))
    bin_centers = 0.5 * (edges_p[:-1] + edges_p[1:])
    bin_vals = np.roll(bin_centers, shft)

    N = np.sum(counts_p)
    m = np.sum(bin_vals * counts_p) / N # mean
    v = np.sum(((bin_vals - m) ** 2) * counts_p) / N # variance
    s = np.sqrt(v) # standard deviation

    p_off = float(m + shft * bin_width)
    Crcvr = np.array([1, np.exp(-1j*p_off)], dtype=im.dtype) 

    if s > 1:
        print(
            "WARNING: Large variance in phase estimate across voxels. Double check that "
            "there is no spatial variation of phase, which can occur if using a "
            "two-channel coil that is not a quad coil."
        )

    # Plotting for QA
    mid_z = im.shape[3] // 2 # note that first dim is receivers
    slc_idx_im = (0, slice(None), slice(None), mid_z) + (0,) * (im.ndim - 4)
    slc_idx_p = (slice(None), slice(None), mid_z) 

    fig = None
    if make_qa_fig:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].imshow(np.abs(im[slc_idx_im]), cmap="gray", aspect="auto")
        axes[0].set_title("image")

        axes[1].imshow(mask[slc_idx_p], cmap="gray", aspect="auto")
        axes[1].set_title("mask used")

        im_plot = axes[2].imshow(p_off_all[slc_idx_p], aspect="auto")
        plt.colorbar(im_plot, ax=axes[2])
        axes[2].set_title("channel phase diff (masked).\nShould not spatially vary")
        plt.tight_layout()

    return Crcvr, fig


def est_receiver_sens(im, mask, fftDim, beta_val=30, make_qa_fig=True):
    # Finds receiver sensitivities 
    # First dim of im should be receivers, followed by normal spatial dims then any extra dims after that
    # Complex sum over all dimensions beyond 3 to improve SNR and thus make separation of noise and signal easier
    if im.ndim>4:
        axes_to_sum = tuple(range(4, im.ndim))
        im_a = np.sum(im, axis=axes_to_sum)
    else:
        im_a = im

    # Bruker often has offcenter data in k-space, so we center it before doing low pass filtering
    for nd in range(fftDim):
        im_a = fft.fft(im_a,axis=nd+1) # nd+1 because 1st dim is receivers
        ax_keep = nd+1
        im_test = np.sum(np.abs(im_a),
            axis=tuple(i for i in range(im_a.ndim) if i!= ax_keep))
        shft = -np.argmax(im_test)
        if shft != 0:
            im_a = np.roll(im_a, shift=shft, axis=nd+1)
        #im_a = fft.ifft(im_a,axis=nd+1)

    # Create low pass filter using Kaiser window
    win = np.multiply.outer(np.outer(
        windows.kaiser(im_a.shape[1],beta_val),
        windows.kaiser(im_a.shape[2],beta_val)),
        windows.kaiser(im_a.shape[3],beta_val))
    win = fft.ifftshift(win)
    win = win.astype(im_a.dtype)

    # Apply low pass filter, then return to image domain
    im_a = im_a * win
    im_a = fft.ifftn(im_a, axes=(1,2,3))

    # Find receivers
    rsos = np.sqrt(np.sum(np.abs(im_a)**2,0))
    Crcvr = im_a / rsos

    # Plotting for QA
    mid_z = im_a.shape[3] // 2 
    fig = None
    if make_qa_fig:
        fig, axes = plt.subplots(2, im_a.shape[0], figsize=(8, 8))
        for nr in range(im_a.shape[0]):
            axes[0,nr].imshow(np.abs(im_a[nr,:,:,mid_z]), cmap="gray", aspect="auto")
            axes[0,nr].set_title(f"lpf image, r{nr}")

            axes[1,nr].imshow(np.abs(Crcvr[nr,:,:,mid_z]), cmap="gray", aspect="auto")
            axes[1,nr].set_title(f"Receiver Sensitivity, r{nr}")
        plt.tight_layout()

    return Crcvr, fig


def combine_coils(im,combinetype,acqDim):
    # Combines coils. 
    # Assumes quad coil if combinetype==1. Otherwise, SENSE1 is used.
    #   If combinetype > 2, no prewhitening is performed

    if combinetype < 1 or combinetype > 3:
        raise ValueError(f"Unexpected combinetype received, got {combinetype}")
    
    # Find noise mask
    mask, thresh = findNoiseMask(im)
    
    # Move channel axis to front: (2, x, y, ...)
    im = np.moveaxis(im, -1, 0)

    # --- 1. Find noise covariance between channels and pre-whiten ---
    if combinetype < 3:
        # Extract noise
        nse = im[:, mask] 
        nse = nse.reshape(nse.shape[0],-1)

        psi = (nse @ nse.conj().T) / (nse.shape[1] - 1)
        L = cholesky(psi, lower=True)
        # Keep magnitudes relative to first channel. Othersize, normalizes such that noise std = 1.
            # Removed because it may actually good to have all images that go through this come out normalized to noise std.
        # L = L / np.real(L[0, 0])  
        del nse

        # Apply pre-whitening: L \ im(:,:)
        im_shape = im.shape
        im = solve_triangular(L, im.reshape(im.shape[0], -1), lower=True)
        im = np.reshape(im,im_shape)

    # Find receiver sensitivities
    if combinetype==1:
        Crcvr, fig = est_quad_coil_params(im,~mask)
    else:
        Crcvr, fig = est_receiver_sens(im,~mask,acqDim)

    # Apply sensitivites to combine receivers
    if Crcvr.ndim < im.ndim:
        Crcvr = np.expand_dims(Crcvr,tuple(range(Crcvr.ndim,im.ndim)))
    im = np.sum(np.conj(Crcvr) * im, axis=0)

    # TODO: need to return fig to be saved in calling fucntion

    return im