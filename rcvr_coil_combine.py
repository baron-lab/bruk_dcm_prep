import numpy as np
from scipy.ndimage import convolve
from scipy.linalg import cholesky, solve_triangular, lstsq
from scipy import fft
from scipy.signal import windows
from scipy.optimize import minimize_scalar
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

    # Check for too few or too many bins
    ntry = 0
    while i_max < 9: 
        if n_max < 50 or ntry>10:
            raise RuntimeError(
                f"Cannot find noise threshold (i_max={i_max}, n_max={n_max})."
            )
        num_bins = num_bins*2
        counts, edges = np.histogram(im_a, bins=num_bins)
        i_max = int(np.argmax(counts))
        n_max = counts[i_max]
        ntry = ntry+1

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

def process_repetitions(im,fftDim, bval, beta_val=30, make_qa_fig=True):
    # Combines repetitions. TODO: this should only correct drift, because I think that combining reps should only be done after denoising, because I don't trust phase removal...
    # Performs phase drift correction. Estimates magnitude drift for QA
    # Assumes PE on 2nd dim (i.e., dim=1)

    if im.ndim < 5 or im.shape[4] < 2:
        print("WARNING (combine_repetitions): multiple repetitions no found")
        return im

    # Bruker often has offcenter data in k-space, so we center it before doing anything else
    if np.iscomplexobj(im):
        for nd in range(fftDim):
            im = fft.fft(im,axis=nd) 
            ax_keep = nd
            im_test = np.sum(np.abs(im),
                axis=tuple(i for i in range(im.ndim) if i!= ax_keep))
            shft = -np.argmax(im_test)
            if shft != 0:
                im = np.roll(im, shift=shft, axis=nd)
            im = fft.ifft(im,axis=nd)

    # Find noise mask, then get signal magnitudes for all to assess signal drift
    mask, thresh = findNoiseMask(im)
    sigVals = np.zeros(im.shape[3:5])
    for n3 in range(im.shape[3]):
        for n4 in range(im.shape[4]):
            im_a = im[:,:,:,n3,n4]
            im_a = np.abs(im_a[~mask])
            sigVals[n3,n4] = np.mean(im_a) #np.percentile(im_a,80). Percentile would be more robust, but it is way slower than mean

    # Check for smallest bval, if this is diffusion weighted
    if len(bval) > 0:
        bval = np.array(bval)
        b0ind_logical = bval < bval.min() + 50 # TODO: make sure this is okay for ex vivo
        b0ind = np.where(b0ind_logical)[0]
        b0ref = b0ind[-1] # use the last one as reference b/c eddies more stable by end

    # Plotting signal drift
    fig = []
    if make_qa_fig:   
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))     
        if len(bval) > 0:
            
            n = np.arange(len(b0ind))
            for n4 in range(im.shape[4]):
                axes[0,1].plot(n+len(n)*n4,100*sigVals[b0ind,n4]/sigVals[b0ind[0],0])
            axes[0,1].set_title(f"mean signal across b0 images\n and repetitions (colors)\n NOT CORRECTED")
            axes[0,1].set_xlabel('b0 number')
            axes[0,1].set_ylabel('signal (percent)')
        n = np.arange(im.shape[3])
        for n4 in range(im.shape[4]):
            axes[0,0].plot(n+n4*im.shape[3],sigVals[:,n4])
        axes[0,0].set_title(f"mean signal across dim 4\n and repetitions (colors)\n NOT CORRECTED")
        axes[0,0].set_xlabel('index in dim 4')
        axes[0,0].set_ylabel('signal (raw)')

    # For now we only correct drift for dwi. Should probably do it for any EPI, but for now this code is mostly used for dti
    if len(bval) == 0:
        print("WARNING (combine_repetitions): not DWI; skipping drift image shift corrections.")
        return im

    # Check for complext data before continuing
    if not np.iscomplexobj(im):
        print("WARNING (combine_repetitions): no complex data. Will not assess shifts from drift.")
        return im

    # Loop through all dwi to find drift for each case
    opt = {"xtol": 0.01}
    # Create basic phase ramp
    rmp = fft.ifftshift(np.arange(-im.shape[1]//2, im.shape[1]//2, 1, dtype=im.dtype)) * 3.14 / (im.shape[1]/2)
    rmp = rmp[None, :, None]
    im_a = im
    # Do the looping for repetitions 
    if im.shape[4]>1:
        shft_all = np.zeros((im.shape[3], im.shape[4]-1))
        for n3 in range(im.shape[3]):
            im_f = fft.fft(im[:,:,:,n3,0:-1],axis=1) # use LAST repetition as reference, since eddies more stable then
            for nr in range(im.shape[4]-1):
                res = minimize_scalar(lambda x: get_cost(x, im[:,:,:,n3,-1], im_f[:,:,:,nr], rmp),
                    bracket=(-2,0,2),options=opt)
                #im_a[:,:,:,n3,nr] = fft.ifft(im_f[:,:,:,nr] * np.exp(1j*res.x*rmp), axis=1) # TODO: should actually fit a line or something to all shifts, because this will be noisy.
                shft_all[n3,nr] = res.x

    # Find shifts of b0 images (for DWI)  
    if len(bval)>0:
        shft_b0 = np.zeros((np.sum(b0ind_logical), im.shape[4]))
        for nb0 in range(len(b0ind)):
            for nr in range(im.shape[4]):
                im_f = fft.fft(im[:,:,:,b0ind[nb0],nr],axis=1)
                res = minimize_scalar(lambda x: get_cost(x, im[:,:,:,b0ref,-1], im_f, rmp),
                    bracket=(-2,0,2),options=opt)
                shft_b0[nb0,nr] = res.x

    # Fit a polynomial to the shifts
    shifts_corr = []
    if len(bval) > 0:
        # Jointly fit using b0 results and results across repetitions
        # Start building forward matrix for Ax=b with b0 relative to each other
          # here, x = [a, b, c, d] in the equation shift = a+b*n+c*n^2+d*n^3, where n is the index of the volume
        max_o = min((im.shape[3]-1)//3, 2)
        n = b0ind
        for nr in range(1,im.shape[4]):
            n = np.concat((n,nr*im.shape[3]+b0ind), axis=0)
        b0ind_all = n
        A = np.ones(n.shape)
        for o in range(1,max_o+1):
            A = np.column_stack((A, n**o))
        b = np.reshape(shft_b0.T,(-1))
        w = np.ones(b.shape) # weights for weighted least squares
        # Get all indices for plotting, and for working in repetitions
        n_all = np.arange(im.shape[3])
        n_r = n_all 
        for nr in range(1,im.shape[4]):
            n_next = nr*im.shape[3]+np.arange(im.shape[3])
            n_all = np.concat((n_all,n_next), axis=0)
            n_r = np.column_stack((n_r, n_next))
        # Work in repetitions. Here we exploit that shift between repetitions is the difference of baseline shifts
        if im.shape[4]>1:  
            A2 = np.zeros(n_r.shape[0]*(n_r.shape[1]-1))
            for o in range(1,max_o+1):
                A2 = np.column_stack((A2, np.reshape(n_r[:,0:-1]**o-n_r[:,-1:]**o, (-1))))
            b2 = np.reshape(shft_all,(-1))
            w2 = np.reshape(sigVals[:,0:-1], (-1))
            w2 = w2/np.sum(w2)*np.sum(w) # net weighting here equivalent to b0 shifts weighting
            A = np.concat((A,A2),axis=0)
            b = np.concat((b,b2),axis=0)
            w = np.concat((w,w2),axis=0)
        elif b0ind[0]>0 or b0ind[-1]<im.shape[3]-2:
            # If b0 images are too far from each end, will have to extrapolate, so stick to linear
            A = A[:,0:2]
        # Solve
        x = lstsq(A*w[:,None], b*w)
        x = x[0]
        shifts_corr = x[0]
        for o in range(1,A.shape[1]):
            shifts_corr = shifts_corr + x[o]*n_all**o
        # Plot
        if make_qa_fig:
            axes[1,1].plot(n_all,shifts_corr)
            axes[1,1].scatter(b0ind_all,np.reshape(shft_b0.T,(-1)))
            axes[1,1].set_title(f"image shifts from drift across b0s\n CORRECTED")
            axes[1,1].set_xlabel('volume number')
            axes[1,1].set_ylabel('voxels')
            # Across repetitions
            if im.shape[4] > 1:
                n_r = np.arange(im.shape[3])
                y2 = np.reshape(shifts_corr,(im.shape[4],-1))
                for nr in range(im.shape[4]-1):       
                    axes[1,0].plot(y2[nr,:]-y2[-1,:])
                axes[1,0].plot(shft_all,ls=':')
                axes[1,0].set_title(f"image shifts for each repetition\n relative to last repetition")
                axes[1,0].set_xlabel('volume number')
                axes[1,0].set_ylabel('voxels')
            plt.tight_layout()
        shifts_corr = np.reshape(shifts_corr, (im.shape[4],im.shape[3]))
    elif im.shape[4]>1:
        # Here we just fit each repetition separately, if there is enough values for fitting 
          # Fits are weighted by mean signal amplitude squared.
        # Otherwise just directly use each shift found
        max_o = min((im.shape[3]-1)//3, 2)
        if max_o > 0:
            n = np.arange(im.shape[3])
            A = np.ones(n.shape)
            w = sigVals[:,-1] # Assumes reference was the last repetition
            x = np.zeros((max_o+1,shft_all.shape[1]))
            for o in range(1,max_o+1):
                A = np.column_stack((A,n**o))
            shifts_corr = np.zeros((0,im.shape[3]))
            for nr in range(shft_all.shape[1]):
                b = shft_all[:,nr]
                x_a = lstsq(A*w[:,None], b*w)
                x[:,nr] = x_a[0]
                # Plot
                y = x[0,nr]
                for o in range(1,A.shape[1]):
                    y = y + x[o,nr]*n**o
                shifts_corr = np.concat((shifts_corr,y[None,:]),axis=0)
                if make_qa_fig: 
                    axes[1,0].plot(n,y)
                    axes[1,0].plot(n,shft_all[:,nr],ls=':')
                    axes[1,0].set_title(f"image shifts for each repetition\n relative to last repetition")
                    axes[1,0].set_xlabel('volume number')
                    axes[1,0].set_ylabel('voxels')
                    plt.tight_layout()
            shifts_corr = np.concat((shifts_corr,np.zeros((1,im.shape[3]))),axis=0)
        else:
            shifts_corr = np.concat((shft_all.T, np.zeros((1,im.shape[3]))), axis=0)

    # Apply shifts. 
    if shifts_corr.size > 0:
        shifts_corr = shifts_corr.astype(im.dtype)
        for n3 in range(im.shape[3]):
            for n4 in range(im.shape[4]):
                if shifts_corr[n4,n3] > 0.01:
                    im_f = fft.fft(im[:,:,:,n3,n4],axis=1)
                    cor = np.exp(1j*shifts_corr[n4,n3]*rmp)
                    im[:,:,:,n3,n4] = fft.ifft(im_f * cor, axis=1)

    return im, fig
    
    
def get_cost(shft, im_ref, im_f_in, rmp):
    im_t = fft.ifft(im_f_in * np.exp(1j*shft*rmp), axis=1)
    cost = np.sum(np.abs(im_t) * np.abs(im_ref))
    return -cost