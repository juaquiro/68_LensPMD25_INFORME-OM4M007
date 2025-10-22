
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Dict

# -----------------------------
# Utilities
# -----------------------------
def _force_sync(x=None):
    """
    Force all pending TF work to finish without using tf.experimental.async_wait().
    We do it by materializing a tiny tensor to host (blocking until kernels complete).
    """
    if x is None:
        _ = tf.constant(0.0).numpy()
    else:
        _ = tf.reduce_sum(tf.cast(x, tf.float32)).numpy()

def _fftshift2d(x):
    h = tf.shape(x)[-2]
    w = tf.shape(x)[-1]
    return tf.roll(tf.roll(x, shift=h//2, axis=-2), shift=w//2, axis=-1)

def _gaussian_window_tf(N:int, M:int, sigma_y:float=None, sigma_x:float=None, dtype=tf.float32):
    if sigma_x is None: sigma_x = M/3.0
    if sigma_y is None: sigma_y = N/3.0
    y = tf.range(N, dtype=dtype) - tf.cast(N//2, dtype)
    x = tf.range(M, dtype=dtype) - tf.cast(M//2, dtype)
    Y, X = tf.meshgrid(y, x, indexing='ij')
    W = tf.exp(-0.5*(X**2)/(sigma_x**2) - 0.5*(Y**2)/(sigma_y**2))
    return W  # (N,M)

def _mat2gray_tf(x, mask=None):
    if mask is not None:
        x_masked = tf.boolean_mask(x, mask)
        mn = tf.reduce_min(x_masked)
        mx = tf.reduce_max(x_masked)
    else:
        mn = tf.reduce_min(x); mx = tf.reduce_max(x)
    rng = tf.maximum(mx - mn, tf.cast(1e-12, x.dtype))
    return (x - mn) / rng

# -----------------------------
# TF feature extractor (GPU)
# -----------------------------

def tf_calc_feature_batch(B: tf.Tensor, featureName: str = "feature_projected_DFT"):
    """
    B: (N,M,L) float32 -> X_sel: (L,F) float32, S
    Exact TF reimplementation of ml_spatialfreq_utils.calc_feature_batch (NumPy) so A/B matches.
    """
    assert featureName in ("feature_GV","feature_DFT","feature_projected_DFT")
    assert B.shape.rank == 3, "B must be (N,M,L)"
    B = tf.cast(B, tf.float32)
    N = tf.shape(B)[0]; M = tf.shape(B)[1]; L = tf.shape(B)[2]

    # ----- Build spatial Gaussian window with 1-based centering like NumPy -----
    # NumPy: x0 = floor(M/2)+1 ; y0 = floor(N/2)+1 ; x = [1..M]-x0 ; y = [1..N]-y0
    x0 = (tf.shape(B)[1] // 2) + 1
    y0 = (tf.shape(B)[0] // 2) + 1
    xx = tf.range(1, tf.shape(B)[1] + 1, dtype=tf.int32)
    yy = tf.range(1, tf.shape(B)[0] + 1, dtype=tf.int32)
    XX, YY = tf.meshgrid(xx, yy, indexing='xy')
    x = tf.cast(XX - x0, tf.float32)
    y = tf.cast(YY - y0, tf.float32)
    sigma_x = tf.cast(tf.shape(B)[1], tf.float32) / 3.0
    sigma_y = tf.cast(tf.shape(B)[0], tf.float32) / 3.0
    W2 = tf.exp(-0.5*(x**2)/(sigma_x**2) - 0.5*(y**2)/(sigma_y**2))  # (N,M)

    # ----- DC removal on raw patch, then apply spatial window -----
    mu_sp = tf.reduce_mean(B, axis=(0,1), keepdims=True)   # (1,1,L)
    B0    = B - mu_sp
    W     = tf.expand_dims(W2, axis=-1)                    # (N,M,1)
    Bwin  = B0 * W

    if featureName == "feature_GV":
        # GV features computed on Bwin (match NumPy)
        eps_ = tf.constant(1e-12, tf.float32)
        muGV = tf.reduce_mean(Bwin, axis=(0,1))            # (L,)
        E2   = tf.reduce_mean(Bwin*Bwin, axis=(0,1))       # (L,)
        sigmaGV = tf.sqrt(tf.maximum(E2 - muGV*muGV, eps_))
        X = tf.reshape(tf.transpose(Bwin, perm=[2,0,1]), [L, -1])  # (L,N*M)
        X_sel = (X - muGV[:,None]) / sigmaGV[:,None]
        return tf.cast(X_sel, tf.float32), {"W": W2}

    # ----- FFT over spatial dims, zero DC, fftshift, then frequency weighting -----
    G = tf.signal.fft2d(tf.cast(Bwin, tf.complex64))       # (N,M,L)
    # Zero DC bin (row 0, col 0) per patch BEFORE fftshift (as NumPy does)
    # Easiest way: overwrite slice
    # Build an index mask to set [0,0,:] = 0
    zeros_dc = tf.tensor_scatter_nd_update(
        G,
        indices=tf.concat([
            tf.zeros([tf.shape(G)[2], 1], tf.int32),   # row=0
            tf.zeros([tf.shape(G)[2], 1], tf.int32),   # col=0
            tf.reshape(tf.range(tf.shape(G)[2], dtype=tf.int32), [-1,1])  # page
        ], axis=1),
        updates=tf.zeros([tf.shape(G)[2]], dtype=G.dtype)
    )
    G = zeros_dc
    # Center spectrum
    G = _fftshift2d(G)
    # Multiply by SAME spatial Gaussian window (frequency weighting) like NumPy
    G = G * tf.cast(W, G.dtype)

    # ----- Magnitude, normalization over FULL spectrum -----
    G_abs = tf.math.abs(G)                          # (N,M,L)
    GNorm = tf.reduce_sum(G_abs, axis=(0,1))        # (L,)

    # ----- Semiplane wx >= 0: columns x0..M in 1-based -> zero-based [x0-1 : M) -----
    start = x0 - 1                                  # zero-based
    abs_G_sp = G_abs[:, start:, :]                  # (N, Wxp, L)
    Wxp = tf.shape(abs_G_sp)[1]

    # ----- Choose top vs bottom half per patch by energy -----
    # Top rows: [0 : y0-1) ; Bottom rows: [y0 : N)
    sumTop    = tf.reduce_sum(abs_G_sp[0:(y0-1), :, :], axis=(0,1))    # (L,)
    sumBottom = tf.reduce_sum(abs_G_sp[y0:, :, :], axis=(0,1))         # (L,)
    isTopGreater = tf.greater(sumTop, sumBottom)                        # (L,)

    # Build two candidates, select per page
    abs1 = tf.identity(abs_G_sp)                                        # zero TOP when ~isTopGreater
    abs1 = tf.tensor_scatter_nd_update(abs1,
        indices=tf.stack([tf.range(0, y0-1, dtype=tf.int32)], axis=-1),
        updates=tf.zeros_like(abs1[0:(y0-1), :, :]))
    abs2 = tf.identity(abs_G_sp)                                        # zero BOTTOM when isTopGreater
    abs2 = tf.tensor_scatter_nd_update(abs2,
        indices=tf.stack([tf.range(y0, tf.shape(abs2)[0], dtype=tf.int32)], axis=-1),
        updates=tf.zeros_like(abs2[y0:, :, :]))

    # Select per L: abs_sel[:,:,k] = abs2 if isTopGreater[k] else abs1
    # Vectorized selection:
    mask = tf.reshape(tf.cast(isTopGreater, abs1.dtype), [1,1,-1])
    abs_sel = abs1*(1.0 - mask) + abs2*mask                              # (N,Wxp,L)

    eps_ = tf.constant(1e-12, tf.float32)
    GNorm_safe = tf.maximum(GNorm, eps_)                                  # (L,)

    if featureName == "feature_DFT":
        X = tf.reshape(tf.transpose(abs_sel, perm=[2,0,1]), [L, -1])     # (L, N*Wxp)
        X_sel = X / GNorm_safe[:,None]
    else:
        G_sp_XP = tf.reduce_sum(abs_sel, axis=0)                          # (Wxp,L)
        G_sp_YP = tf.reduce_sum(abs_sel, axis=1)                          # (N,L)
        X = tf.concat([tf.transpose(G_sp_XP), tf.transpose(G_sp_YP)], axis=1)  # (L, Wxp+N)
        X_sel = X / GNorm_safe[:,None]

    return tf.cast(X_sel, tf.float32), {"W": W2}

    
# -----------------------------
# GPU-optimized pipeline with per-phase timing
# -----------------------------
def calc_spatial_freqs_supervised_regression_batch_gpu(
    g: np.ndarray,
    trained_model,
    feature_name: str,
    M_ROI: Optional[np.ndarray] = None,
    return_timing: bool = True,
):
    """
    Returns: (w_phi, phi_x, phi_y, theta, QM, M_proc, timings_dict if return_timing)
    """
    import time

    if M_ROI is None:
        M_ROI = np.ones_like(g, dtype=bool)
    M_ROI = M_ROI.astype(bool)

    model, scaler, meta = trained_model.load()
    N = trained_model.DB_info.patch_NR
    M = trained_model.DB_info.patch_NC
    r = N//2; c = M//2  # center offset

    NR, NC = g.shape

    #SECTION 1 BEGIN GPU optimized feature calculation NOT WORKING
    ######################################################################################
    #######################################################################################
    #######################################################################################

    
    # Compute M_proc (valid centers inside ROI and borders)
    M_borders = np.zeros_like(M_ROI, dtype=bool)
    M_borders[r:NR-(N-r), c:NC-(M-c)] = True
    M_proc = (M_ROI & M_borders).astype(np.float32)
    ir, jc = np.where(M_proc > 0)
    L = ir.size

    if L == 0:
        nanmap = np.full_like(g, np.nan, dtype=np.float32)
        out = (nanmap, nanmap, nanmap, nanmap, np.zeros_like(g, np.float32), np.zeros_like(g, np.float32))
        return (*out, {"total": 0.0, "patch": 0.0, "feature": 0.0, "predict": 0.0}) if return_timing else out

    timings = {}

    # Phase 1: Patch extraction (GPU via tf.image.extract_patches)
    t0 = time.perf_counter()
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)[None, ..., None]
    ksizes = [1, N, M, 1]
    strides = [1, 1, 1, 1]
    rates   = [1, 1, 1, 1]
    patches_all = tf.image.extract_patches(images=g_tf, sizes=ksizes, strides=strides, rates=rates, padding='VALID')
    patches_all = tf.reshape(patches_all, [NR - N + 1, NC - M + 1, N, M])
    # gather valid centers
    top = ir - r; left = jc - c
    inds = tf.stack([tf.convert_to_tensor(top, tf.int32), tf.convert_to_tensor(left, tf.int32)], axis=1)
    patches_gather = tf.gather_nd(patches_all, inds)  # (L,N,M)
    B = tf.transpose(patches_gather, perm=[1,2,0])    # (N,M,L)
    _force_sync(B)
    timings["patch"] = time.perf_counter() - t0

    # Phase 2: Feature extraction (GPU)
    t1 = time.perf_counter()
    # Map legacy "feature_normalized_DFT" to "feature_DFT"
    feat_name = 'feature_DFT' if feature_name == 'feature_normalized_DFT' else feature_name

    #AQDEBUG 22OCT25 here Xs is a tensorlow tensor
    Xs, _ = tf_calc_feature_batch(B, feat_name)
    #AQDEBUG 22OCT25 use numpy version for testing
    from ml_spatialfreq_utils import calc_feature_batch
    X_np, _=calc_feature_batch(B.numpy(), feat_name)
    Xs_np = scaler.transform(X_np)
    Xs=tf.convert_to_tensor(Xs_np, dtype=tf.float32)


    # Optional scaler (if you persisted mean/scale in meta)
    if 'scaler_mean' in meta and 'scaler_scale' in meta:
        mean = tf.convert_to_tensor(meta['scaler_mean'], dtype=tf.float32)
        scale = tf.convert_to_tensor(meta['scaler_scale'], dtype=tf.float32)
        Xs = (Xs - mean) / tf.maximum(scale, 1e-8)
    _force_sync(Xs)
    
    timings["feature"] = time.perf_counter() - t1


    

    #SECTION 2 ENDS GPU optimized feature calculation  NOT WORKING
    ######################################################################################
    #######################################################################################
    #######################################################################################

    #SECTION 2 BEGIN NON GPU optimized feature calculation 
    ######################################################################################
    #######################################################################################
    #######################################################################################

    """
    
    from ml_spatialfreq_utils import calc_feature_batch
    timings = {}

    t0 = time.perf_counter()
    timings["patch"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    
    # Compute valid centers where an N x M patch fully lies in ROI.
    # We'll use convolution with an N x M ones kernel to count ROI pixels per center.
    from scipy.signal import convolve2d
    Mvalid_counts = convolve2d(M_ROI.astype(np.uint8), np.ones((N,M), dtype=np.uint8), mode="same", boundary="fill")
    full_area = int(N*M)
    valid_centers = (Mvalid_counts == full_area)

    # Avoid borders explicitly to keep N x M fully inside image bounds
    if r > 0:
        valid_centers[:r, :] = False
        valid_centers[-r:, :] = False
    if c > 0:
        valid_centers[:, :c] = False
        valid_centers[:, -c:] = False

    ir, jc = np.where(valid_centers)
    L = len(ir)
    if L == 0:
        z = np.zeros_like(g, dtype=np.float32)
        return z, z, z, z, np.zeros_like(g, bool), M_ROI.astype(float)

    # Extract patches into stack B (N x M x L)
    B = np.zeros((N, M, L), dtype=np.float32)
    for k in range(L):
        i = ir[k]; j = jc[k]
        B[:,:,k] = g[i-r:i-r+N, j-c:j-c+M]

    # Features
    fname = feature_name if feature_name != "feature_normalized_DFT" else "feature_DFT"
    X, _S = calc_feature_batch(B, fname)

    #sanity check between numpy solution and tensor flow solution
    X_tf, S_tf = tf_calc_feature_batch(tf.convert_to_tensor(B, tf.float32), 'feature_projected_DFT')
    print(np.allclose(X, X_tf.numpy(), rtol=1e-4, atol=1e-5))

    
    
    # Scale if scaler exists
    if scaler is not None:
        try:
            Xs = scaler.transform(X)
        except Exception:
            Xs = X
    else:
        Xs = X
    timings["feature"] = time.perf_counter() - t1

    """

    
    

    #SECTION 2 ENDS NON GPU optimized feature calculation 
    ######################################################################################
    #######################################################################################
    #######################################################################################

   
    # Phase 3: Predict
    t2 = time.perf_counter()
    Yhat = model(Xs, training=False)
    # Force materialization
    Yhat_np = Yhat.numpy()
    timings["predict"] = time.perf_counter() - t2

    # Scatter back
    w_phi = np.full_like(g, np.nan, dtype=np.float32)
    phi_x = np.full_like(g, np.nan, dtype=np.float32)
    phi_y = np.full_like(g, np.nan, dtype=np.float32)
    theta = np.full_like(g, np.nan, dtype=np.float32)

    w_phi[ir, jc] = Yhat_np[:,0].astype(np.float32)
    if Yhat_np.shape[1] >= 4:
        phi_x[ir, jc] = Yhat_np[:,1].astype(np.float32)
        phi_y[ir, jc] = Yhat_np[:,2].astype(np.float32)
        theta[ir, jc] = Yhat_np[:,3].astype(np.float32)

    # Quality map
    diff_mag = np.abs(w_phi - np.abs(phi_x + 1j*phi_y)).astype(np.float32)
    # mat2gray
    M_proc = ~np.isnan(w_phi) # valid where we wrote predictions
    if np.any(M_proc > 0):
        vals = diff_mag[M_proc > 0]
        mn, mx = np.min(vals), np.max(vals)
    else:
        mn, mx = np.min(diff_mag), np.max(diff_mag)
    rng = max(mx - mn, 1e-12)
    diff01 = (diff_mag - mn) / rng
    
    QM = M_proc * (1.0 - diff01)

    timings["total"] = timings["patch"] + timings["feature"] + timings["predict"]

    out = (w_phi, phi_x, phi_y, theta, QM.astype(np.float32), M_proc.astype(np.float32))
    return (*out, timings) if return_timing else out

