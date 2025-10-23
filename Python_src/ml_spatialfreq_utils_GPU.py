
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
    Exact TF reimplementation of ml_spatialfreq_utils.calc_feature_batch (NumPy).
    """
    assert featureName in ("feature_GV","feature_DFT","feature_projected_DFT")
    assert B.shape.rank == 3, "B must be (N,M,L)"
    B = tf.cast(B, tf.float32)
    N = tf.shape(B)[0]; M = tf.shape(B)[1]; L = tf.shape(B)[2]

    # ----- Window with 1-based centering, like NumPy -----
    # x0 = floor(M/2)+1 ; y0 = floor(N/2)+1 ; x=[1..M]-x0 ; y=[1..N]-y0
    x0 = (M // 2) + 1
    y0 = (N // 2) + 1
    xx = tf.range(1, M + 1, dtype=tf.int32)
    yy = tf.range(1, N + 1, dtype=tf.int32)
    XX, YY = tf.meshgrid(xx, yy, indexing='xy')
    x = tf.cast(XX - x0, tf.float32)
    y = tf.cast(YY - y0, tf.float32)
    sigma_x = tf.cast(M, tf.float32) / 3.0
    sigma_y = tf.cast(N, tf.float32) / 3.0
    W2 = tf.exp(-0.5*(x**2)/(sigma_x**2) - 0.5*(y**2)/(sigma_y**2))      # (N,M)
    W = W2[..., None]                                                     # (N,M,1)

    # ----- DC removal (spatial) THEN apply spatial window (NumPy order) -----
    mu_sp = tf.reduce_mean(B, axis=(0,1), keepdims=True)                  # (1,1,L)
    B0    = B - mu_sp
    Bwin  = B0 * W

    if featureName == "feature_GV":
        eps_ = tf.constant(1e-12, tf.float32)
        muGV = tf.reduce_mean(Bwin, axis=(0,1))                           # (L,)
        E2   = tf.reduce_mean(Bwin*Bwin, axis=(0,1))                      # (L,)
        sigmaGV = tf.sqrt(tf.maximum(E2 - muGV*muGV, eps_))
        X = tf.reshape(tf.transpose(Bwin, perm=[2,0,1]), [L, -1])         # (L, N*M)
        X_sel = (X - muGV[:,None]) / sigmaGV[:,None]
        return tf.cast(X_sel, tf.float32), {"W": W2}

    # ----- FFT per patch, zero DC, fftshift, then freq weighting with same W -----
    # tf.signal.fft2d always transforms the last two axes, while NumPy call transforms axes (0,1) of an array with shape (N, M, L).
    # NumPy: np.fft.fft2(Bwin_np, axes=(0,1)) → FFT over the first two axes (N, M), treating L as batch.
    # TensorFlow: tf.signal.fft2d(Bwin_tf) → FFT over the innermost two axes. With shape (N, M, L), TF will transform axes (M, L), which is wrong for your case.
    # send the two first axes to the innermost dimmnesions
    B_chw = tf.transpose(Bwin, perm=[2, 0, 1])                      # (L,N,M)
    G_chw = tf.signal.fft2d(tf.cast(B_chw, tf.complex64))
    G = tf.transpose(G_chw, perm=[1, 2, 0])    
    
    # zero DC bin [0,0,:] BEFORE fftshift
    idx = tf.stack([
        tf.zeros([L], tf.int32),            # rows=0
        tf.zeros([L], tf.int32),            # cols=0
        tf.range(L, dtype=tf.int32)         # pages k
    ], axis=1)
    G = tf.tensor_scatter_nd_update(G, idx, tf.zeros([L], G.dtype))
    G = tf.signal.fftshift(G, axes=(0, 1))                               # fftshift axis 0 and 1
    G = G * tf.cast(W, G.dtype)                                           # same Gaussian in freq, boradcast over pages

    # ----- Magnitude and full-spectrum normalization -----
    G_abs = tf.math.abs(G)                                                # (N,M,L)
    GNorm = tf.reduce_sum(G_abs, axis=(0,1))                               # (L,)
    eps_ = tf.constant(1e-12, tf.float32)
    GNorm = tf.maximum(GNorm, eps_)

    # ----- Semiplane wx >= 0: columns x0..M (1-based) -> [x0-1 : M) 0-based -----
    start = x0 - 1
    abs_G_sp = G_abs[:, start:, :]                                        # (N, Wxp, L)
    Wxp=tf.shape(abs_G_sp)[1]

    # ----- Decide top vs bottom half by energy inside that semiplane -----
    # Top rows: [0 : y0-1), Bottom rows: [y0 : N)
    sumTop    = tf.reduce_sum(abs_G_sp[0:(y0-1), :, :], axis=(0,1))       # (L,)
    sumBottom = tf.reduce_sum(abs_G_sp[y0:, :, :], axis=(0,1))            # (L,)
    isTopGreater = sumTop > sumBottom                                     # (L,)

    # Build row masks and select with tf.where (no scatter)
    row_idx   = tf.range(tf.shape(abs_G_sp)[0])
    top_mask  = tf.cast(row_idx[:,None,None] <  (y0-1), abs_G_sp.dtype)   # (N,1,1)
    bot_mask  = tf.cast(row_idx[:,None,None] >=  y0,    abs_G_sp.dtype)   # (N,1,1)
    keep_top  = abs_G_sp * top_mask
    keep_bot  = abs_G_sp * bot_mask
    abs_sel   = tf.where(tf.reshape(isTopGreater, [1,1,-1]), keep_top, keep_bot)  # (N,Wxp,L)

    # ----- Assemble features -----
    if featureName == "feature_DFT":
        X = tf.reshape(tf.transpose(abs_sel, perm=[2,0,1]), [L, -1])      # (L, N*Wxp)
        X_sel = X / GNorm[:,None]
    else:  # feature_projected_DFT
        G_sp_XP = tf.reduce_sum(abs_sel, axis=0)                           # (Wxp,L)
        G_sp_YP = tf.reduce_sum(abs_sel, axis=1)                           # (N,L)
        X = tf.concat([tf.transpose(G_sp_XP), tf.transpose(G_sp_YP)], axis=1)  # (L, Wxp+N)
        X_sel = X / GNorm[:,None]

    # ----- Optional debug/inspection outputs -----
    if featureName != "feature_GV":
        S = {
            "x0": x0,
            "y0": y0,
            "W": W2,                 # one window (all identical across L)
            "GNorm": GNorm,          # (L,1)
            "abs_G_sp": abs_G_sp,    # (N, Wxp, L)
            "abs_sel": abs_sel,      # (N, Wxp, L)
            "mu_sp": mu_sp,          # (1,1,L)
            "X": X ,                 # (L, N*Wxp)
            "isTopGreater": isTopGreater, # (L,)
            "L": L,                  # ()
            "G_sp_XP": G_sp_XP,      # (L, Wxp)
            "G_sp_YP": G_sp_YP,      # (L, N)
            "Wxp": Wxp,              # ()
            "G_abs": G_abs,          # (N,M,L)
            "G": G,                  # (N,M,L)
            "Bwin": Bwin             # (N,M,L)
        }
    else:
        S=None


    return tf.cast(X_sel, tf.float32), S


    
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

    # AQDEBUG 23OCT25 to compare numpy vs TF see DEBUG_Predict_TF_SpatialFreqs notebook
    Xs, _ = tf_calc_feature_batch(B, feat_name)
    
    # --- Feature scaling (robust) ---
    # If the model begins with a Keras Normalization layer, skip external scaling.
    use_external_scaling = True
    try:
        from tf.keras.layers import Normalization as _Norm # type: ignore
        if isinstance(getattr(model, "layers", [None])[0], _Norm):
            use_external_scaling = False
    except Exception:
        pass

    if use_external_scaling:
        # 1) Preferred: scaler has mean_/var_ or mean_/scale_
        if scaler is not None:
            has_mean = hasattr(scaler, "mean_")
            has_var  = hasattr(scaler, "var_")
            has_scale = hasattr(scaler, "scale_")

            if has_mean and (has_var or has_scale):
                mean_vec = tf.convert_to_tensor(scaler.mean_, dtype=tf.float32)
                if has_var:
                    std_vec = tf.sqrt(tf.convert_to_tensor(scaler.var_, dtype=tf.float32))
                else:
                    std_vec = tf.convert_to_tensor(scaler.scale_, dtype=tf.float32)

                # shape guard: (F,) -> (1,F) for broadcasting against Xs: (L,F)
                mean_vec = tf.reshape(mean_vec, [1, -1])
                std_vec  = tf.reshape(std_vec,  [1, -1])

                # sanity check: feature dims match
                f_model = int(Xs.shape[-1])
                f_scaler = int(mean_vec.shape[-1])
                if f_model != f_scaler:
                    raise ValueError(
                        f"Scaler length {f_scaler} does not match feature length {f_model}. "
                        "Check patch size / feature variant and the scaler used at training."
                    )

                Xs = (Xs - mean_vec) / tf.maximum(std_vec, 1e-8)
        
            # 2) Fallback: use scaler.transform on CPU, then convert back
            elif hasattr(scaler, "transform"):
                Xs = tf.convert_to_tensor(scaler.transform(Xs.numpy()).astype(np.float32))
      # If nothing applied, we proceed unscaled (assuming model handles it)

    _force_sync(Xs)
    
    timings["feature"] = time.perf_counter() - t1

   
    # Phase 3: Predict
    t2 = time.perf_counter()
    Yhat = model(Xs, training=False, batch_size=8192)
    #Yhat = model.predict(Xs, verbose=0, batch_size=8192)
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

