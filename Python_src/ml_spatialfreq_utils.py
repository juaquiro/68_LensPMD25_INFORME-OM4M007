
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# -----------------------------
# naturalSort (from naturalSort.asv)
# -----------------------------

def natural_sort(strings: List[str], prefix: str) -> List[str]:
    '''
    Sort a list of strings in "natural" order based on the integer after a given prefix.
    Example: ["X_1","X_2","X_10"] -> ["X_1","X_2","X_10"]
    '''
    def extract_num(s: str) -> int:
        s2 = s.replace(prefix, "")
        m = re.search(r"(\d+)", s2)
        return int(m.group(1)) if m else -1
    return sorted(strings, key=extract_num)

# -----------------------------
# expandMatrixField (table helper) -- Python analogue
# -----------------------------

def expand_matrix_field(dict_rows: List[Dict], field_name: str) -> List[Dict]:
    '''
    Given a list of dict rows, where rows[i][field_name] is a 1D or 2D array,
    expand into multiple scalar fields <field_name>_1, ..., and remove original field.
    Returns a NEW list of dict rows.
    '''
    out = []
    for row in dict_rows:
        row = dict(row)  # shallow copy
        import numpy as _np
        M = _np.asarray(row.get(field_name))
        if M.ndim == 1:
            vals = M
        elif M.ndim == 2:
            vals = M.ravel()
        else:
            raise ValueError(f"{field_name} must be 1D or 2D, got shape {M.shape}")
        for i, v in enumerate(vals, start=1):
            row[f"{field_name}_{i}"] = float(v)
        row.pop(field_name, None)
        out.append(row)
    return out

# -----------------------------
# calcFeatureBatch (from calcFeatureBatch.asv) -- simplified but consistent
# -----------------------------

def _gaussian_window(N: int, M: int, sigma_y: float=None, sigma_x: float=None) -> np.ndarray:
    if sigma_x is None: sigma_x = M/3.0
    if sigma_y is None: sigma_y = N/3.0
    y, x = np.mgrid[0:N, 0:M]
    x = x - (M//2)
    y = y - (N//2)
    W = np.exp(-0.5*(x**2)/sigma_x**2 - 0.5*(y**2)/sigma_y**2)
    return W

def calc_feature_batch(B: np.ndarray, featureName: str = "feature_projected_DFT"):
    """
    Vectorized feature extractor for a stack of patches.

    Parameters
    ----------
    B : (N, M, L) ndarray
        Each page B[:, :, k] is one patch.
    featureName : {"feature_GV","feature_DFT","feature_projected_DFT"}

    Returns
    -------
    X_sel : (L, F) ndarray
        Row i is the feature vector of patch i.
    S : dict
        Optional intermediates for debugging/inspection.
    """
    assert featureName in ("feature_GV", "feature_DFT", "feature_projected_DFT")

    # ----- Sizes and dtype like MATLAB -----
    N, M, L = B.shape
    B = B.astype(np.float64, copy=False)

    # ----- Spatial Gaussian window (same as MATLAB code) -----
    # MATLAB uses 1-based coords and centers at x0 = floor(M/2)+1, y0 = floor(N/2)+1
    x0 = (M // 2) + 1
    y0 = (N // 2) + 1
    # Build 1-based meshgrid to match the MATLAB math
    xx, yy = np.meshgrid(np.arange(1, M + 1), np.arange(1, N + 1))
    x = xx - x0
    y = yy - y0
    sigma_x = M / 3.0
    sigma_y = N / 3.0
    W2 = np.exp(-0.5 * (x ** 2) / (sigma_x ** 2) - 0.5 * (y ** 2) / (sigma_y ** 2))  # (N, M)
    # Broadcast to (N, M, L)
    W = np.repeat(W2[:, :, None], L, axis=2)

    # ----- DC removal (spatial) then apply spatial window -----
    # MATLAB: mu_sp = mean(B,[1 2]) -> (1,1,L)
    mu_sp = B.mean(axis=(0, 1), keepdims=True)  # (1,1,L)
    B0 = B - mu_sp
    Bwin = B0 * W

    # ----- FFT per patch, kill DC, center spectrum, low-pass with same W -----
    # fft over dims (0,1)
    G = np.fft.fft2(Bwin, axes=(0, 1))
    G[0, 0, :] = 0.0  # remove DC per patch
    # center zero-frequency along spatial dims (two fftshift calls in MATLAB)
    G = np.fft.fftshift(G, axes=(0, 1))
    # Gaussian low-pass in frequency domain with same W (W2 replicated)
    G = G * W

    # ----- Shared quantities for spectral features -----
    G_abs = np.abs(G)                     # (N, M, L)
    # MATLAB: GNorm = reshape(sum(sum(G_abs,1),2), [L 1])
    GNorm = G_abs.sum(axis=(0, 1))[:, None]  # (L,1)

    # Semiplane wx >= 0 corresponds to columns x0..M in 1-based -> zero-based slice x0-1 : M
    wx_cols = np.arange(x0 - 1, M)
    abs_G_sp = G_abs[:, wx_cols, :]       # (N, Wxp, L)
    Wxp = abs_G_sp.shape[1]               # floor(M/2)+1

    # Decide which wy half to keep: compare energy above vs below y0 in the wx>=0 semiplane
    # Top means rows 1..(y0-1) in 1-based -> [0 : y0-1) in 0-based
    sumTop = abs_G_sp[0:(y0 - 1), :, :].sum(axis=(0, 1))[:, None]       # (L,1), wy>0
    # Bottom means rows (y0+1)..N -> [y0 : N) in 0-based
    sumBottom = abs_G_sp[y0:, :, :].sum(axis=(0, 1))[:, None]           # (L,1), wy<0
    isTopGreater = (sumTop > sumBottom).reshape(L)                      # (L,)

    # Build two candidates and select per page (exactly like MATLAB)
    abs1 = abs_G_sp.copy()                  # zero TOP when ~isTopGreater
    abs1[0:(y0 - 1), :, :] = 0.0
    abs2 = abs_G_sp.copy()                  # zero BOTTOM when isTopGreater
    abs2[y0:, :, :] = 0.0

    abs_sel = abs1.copy()
    # for pages where top is greater, use abs2
    if L > 0:
        abs_sel[:, :, isTopGreater] = abs2[:, :, isTopGreater]

    eps_ = np.finfo(float).eps

    # ----- Feature assembly -----
    if featureName == "feature_GV":
        # muGV = mean(Bwin,[1 2]) -> (L,1); E2 = mean(Bwin.^2,[1 2]) -> (L,1)
        muGV = Bwin.mean(axis=(0, 1))[:, None]                    # (L,1)
        E2 = (Bwin ** 2).mean(axis=(0, 1))[:, None]               # (L,1)
        sigmaGV = np.sqrt(np.maximum(E2 - muGV ** 2, eps_))       # (L,1)

        # Flatten patches: (N*M, L).T -> (L, N*M)
        X = Bwin.reshape(N * M, L).T                              # (L, N*M)
        X_sel = (X - muGV) / sigmaGV                              # broadcast (L,1)

    elif featureName == "feature_DFT":
        # Flatten selected semiplane magnitudes: (N*Wxp, L).T -> (L, N*Wxp)
        X = abs_sel.reshape(N * Wxp, L).T                         # (L, N*Wxp)
        X_sel = X / np.maximum(GNorm, eps_)                       # rowwise normalize by GNorm (L,1)

    else:  # "feature_projected_DFT"
        # X-projection: sum over rows -> (Wxp, L).T -> (L, Wxp)
        G_sp_XP = abs_sel.sum(axis=0).T                           # (L, Wxp)
        # Y-projection: sum over cols -> (N, L).T -> (L, N)
        G_sp_YP = abs_sel.sum(axis=1).T                           # (L, N)
        X = np.concatenate([G_sp_XP, G_sp_YP], axis=1)            # (L, Wxp+N)
        X_sel = X / np.maximum(GNorm, eps_)                       # (L, Wxp+N)

    # ----- Optional debug/inspection outputs -----
    S = {
        "x0": x0,
        "y0": y0,
        "W": W2,                 # one window (all identical across L)
        "GNorm": GNorm,          # (L,1)
        "abs_G_sp": abs_G_sp,    # (N, Wxp, L)
        "abs_sel": abs_sel       # (N, Wxp, L)
    }
    if featureName != "feature_GV":
        try:
            S["G_sp_XP"] = abs_sel.sum(axis=0).T  # (L, Wxp)
            S["G_sp_YP"] = abs_sel.sum(axis=1).T  # (L, N)
        except Exception:
            S["G_sp_XP"] = None
            S["G_sp_YP"] = None

    return X_sel.astype(np.float64, copy=False), S

# -----------------------------
# Regression wrapper similar to calcSpatialFreqsSupervisedRegressionBatch
# -----------------------------

@dataclass
class DBInfo:
    featureName: str
    patch_NR: int
    patch_NC: int

@dataclass
class TrainedModelTF:
    '''Python-side handle analogous to MATLAB trainedModel struct.'''
    model_path: str
    scaler_path: Optional[str]
    meta_path: Optional[str]
    DB_info: DBInfo

    def load(self):
        from tensorflow import keras
        import pickle, json
        model = keras.models.load_model(self.model_path)
        scaler = None
        if self.scaler_path:
            with open(self.scaler_path, "rb") as f:
                scaler = pickle.load(f)
        meta = None
        if self.meta_path:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        return model, scaler, meta

def calc_spatial_freqs_supervised_regression_batch(
    g: np.ndarray,
    trained_model: TrainedModelTF,
    feature_name: str,
    M_ROI: Optional[np.ndarray] = None,
):
    '''
    Python analogue to calcSpatialFreqsSupervisedRegressionBatch.m
    Returns: w_phi, phi_x, phi_y, theta, QM, M_proc
    All maps are the same size as g.
    '''
    if M_ROI is None:
        M_ROI = np.ones_like(g, dtype=bool)
    M_ROI = M_ROI.astype(bool)

    model, scaler, meta = trained_model.load()
    assert feature_name == trained_model.DB_info.featureName, "FeatureName does not match trained model"

    N = trained_model.DB_info.patch_NR
    M = trained_model.DB_info.patch_NC
    r = N//2; c = M//2  # center offset

    NR, NC = g.shape
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

    # Scale if scaler exists
    if scaler is not None:
        try:
            Xs = scaler.transform(X)
        except Exception:
            Xs = X
    else:
        Xs = X

    # Predict (multi-output): assume order [w, wx, wy, theta] or [w, phi_x, phi_y, theta]
    Yhat = model.predict(Xs, verbose=0)
    if Yhat.ndim == 1:
        Yhat = Yhat[:, None]

    # Map back to images
    w_phi  = np.zeros_like(g, dtype=np.float32)
    phi_x  = np.zeros_like(g, dtype=np.float32)
    phi_y  = np.zeros_like(g, dtype=np.float32)
    theta  = np.zeros_like(g, dtype=np.float32)
    QM     = np.zeros_like(g, dtype=bool)

    for k in range(L):
        i = ir[k]; j = jc[k]
        w_phi[i, j] = float(Yhat[k, 0])
        if Yhat.shape[1] >= 4:
            phi_x[i, j] = float(Yhat[k, 1])
            phi_y[i, j] = float(Yhat[k, 2])
            theta[i, j] = float(Yhat[k, 3])
        QM[i, j] = True

    M_proc = QM.astype(np.float32)
    return w_phi, phi_x, phi_y, theta, QM, M_proc

# -----------------------------
# Synthetic pattern generator (to mirror MATLAB test)
# -----------------------------

def synth_fringe(
    NR: int,
    NC: int,
    w0_x: float,
    w0_y: float,
    psi: float = 0.0,
    modulation: float = 1.0,
    background: float = 0.0,
    noise_std: float = 0.0,
    return_maps: bool = False,
):
    """
    Generate a sinusoidal fringe pattern:
        g = background + modulation * cos(w0_x * X + w0_y * Y + psi) + noise

    If return_maps=True, also return:
        w_phi  = |phi_x + i*phi_y|
        phi_x  = d(phi)/dx
        phi_y  = d(phi)/dy
        theta  = atan2(-phi_y, phi_x)
    All derivatives use unit pixel spacing.
    """
    # pixel coordinates (row-major y, column-major x)
    y, x = np.mgrid[0:NR, 0:NC]

    # total phase
    phi = w0_x * x + w0_y * y + psi

    # fringe
    g = background + modulation * np.cos(phi)
    if noise_std > 0:
        g = g + np.random.normal(0, noise_std, size=g.shape)

    if not return_maps:
        return g.astype(np.float32)

    # gradients: numpy returns (d/dy, d/dx)
    gy, gx = np.gradient(phi)     # gy = dphi/dy, gx = dphi/dx
    phi_x = gx
    phi_y = gy

    # local spatial frequency magnitude and fringe orientation
    w_phi = np.abs(phi_x + 1j * phi_y)
    theta = np.arctan2(-phi_y, phi_x)

    return (
        g.astype(np.float32),
        w_phi.astype(np.float32),
        phi_x.astype(np.float32),
        phi_y.astype(np.float32),
        theta.astype(np.float32),
    )


def peaks(NR: int, NC: int) -> np.ndarray:
    """
    MATLAB-like peaks surface over [-3, 3] × [-3, 3].
    Returns an array of shape (NR, NC).
    """
    ys = np.linspace(-3, 3, NR)
    xs = np.linspace(-3, 3, NC)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    Z = (
        3 * (1 - X)**2 * np.exp(-(X**2) - (Y + 1)**2)
        - 10 * (X/5 - X**3 - Y**5) * np.exp(-X**2 - Y**2)
        - (1/3) * np.exp(-((X + 1)**2) - Y**2)
    )
    return Z
