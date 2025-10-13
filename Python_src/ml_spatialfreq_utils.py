
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

def calc_feature_batch(B: np.ndarray, feature_name: str):
    '''
    Vectorized feature extractor for a stack of patches.
    B: N x M x L stack
    feature_name: "feature_GV" | "feature_DFT" | "feature_projected_DFT"
    Returns: X_sel (L x F), and a dict S with some intermediates for debugging
    '''
    if B.ndim != 3:
        raise ValueError("B must be N x M x L")
    N, M, L = B.shape
    # Window
    W = _gaussian_window(N, M)
    W3 = np.repeat(W[..., None], L, axis=2)  # N x M x L

    # DC removal and windowing
    mu = B.mean(axis=(0,1), keepdims=True)  # 1 x 1 x L
    B0 = B - mu
    Bwin = B0 * W3

    # FFT per patch
    G = np.fft.fft2(Bwin, axes=(0,1))
    # Remove DC frequency
    G[0,0,:] = 0.0
    # Shift for visualization/consistency
    Gshift = np.fft.fftshift(G, axes=(0,1))
    absG = np.abs(Gshift)

    # Normalize per patch to reduce illumination dependence
    gnorm = absG.reshape(N*M, L).sum(axis=0) + 1e-12
    absG_norm = absG / gnorm[None, None, :]

    # Basic feature sets
    if feature_name == "feature_GV":
        # Simple gradient-based stats (spatial domain)
        Gy, Gx = np.gradient(Bwin, axis=(0,1))
        f1 = np.mean(np.abs(Gx), axis=(0,1))
        f2 = np.mean(np.abs(Gy), axis=(0,1))
        f3 = np.sqrt(np.mean(Bwin**2, axis=(0,1)))
        X_sel = np.stack([f1, f2, f3], axis=1)
        S = {"W": W, "G": None, "absG_norm": None}
        return X_sel.astype(np.float32), S

    # For frequency-domain features, collapse along one axis and take spectrum projections
    proj_x = absG_norm.sum(axis=0)   # M x L
    proj_y = absG_norm.sum(axis=1)   # N x L

    # Peak locations and a few moments as features
    def peak_and_moments(v2d):
        # v2d: K x L (K bins per patch)
        peak_idx = v2d.argmax(axis=0)
        peak_val = v2d.max(axis=0)
        # first moment around peak within +/- 3 bins
        K, Llocal = v2d.shape
        X = np.arange(K)[:, None]
        feats = []
        for i in range(Llocal):
            p = int(peak_idx[i])
            lo = max(0, p-3); hi = min(K, p+4)
            w = v2d[lo:hi, i]
            x = X[lo:hi]
            wsum = float(w.sum() + 1e-12)
            m1 = float((w * x).sum() / wsum)
            m2 = float(((w * (x - m1)**2).sum()) / wsum)
            feats.append((float(peak_val[i]), float(p), m1, m2))
        return np.array(feats, dtype=np.float32)  # L x 4

    fx = peak_and_moments(proj_x)
    fy = peak_and_moments(proj_y)

    if feature_name in ("feature_DFT", "feature_normalized_DFT"):
        X_sel = np.concatenate([fx, fy], axis=1)  # L x 8
    elif feature_name == "feature_projected_DFT":
        # Add cross quadrants energy as crude directional info
        q1 = absG_norm[:N//2, :M//2, :].reshape(-1, L).sum(axis=0)
        q2 = absG_norm[:N//2, M//2:, :].reshape(-1, L).sum(axis=0)
        q3 = absG_norm[N//2:, :M//2, :].reshape(-1, L).sum(axis=0)
        q4 = absG_norm[N//2:, M//2:, :].reshape(-1, L).sum(axis=0)
        X_sel = np.concatenate([fx, fy, q1[:,None], q2[:,None], q3[:,None], q4[:,None]], axis=1)  # L x 12
    else:
        raise ValueError(f"Unknown feature_name: {feature_name}")

    S = {"W": W, "G": None, "absG_norm": absG_norm}
    return X_sel.astype(np.float32), S

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

def synth_fringe(NR: int, NC: int, w0_x: float, w0_y: float, phi: float=0.0,
                 modulation: float=1.0, background: float=0.0, noise_std: float=0.0) -> np.ndarray:
    '''
    Generate a sinusoidal fringe pattern:
        g = background + modulation * cos(w0_x * X + w0_y * Y + phi) + noise
    '''
    y, x = np.mgrid[0:NR, 0:NC]
    g = background + modulation * np.cos(w0_x * x + w0_y * y + phi)
    if noise_std > 0:
        g = g + np.random.normal(0, noise_std, size=g.shape)
    return g.astype(np.float32)
