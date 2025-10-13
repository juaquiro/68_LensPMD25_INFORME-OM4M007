function [X_sel, S] = calcFeatureBatch(B, featureName)
% calcFeatureBatch  Vectorized feature extractor for a stack of patches.
% 
%   [X_sel, S] = calcFeatureBatch(B, featureName)
%
% Inputs
%   B           : N x M x L array, each page B(:,:,k) is one patch (double/single ok)
%   featureName : "feature_GV" | "feature_DFT" | "feature_projected_DFT"
%
% Output
%   X_sel : L x F feature matrix (row i = feature vector of patch i)
%   S     : (optional) struct with selected intermediates for debugging
%
% Notes
% - Mirrors the per-patch logic in calcFeature.m, including:
%     * DC removal in spatial domain
%     * Gaussian spatial window W (minimize borders)
%     * DFT + DC kill at (1,1), fftshift to center
%     * Gaussian low-pass in frequency domain (same W)
%     * Semiplane selection wx >= 0, and sign disambiguation along wy
%     * Normalization by sum(abs(G(:))) per patch
% - Entire pipeline is vectorized across the L patches (no loops).

arguments
    B (:,:,:) {mustBeNumeric}
    featureName string {mustBeMember(featureName,["feature_GV","feature_DFT","feature_projected_DFT"])} = "feature_projected_DFT"
end

[N, M, L] = size(B);
B = double(B);                         % ensure predictable math

% ----- apodization mask, Spatial Gaussian window (same as calcFeature) ---------------------
x0 = floor(M/2)+1;  y0 = floor(N/2)+1;         % image center (1-based)
[x, y] = meshgrid(1:M, 1:N);
x = x - x0;  y = y - y0;

sigma_x = M/3;
sigma_y = N/3;
W = exp(-0.5*(x.^2)/sigma_x^2 - 0.5*(y.^2)/sigma_y^2);    % N x M
W = repmat(W, 1, 1, L);                                   % N x M x L (broadcast)

% ----- DC removal (spatial), then apply spatial window -------------------
mu_sp = mean(B, [1 2]);                % 1 x 1 x L
B0    = B - mu_sp;                     % zero-mean per patch
Bwin  = B0 .* W;                       % windowed patches

% ----- FFT per patch, kill DC, center spectrum, low-pass with same W ----
G = fft2(Bwin);                        % N x M x L, fft over dims 1&2
G(1,1,:) = 0;                          % remove DC (per patch)
% center zero-frequency to (y0,x0); shift only along spatial dims
G = fftshift(fftshift(G,1), 2);        % center spectrum
G = G .* W;                            % Gaussian low-pass in frequency domain

% ----- Shared quantities for all spectral features -----------------------
G_abs  = abs(G);                       % N x M x L
GNorm  = reshape(sum(sum(G_abs,1),2), [L 1]);      % L x 1

% Semiplane wx >= 0 -> columns x0..M
wx = (x0:M);                           % indices of the non-negative wx semiplane
abs_G_sp = G_abs(:, wx, :);            % N x (floor(M/2)+1) x L
Wxp = numel(wx);                       % = floor(M/2)+1

% Decide which wy half to keep (as in calcFeature):
% Compare energy above vs. below the center row y0 in the wx>=0 semiplane.
sumTop    = reshape(sum(sum(abs_G_sp(1:y0-1,:,:), 1), 2), [L 1]);    % wy>0 L x 1
sumBottom = reshape(sum(sum(abs_G_sp(y0+1:end,:,:), 1), 2), [L 1]);  % wy<0 L x 1
isTopGreater = sumTop > sumBottom;    % logical Lx1: true -> keep top wx>0 (zero bottom)

% Build two candidates and select per page:
abs1 = abs_G_sp;                % version where the TOP half wy>0 is zeroed (use when ~isTopGreater)
abs1(1:y0-1,:,:) = 0;
abs2 = abs_G_sp;                % version where the BOTTOM half wy<0 is zeroed (use when isTopGreater)
abs2(y0+1:end,:,:) = 0;

abs_sel = abs1;                 % N x Wxp x L
abs_sel(:,:,isTopGreater) = abs2(:,:,isTopGreater);

% ----- Feature assembly ---------------------------------------------------
switch featureName
    case "feature_GV"
        % Per-patch mean & std after windowing (as in calcFeature path)
        muGV = reshape(mean(Bwin, [1 2]), [L 1]);                          % L x 1
        % std over pixels: sqrt(E[x^2]-E[x]^2)
        E2   = reshape(mean(Bwin.^2, [1 2]), [L 1]);                       % L x 1
        sigmaGV = sqrt(max(E2 - muGV.^2, eps));                            % L x 1
        
        % Flatten patches: (N*M) x L -> L x (N*M)
        X = reshape(Bwin, N*M, L).';                                       % L x (N*M)
        % Normalize rows by muGV/sigmaGV
        X = (X - muGV) ./ sigmaGV;                                         % broadcasting Lx1
        X_sel = X;                                                         % L x (N*M)
        
    case "feature_DFT"
        % Flatten selected semiplane magnitudes: N*Wxp x L -> L x (N*Wxp)
        X = reshape(abs_sel, N*Wxp, L).';                                  % L x (N*Wxp)
        % Normalize each row by GNorm
        X_sel = X ./ max(GNorm, eps);                                      % L x (N*Wxp)
        
    case "feature_projected_DFT"
        % Projection along rows (X-projection): sum over dim-1
        G_sp_XP = squeeze(sum(abs_sel, 1)).';     % 1 x Wxp x L -> L x Wxp
        % Projection along cols (Y-projection): sum over dim-2
        G_sp_YP = squeeze(sum(abs_sel, 2)).';     % 1 x N x L   -> L x N
        % Concatenate and normalize
        X = [G_sp_XP, G_sp_YP];                            % L x (Wxp + N)
        X_sel = X ./ max(GNorm, eps);                      % L x (Wxp + N)
end

% ----- Optional debug/inspection outputs ---------------------------------
if nargout > 1
    S = struct();
    S.x0 = x0; S.y0 = y0;
    S.W  = W(:,:,1);                         % one window (all identical)
    S.GNorm = GNorm;                         % L x 1
    S.abs_G_sp = abs_G_sp;                   % N x Wxp x L
    S.abs_sel  = abs_sel;                    % N x Wxp x L
    if featureName ~= "feature_GV"
        S.G_sp_XP = []; S.G_sp_YP = [];
        try
            S.G_sp_XP = squeeze(sum(abs_sel, 1)).';   % L x Wxp
            S.G_sp_YP = squeeze(sum(abs_sel, 2)).';   % L x N
        catch
        end
    end
end
end
