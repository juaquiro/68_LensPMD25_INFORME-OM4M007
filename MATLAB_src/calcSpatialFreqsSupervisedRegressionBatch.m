function [w_phi, phi_x, phi_y, theta, QM, M_proc] = ...
    calcSpatialFreqsSupervisedRegressionBatch(g, trainedModel, featureName, M_ROI)

% calcSpatialFreqsSupervisedRegression (optimized)
% - Vectorized valid-patch detection (conv2)
% - Extract only needed patches (no full im2col)
% - Parallel feature computation (parfor if a pool is open)
% - Same outputs: w_phi, phi_x, phi_y, theta, QM, M_proc

arguments
    g (:,:) {mustBeNumeric}
    trainedModel struct {mustHaveFields(trainedModel, {'predictFcn','DB_info','RequiredVariables'})}
    featureName string {mustBeMember(featureName,["feature_GV","feature_DFT","feature_projected_DFT","feature_normalized_DFT"])} = "feature_projected_DFT"
    M_ROI (:,:) {mustBeNumericOrLogical} = ones(size(g))
end

% --- Compatibility shim
if featureName=="feature_normalized_DFT", featureName="feature_DFT"; end
if ~strcmp(featureName, trainedModel.DB_info.featureName)
    error('FeatureName does not match trainedModel.DB_info.featureName');
end

%image and patch size
[NR,NC] = size(g);
N = trainedModel.DB_info.patch_NR;
M = trainedModel.DB_info.patch_NC;

% Center offsets (top-left convention works for odd/even)
% They define how far a patch of size N×M extends above/below and left/right of its center pixel.
% r = number of rows above the center (the “upper half”)
% r2 = number of rows below the center (the “lower half”)
% c = number of columns to the left
% c2 = number of columns to the right
% They are computed so that:
% r+r2+1=N
% c+c2+1=M
r  = floor((N-1)/2);  r2 = (N-1)-r;
c  = floor((M-1)/2);  c2 = (M-1)-c;

% 1) valid centers: centered N×M neighborhood fully inside ROI
valid_center = conv2(single(M_ROI), ones(N,M,'single'), 'same') == N*M;

% map to top-left grid (what sliding windows use)
rows = (1+r):(NR-r2);
cols = (1+c):(NC-c2);
if isempty(rows) || isempty(cols)
    [w_phi,phi_x,phi_y,theta,QM,M_proc] = deal(zeros(NR,NC,'like',g));
    return
end
tl_valid = valid_center(rows, cols);                 % (NR-N+1) x (NC-M+1)

% centers of selected windows
[ir,jc]   = find(tl_valid);                          % top-left grid indices
cent_r    = ir + r;                                  % center row
cent_c    = jc + c;                                  % center col
cent_lin  = sub2ind([NR,NC], cent_r, cent_c);
L         = numel(cent_lin);   % number of NxM blocks fully inside M_ROI           
if L==0
    [w_phi,phi_x,phi_y,theta,QM,M_proc] = deal(zeros(NR,NC,'like',g));
    return
end

% 2) extract only needed patches into (N*M) x L without full im2col
%    Build linear indices for all selected top-lefts + kernel offsets
tl_r = cent_r - r;   % top-left rows
tl_c = cent_c - c;   % top-left cols
base = double(tl_r - 1 + (tl_c - 1)*NR);             % 0-based base index, 1xL
row_off = (0:N-1)';                                  % N x 1
col_off = (0:M-1) * NR;                              % 1 x M
kernel = row_off + col_off;                          % N x M
kernel = kernel(:);                                  % (N*M) x 1
idx_mat = kernel + base.';                           % (N*M) x L (0-based)
idx_mat = idx_mat + 1;                               % to 1-based
C_sel   = g(idx_mat);                                % (N*M) x L
% reshape for batch ops if you want them later
% B = reshape(C_sel, N, M, L);

% 3) feature computation (vectorized or parfor on columns)
F = numel(trainedModel.RequiredVariables);
% X_sel = zeros(L, F);                                  % features at centers only

% use vectorized version of calcFeature
X_sel = calcFeatureBatch(reshape(C_sel,N,M,L), featureName);  % L x F

% we can still use the calcFeature using a for or parfor loop
% for k = 1:L
%     patch = reshape(C_sel(:,k), N, M);
%     X_sel(k,:) = calcFeature(patch, featureName);
% end

% 4) 5x5 mean smoothing of features (same as your code, but only where we have data)
h = ones(5,5)/25;
M_centers = false(NR,NC); M_centers(cent_lin) = true;

X_sel_sm = zeros(L,F);
for f = 1:F
    Xi = zeros(NR,NC);         % sparse image of feature f at centers
    Xi(cent_lin) = X_sel(:,f);
    Xi = conv2(Xi, h, 'same'); % your original averaging (zeros elsewhere)
    X_sel_sm(:,f) = Xi(cent_lin);
end

% 5) Smooth the valid mask and keep pixels with high support (as you did)
M_proc = conv2(double(M_centers), h, 'same') > 0.95;
keep   = M_proc(cent_lin);                          % centers that survive smoothing
cent_lin_keep = cent_lin(keep);

% 6) Predict only at kept centers
RequiredVariables_sorted = local_naturalSortX(trainedModel.RequiredVariables, "X_");
X_tab = array2table(X_sel_sm(keep,:), 'VariableNames', RequiredVariables_sorted);
y = trainedModel.predictFcn(X_tab);                 % Lkeep x 4

% 7) Scatter to images (NaN elsewhere)
[w_phi, phi_x, phi_y, theta] = deal( nan(NR,NC) );
w_phi(cent_lin_keep) = y(:,1);
phi_x(cent_lin_keep) = y(:,2);
phi_y(cent_lin_keep) = y(:,3);
theta(cent_lin_keep) = y(:,4);

% 8) Quality map (same idea)
diff_mag = abs(w_phi - abs(phi_x + 1i*phi_y));
QM = M_proc .* (1 - mat2gray(diff_mag));

end

% ---- helpers -------------------------------------------------------------

function mustHaveFields(s, requiredFields)
missingFields = setdiff(requiredFields, fieldnames(s));
if ~isempty(missingFields)
    error('Struct is missing required fields: %s', strjoin(missingFields, ', '));
end
end

function out = local_naturalSortX(vnames, prefix)
% Sort {'X_1','X_2',...,'X_10'} in numeric order even if they came lexicographic
v = cellstr(vnames);
nums = zeros(numel(v),1);
for i=1:numel(v)
    t = erase(v{i}, prefix);
    d = sscanf(t, '%d'); if isempty(d), d = i; end
    nums(i) = d;
end
[~,p] = sort(nums);
out = v(p);
end
