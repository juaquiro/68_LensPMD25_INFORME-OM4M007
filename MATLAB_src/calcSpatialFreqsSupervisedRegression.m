function [w_phi, phi_x, phi_y, theta, QM, M_proc]=calcSpatialFreqsSupervisedRegression(g, trainedModel, featureName, M_ROI)
% calcSpatialFreqsSupervisedRegression calculates the local spatial freqs,
% phi_x, phi_y and the module w_phi all in rad/px for the input igram g
% with ROI M, using as feature featureName and a matching the trained supervised regression model trainedModel
% this method works better for spatial carrier fringe patters with a
% carrier oriented about 45º
arguments
    g (:,:) {mustBeNumeric} % input igram
    trainedModel struct {mustHaveFields(trainedModel, {'predictFcn', 'DB_info'})} %trainedModel
    featureName string {mustBeMember(featureName, ["feature_GV", "feature_DFT", "feature_projected_DFT"])}="feature_projected_DFT"  % output spatial freqs units
    M_ROI (:,:) {mustBeNumericOrLogical}=ones(size(g)) % input ROI 
end

%AQ 27/9/2025 Patch for name change
if featureName=="feature_normalized_DFT"
    featureName="feature_DFT";
end

%check for featureName, trainedModel matching
if not(strcmp(featureName, trainedModel.DB_info.featureName))
    error('FeatureName does not match trainedModel');
end

%get patch size
patch_NR=trainedModel.DB_info.patch_NR;
patch_NC=trainedModel.DB_info.patch_NC;
x0=floor(patch_NC/2)+1; y0=floor(patch_NC/2)+1; %patch center

% For even/odd patch sizes, include a one-pixel offset to keep symmetry
if rem(patch_NR,2)==0
    rowOffset=1;
else
    rowOffset=0;
end
if rem(patch_NC,2)==0
    colOffset=1;
else
    colOffset=0;
end

patchRowsIdx=-y0+1:y0-1-rowOffset;
patchColsIdx=-x0+1:x0-1-colOffset;

% Extract patches and build feature matrix X_mat
[NR, NC]=size(g);

number_of_features=length(trainedModel.RequiredVariables);
X_mat=zeros(NR*NC, number_of_features);
M_proc=zeros(size(g)); %processed points
for n=y0:NR-y0+1+rowOffset
    % AQDEBUG    n
    for m=x0:NC-x0+1+colOffset
        %check if all points in patch have 
        if all(M_ROI(n+patchRowsIdx, m+patchColsIdx))
            patch_g=g(n+patchRowsIdx, m+patchColsIdx);
            X_mat(sub2ind([NR, NC],n,m), :)=calcFeature(patch_g, featureName);
            M_proc(n,m)=1;
        end
    end
end

% Smooth each feature plane with a 5x5 mean filter
%reshape X_mat as a NRxNCxnumer_of_features to process evert feature as a
%layer
X_layers_mat = reshape(X_mat, NR, NC, number_of_features);
for n=1:number_of_features
    X_layers_mat(:,:, n)=conv2(X_layers_mat(:,:, n), ones(5,5)/25, 'same');
end

%bring back X_layers_mat to X_mat
X_mat=reshape(X_layers_mat, NR*NC, number_of_features);

% Smooth the valid-mask M and keep pixels with high support
M_proc=conv2(M_proc, ones(5,5)/25, 'same');
M_proc=(M_proc>0.95);

% Predict

% AQ 26-AUG-2025 NOTE:
% trainedModel.RequiredVariables are listed in lexicographic order,
% e.g., {'X_1','X_10','X_11','X_12',...}
% but we need *numeric* order {'X_1','X_2','X_3',...}
% so that X_mat columns map to the correct predictor variables.
RequiredVariables_sorted=naturalSort(trainedModel.RequiredVariables, 'X_');
X_tab=array2table(X_mat, 'VariableNames', RequiredVariables_sorted);
y_response=trainedModel.predictFcn(X_tab);

% Recover responses as images
w_phi=reshape(y_response(:,1), [NR, NC]);
phi_x=reshape(y_response(:,2), [NR, NC]);
phi_y=reshape(y_response(:,3), [NR, NC]);
theta=reshape(y_response(:,4), [NR, NC]);

% Simple quality map comparing consistency of w vs wx and wy
QM=M_proc.*(1-mat2gray(abs(w_phi-abs(phi_x+1i*phi_y))));

end

% auxiliary function to validate requiredFields in struct s for the
% arguments block
function mustHaveFields(s, requiredFields)
% Check if the struct has all required fields
missingFields = setdiff(requiredFields, fieldnames(s));
if ~isempty(missingFields)
    error('Struct is missing required fields: %s', strjoin(missingFields, ', '));
end
end





