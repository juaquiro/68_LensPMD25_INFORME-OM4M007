function [oofPred, rmsePerResp, rmseOverall] = kfoldCV_fitrnet_multiresponse(X, Y, kfold, opts)
% Manual K-fold CV for multi-response fitrnet.
% X: N×P predictors (table or matrix compatible with fitrnet)
% Y: N×R responses (numeric matrix, R>=2)
% kfold: number of folds
% opts: struct with fields mirroring fitrnet name-value pairs, e.g.:
%   opts.LayerSizes = [10 10 10];
%   opts.Activations = 'relu';
%   opts.IterationLimit = 1000;
%   opts.Standardize = true;
%
% Returns:
%   oofPred: N×R out-of-fold predictions
%   rmsePerResp: 1×R RMSE computed over all OOF preds
%   rmseOverall: scalar RMSE over all responses concatenated

    if nargin < 4, opts = struct; end
    if ~isfield(opts,'LayerSizes'),      opts.LayerSizes = [10 10 10]; end
    if ~isfield(opts,'Activations'),     opts.Activations = 'relu';    end
    if ~isfield(opts,'IterationLimit'),  opts.IterationLimit = 1000;   end
    if ~isfield(opts,'Standardize'),     opts.Standardize = true;      end

    N = size(X,1);
    R = size(Y,2);
    oofPred = NaN(N, R);

    rng(42);
    cvp = cvpartition(N, 'KFold', kfold);

    for f = 1:kfold
        tr = training(cvp, f);
        te = test(cvp, f);

        % Train on fold's training split with multi-response Y
        mdl = fitrnet( ...
            X(tr,:), Y(tr,:), ...
            'LayerSizes',    opts.LayerSizes, ...
            'Activations',   opts.Activations, ...
            'IterationLimit',opts.IterationLimit, ...
            'Standardize',   opts.Standardize);

        % Predict on held-out fold
        oofPred(te, :) = predict(mdl, X(te,:));
    end

    % Compute per-response RMSE using all out-of-fold predictions
    resid = Y - oofPred;                 % N×R
    msePerResp = mean(resid.^2, 1, 'omitnan');
    rmsePerResp = sqrt(msePerResp);

    % Overall RMSE across all responses/rows
    %resid is a Nx4 table, transform it to matrix
    resid_mat=table2array(resid);
    rmseOverall = sqrt(mean(resid_mat(:).^2, 'omitnan'));
end
