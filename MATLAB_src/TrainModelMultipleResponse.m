%% Regression Learner-style script using fitrnet + validation/test plots
% Adjust this to your Excel file path:

%% clear workspace
close all
clear all

%% parameter

% if ther is a trained model for the selected trainingSet_Idx it will be
% reused
TRAIN_MODEL=true;
% if there is already a trained model or if we do not want to save the
% trained modes
SAVE_MODEL=false;
trainingSet_Idx=7;


%% load data

%root dir for DB (the files)
rootFolderDB="..\ML_Models";%root dir for DB (the files)
trainingSetsDBName = 'DB-trainingSets-OM4M007.xlsx';

trainingSetsDB=fullfile(rootFolderDB, trainingSetsDBName) ;

trainingSetsTb = readtable(trainingSetsDB, 'Sheet', 'Sheet1', 'ReadVariableNames', true, 'Format', 'auto');



DB_info.featureName=trainingSetsTb.featureName{trainingSet_Idx};
DB_info.patch_NC=trainingSetsTb.patch_NC(trainingSet_Idx);
DB_info.patch_NR=trainingSetsTb.patch_NR(trainingSet_Idx);
DB_info.GV_noise_amplitude=trainingSetsTb.GV_noise_amplitude(trainingSet_Idx);
DB_info.NS=trainingSetsTb.NS(trainingSet_Idx);

dateTimeObject=trainingSetsTb.date(trainingSet_Idx);
dateTimeObject.Format = 'dd-MMM-yyyy';
DB_info.date = string(dateTimeObject);  % ["28-Aug-2025"]

DB_info.featureDBFileName=trainingSetsTb.featureDBFileName{trainingSet_Idx};

% Load samples as a table
fileDB=fullfile(rootFolderDB, DB_info.featureDBFileName);
dataTb = readtable(fileDB);
nameList2Exclude={'w','wx','wy','theta', 'fringe_bkgrd', 'fringe_mod'};

% Specify predictors and response, for the NN training
predictorNameList=setdiff(dataTb.Properties.VariableNames, nameList2Exclude);
responseNameList   = {'w','wx','wy','theta'};

% Checks
missingPredictors = setdiff(predictorNameList, dataTb.Properties.VariableNames);
if ~isempty(missingPredictors)
    error('Missing predictor(s) in table: %s', strjoin(missingPredictors, ', '));
end
if ~ismember(responseNameList, dataTb.Properties.VariableNames)
    error('Response variable "%s" not found in the table.', responseNameList{:});
end

%load predictor and response
predictors = dataTb(:, predictorNameList);
%response   = dataTb.(responseName);
responses = dataTb(:, responseNameList);

% Remove rows with missing data
validMask = all(~ismissing(predictors), 2) & all(~ismissing(responses), 2);
predictors = predictors(validMask, :);
responses   = responses(validMask, :);

%% Three-way split: Train (~70%), Validation (15%), Test (15%)
rng(42); % reproducible: initializes the MATLAB® random number generator using fixed seed

% First, carve out the test set (15%)
cvpTest = cvpartition(height(predictors), 'HoldOut', 0.15);
X_notTest = predictors(training(cvpTest), :);
y_notTest = responses(training(cvpTest), :);
X_test    = predictors(test(cvpTest), :);
y_test    = responses(test(cvpTest), :);

% From the remaining (85%), carve out a validation set of ~15% of total.
% 15% of total equals 15/85 ≈ 0.176 of the remainder.
valFractionOfRemainder = 0.15 / 0.85; % ≈ 0.176
cvpVal  = cvpartition(height(X_notTest), 'HoldOut', valFractionOfRemainder);

% final training data
X_train = X_notTest(training(cvpVal), :);
y_train = y_notTest(training(cvpVal), :);
X_val   = X_notTest(test(cvpVal), :);
y_val   = y_notTest(test(cvpVal), :);

fprintf('Split sizes — Train: %d, Val: %d, Test: %d\n', ...
    height(X_train), height(X_val), height(X_test));

%% Fit regression neural network on the TRAIN set

if TRAIN_MODEL
    regressionNeuralNetwork = fitrnet( ...
        X_train, ...
        y_train, ...
        'LayerSizes', [50 50 50 50], ...
        'Activations', 'relu', ...
        'Lambda', 0, ...
        'IterationLimit', 1000, ...
        'Standardize', true);
else %LOAD_TRAINED_MODEL
    %trainedModelFileName="trainedModel_w_wx_wy_theta_feature_projected_DFT_13x13_GVN-0_NS-15000_28-Aug-2025";
    trainedModelFileName=trainingSetsTb.trainedModel{trainingSet_Idx};                          

    rootModelFolder="..\ML_Models";
    trainedModelFileName=fullfile(rootModelFolder, trainedModelFileName);

    S=load(trainedModelFileName);
    regressionNeuralNetwork=S.trainedModel.RegressionNeuralNetwork;
end


% Quick holdout evaluations
y_val_pred  = predict(regressionNeuralNetwork, X_val); %validation
y_test_pred = predict(regressionNeuralNetwork, X_test); %testing

valRMSE  = sqrt(mean((y_val  - y_val_pred ).^2)); %validation
testRMSE = sqrt(mean((y_test - y_test_pred).^2)); %testing

fprintf('Validation RMSE: \n'); disp(valRMSE);
fprintf('Test RMSE:\n'); disp(testRMSE);

%% manual K-fold CV that keeps your multi-response
% the 5-fold cross-validation with crossval only works for single response
% models
opts = struct();
opts.LayerSizes     = regressionNeuralNetwork.LayerSizes;
opts.Activations    = regressionNeuralNetwork.Activations;
opts.IterationLimit = regressionNeuralNetwork.ConvergenceInfo.Iterations;
opts.Standardize    = regressionNeuralNetwork.ModelParameters.StandardizeData;
opts.Lambda         = regressionNeuralNetwork.ModelParameters.Lambda;   % if you set this explicitly

[kpred, rmsePerResp, rmseAll] = kfoldCV_fitrnet_multiresponse(predictors, responses, 5, opts);
fprintf('5-fold CV RMSE (train folds)\n'); disp(rmsePerResp);

%% Validation plots ----------

%transform to arrays
y_val=table2array(y_val);
y_test=table2array(y_test);

for n=1:length(responseNameList)
    response=responseNameList{n};

    % (i) Predicted vs True for validation set
    figure('Name',sprintf('Validation %s: Predicted vs True', response));
    scatter(y_val(:, n), y_val_pred(:, n), 25, 'filled'); grid on;
    xlabel(sprintf('True %s (Validation)', response));
    ylabel(sprintf('Predicted %s (Validation)', response));
    title(sprintf('Validation %s: Predicted vs True (RMSE = %.4f)', response, valRMSE.(response)));
    % reference line y = x
    hold on;
    lims = [min([y_val(:, n); y_val_pred(:, n)]), max([y_val(:, n); y_val_pred(:, n)])];
    plot(lims, lims, 'k--', 'LineWidth', 1); xlim(lims); ylim(lims);
    hold off;


    % (ii) Residuals for validation set (true - pred)
    val_resid = y_val(:, n) - y_val_pred(:, n);
    figure('Name',sprintf('Validation %s: Residuals', response));
    scatter(y_val_pred(:,n), val_resid, 25, 'filled'); grid on;
    xlabel(sprintf('Predicted %s (Validation)', response));
    ylabel(sprintf('%s Residual = True - Predicted', response));
    title(sprintf('Validation %s: Residuals vs Predicted', response));   
    yline(0, 'k--', 'LineWidth', 1);
    ylim([-0.3 0.3]);

    %% Test plots ----------
    % (i) Predicted vs True for validation set
    figure('Name',sprintf('Test %s: Predicted vs True', response));
    scatter(y_test(:, n), y_test_pred(:, n), 25, 'filled'); grid on;
    xlabel(sprintf('True %s (Test)', response));
    ylabel(sprintf('Predicted %s (Test)', response));
    title(sprintf('Test %s: Predicted vs True (RMSE = %.4f)', response, valRMSE.(response)));
    % reference line y = x
    hold on;
    lims = [min([y_test(:, n); y_test_pred(:, n)]), max([y_test(:, n); y_test_pred(:, n)])];
    plot(lims, lims, 'k--', 'LineWidth', 1); xlim(lims); ylim(lims);
    hold off;

    % (ii) Residuals for validation set (true - pred)
    test_resid = y_test(:, n) - y_test_pred(:, n);
    figure('Name',sprintf('Test %s: Residuals', response));
    scatter(y_test_pred(:,n), test_resid, 25, 'filled'); grid on;
    xlabel(sprintf('Predicted %s (Test)', response));
    ylabel(sprintf('%s Residual = True - Predicted', response));
    title(sprintf('Test %s: Residuals vs Predicted', response));
    yline(0, 'k--', 'LineWidth', 1);
    ylim([-0.3 0.3]);

end

%% Package a "trainedModel" struct similar to Regression Learner exports

if TRAIN_MODEL

    trainedModel.RegressionNeuralNetwork = regressionNeuralNetwork;
    trainedModel.predictFcn = @(tbl) predict(trainedModel.RegressionNeuralNetwork, tbl(:, predictorNameList));
    trainedModel.RequiredVariables = predictorNameList;
    trainedModel.About = 'Regression model trained with fitrnet.';
    trainedModel.HowToPredict = sprintf([ ...
        'To make predictions on a new table Tnew, use:\n' ...
        '  yfit = trainedModel.PredictFcn(Tnew)\n' ...
        'Tnew must contain the variables:\n  %s\n'], strjoin(predictorNameList, ', '));
    trainedModel.Date=datetime();
    trainedModel.Info=sprintf("generated by %s", mfilename());
    trainedModel.DB_info=DB_info;

    %same name as DB_info.featureDBFileName but with trainedModel instead of DB
    % trainedModelFileName=sprintf("trainedModel_%s_%ix%i_GVN-%i_NS-%i_%s.mat",...
    %     DB_info.featureName, DB_info.patch_NR,...
    %     DB_info.patch_NR, DB_info.GV_noise_amplitude,...
    %     DB_info.NS, DB_info.date);
    [~, fname, ~] = fileparts(DB_info.featureDBFileName);
    responsesStr = strjoin(responseNameList, '_');
    trainedModelFileName=strrep(fname, 'DB_', sprintf("trainedModel_%s_%s_", string(datetime('today')), responsesStr));

    trainedModelFileName=fullfile(rootFolderDB, trainedModelFileName);

    
    disp(trainedModelFileName)
    disp(DB_info.featureDBFileName)

    if SAVE_MODEL
        save(trainedModelFileName, 'trainedModel');
    end
end