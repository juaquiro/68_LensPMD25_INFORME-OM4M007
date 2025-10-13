%script for checking and visualize features
%AQ 14AUG25
% clear
close all
clear all

%% params
PLOT_FETAURES=true;

%% spatial freqs

featCase=1;
switch featCase
    case 1
        w=2*ones(1,3); %module of spatial frec [0, pi] rad/sample
        w=2*pi*4/13*ones(1,3);
        theta=[-45, 0, 45]*pi/180; %fringe orientation, [-pi/2, pi/2]
        wx=w.*cos(theta); % spatial freq x only positive random spatial freqs [0, pi] rad/px
        wy=w.*sin(theta);  % spatial freq y random spatial freqs [-pi, pi] rad/px
    case 2
        fileDB = ".\ML_Models\DB_feature_projected_DFT_13x13_GVN-0_NS-10000_24-Aug-2025.xlsx";
        dataTb = readtable(fileDB, 'Sheet', 'w3');
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
        responses = dataTb(:, responseNameList);

        w=responses.w';
        wx=responses.wx';
        wy=responses.wy';
        theta=responses.theta';
end

%fringe period
px=2*pi./wx; % fringe period px
py=2*pi./wy; %fringe period px


NS=length(w);

%% features Visualization
%featureTypeNameList={"feature_GV", "feature_DFT", "feature_nonNormalized_DFT", "feature_projected_DFT"};
featureNameList={"feature_projected_DFT"};
featureVectorList=cell(NS, length(featureNameList)); % here we will store all featureTypes for all samples

% igrams
NR=13; %rows px
NC=13; %cols px
[x,y]=meshgrid(1:NC, 1:NR);
x0=floor(NC/2)+1; y0=floor(NR/2)+1; %image center
x=x-x0; y=y-y0;
signY=sign(y);

FT_axes_ffx=x(1, :); %Fourier transform x axes in fringes/field
FT_axes_ffy=y(:, 1); %Fourier transform y axes in fringes/field

ffx=NC*wx/(2*pi); %spatial freq x fringes/field
ffy=NR*wy/(2*pi); %spatial freq y fringes/field

%aditive noise level GV for each NRxNC patch
GV_noise_amplitude=0;
GV_noise = rand(NR, NC, NS) * GV_noise_amplitude; %uniform distribution [0, GV_noise_amplitude
% [0-2pi] arbirtary phase shift
phase_shift=0*2*pi*rand(1, NS);
%phase_shift=[0, pi/3, pi/2];


fringe_bkgrd=100*ones(1, NS);
fringe_mod=50*ones(1,NS);

for ns=1:NS
    g=uint8(fringe_bkgrd(ns)+fringe_mod(ns)*cos(phase_shift(ns)+x*wx(ns)+y*wy(ns))+GV_noise(:, :, ns));
    for ft=1:length(featureNameList)
        [featureVectorList{ns, ft}, S]=calcFeature(g, featureNameList{ft});
    end

    if PLOT_FETAURES

        %fringe pattern
        hg=figure('Position', [100, 100, 800, 800]); % Set window size;
        imshow(g, 'InitialMagnification', 'fit'); % Keep the scaling
        axis equal;
        str_f = sprintf('period:p=(%0.2f,%0.2f) px, modulation m=%0.2f GV',px(ns), py(ns), fringe_mod(ns));
        title(str_f);
        set(gca, 'Position', [0.05 0.1 0.9 0.85]); %make room for the title
        drawnow;

        %fourier transform
        hG=figure('Position', [100, 100, 800, 800]); % Set window size;
        imagesc(FT_axes_ffx, FT_axes_ffy, abs(S.G)); % Keep the scaling
        axis equal;
        str_f = sprintf('statial freq: w=(%0.2f,%0.2f) ff, modulation m=%0.2f GV',ffx(ns), ffy(ns), fringe_mod(ns));
        title(str_f);
        %set(gca, 'Position', [0.05 0.1 0.9 0.85]); %make room for the title
        drawnow;

        hG=figure('Position', [100, 100, 800, 800]); % Set window size;
        imagesc(FT_axes_ffx(x0:end), FT_axes_ffy, S.abs_G_sp); % Keep the scaling
        axis equal; axis tight;
        str_f = sprintf('statial freq: w=(%0.2f,%0.2f) ff, modulation m=%0.2f GV',ffx(ns), ffy(ns), fringe_mod(ns));
        title(str_f);
        %set(gca, 'Position', [0.05 0.1 0.9 0.85]); %make room for the title
        drawnow;

        figure;
        idx = find([featureNameList{:}] == "feature_projected_DFT");
        plot(featureVectorList{ns, idx});  title(str_f); grid; drawnow;
    end

end

%% spatial freqs estimation using Load trained model

% Root folder with the DB ioncluding trainingSets and Trained Models 

rootFolderDB="..\local_data\ML_Models";%root dir for DB (the files)
trainingSetsDBName = 'DB-trainingSets-OM4M007.xlsx';

trainingSetsDB=fullfile(rootFolderDB, trainingSetsDBName) ;

trainingSetsTb = readtable(trainingSetsDB, 'Sheet', 'Sheet1', 'ReadVariableNames', true, 'Format', 'auto');

% select trained model from DB
trainingSet_Idx=7;
trainedModelFileName=trainingSetsTb.trainedModel{trainingSet_Idx};       
rootModelFolder="..\local_data\ML_Models";
trainedModelFileName=fullfile(rootModelFolder, trainedModelFileName);
sprintf("Loaded Trained Model: %s", trainedModelFileName)

S=load(trainedModelFileName);

X_mat = cell2mat(cellfun(@(x) x(:)', featureVectorList, 'UniformOutput', false));

%AQ26AUG25 OJO S.trainedModel.RequiredVariables vienen en orden
%lexicografico {'X_1'}    {'X_10'}    {'X_11'}    {'X_12'}    
%y aqui hace falta que esten en orden numerico {'X_1'}    {'X_2'}    {'X_3'}  
%para que las columnas de X_mat vayan a la variable de prediccion apropiada
RequiredVariables_sorted=naturalSort(S.trainedModel.RequiredVariables, 'X_');
X_tb=array2table(X_mat, 'VariableNames', RequiredVariables_sorted);

y_pred_mat=S.trainedModel.predictFcn(X_tb);

y_pred_tb=array2table(y_pred_mat, 'VariableNames', S.trainedModel.RegressionNeuralNetwork.ResponseName);

y_act_mat=[w', wx', wy', theta'];
y_act_tb=array2table(y_act_mat, 'VariableNames', S.trainedModel.RegressionNeuralNetwork.ResponseName);

disp("estimated values")
disp(y_pred_tb);

disp("expected values")
disp(y_act_tb);

disp("error")
disp(y_act_tb-y_pred_tb);







