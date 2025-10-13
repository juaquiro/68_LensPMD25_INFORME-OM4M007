%% This script verifies a previously trained model created with TrainModelMultipleResponse
%  (Solo comentarios corregidos/clarificados; el código permanece igual)

%% clear

clear all
close all

%% Parameters

%size
NR=511;
NC=512;
%spatial carrier
w0_x=pi/4;
w0_y=pi/4;
[x,y]=meshgrid(1:NC, 1:NR); x=x-0.5*NC; y=y-0.5*NR;
% modulating phase
p=peaks(NR); p=imresize(p, [NR, NC]);

%total phase phi
phi=p + pi/2*x + pi/2*y;

% rescale
phi=imresize(phi, 1);
[NR, NC]=size(phi);

% generate fringe pattern (8-bit)
M_ROI=abs(x+1i*y)<0.4*NR; 
%M_ROI=ones(size(phi));
g=uint8(M_ROI.*(100+40*cos(phi)+2*randn(size(phi))));

% ground truth spatial freqs and orientation angle for comparison
[phi_x, phi_y]=gradient(phi); %local components of the spatial freqs 
w_phi=abs(phi_x+1i*phi_y); %local spatial freq
theta=atan2(-phi_y, phi_x); % fringe orientation

%% Load trained model

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
featureName=S.trainedModel.DB_info.featureName;
trainedModel=S.trainedModel;


%% calculate spatial freqs

tic
[pred_w_phi, pred_phi_x, pred_phi_y, pred_theta, QM, M_proc]=calcSpatialFreqsSupervisedRegressionBatch(g, trainedModel, featureName , M_ROI);
toc


%% Plot results
MNan=M_proc./M_proc; %when MQ==0 MNan=nan

figure('Name','fringe pattern');
imagesc(g); colormap gray
title('fringe pattern')

figure('Name','pred \phi_x');
imagesc(pred_phi_x.*MNan);
title('predicted \phi_x ML')

figure('Name','ºphi_x');
imagesc(phi_x);
title('ground truth \phi_x')

figure('Name','predicted \phi_y');
imagesc(pred_phi_y.*MNan);
title('predicted \phi_y ML')

figure('Name','\phi_y');
imagesc(phi_y);
title('ground truth \phi_y')

figure('Name','w_\phi');
imagesc(pred_w_phi.*MNan);
title('predicted w_\phi ML')

figure('Name','w_\phi');
imagesc(w_phi);
title('ground truth w_\phi')

figure('Name','predicted \theta');
imagesc(pred_theta.*MNan);
title('predicted \theta ML')

figure('Name','\theta');
imagesc(theta);
title('predicted \theta')


figure('Name','QM');
imagesc(QM);
title('Quality Map for ML estimation')

% Histograms on valid pixels
histEdge1=linspace(-pi, pi, 100);
figure; histogram(w_phi(M_proc), histEdge1); title('hist(w_\phi)'); xlabel('w_\phi rad/px')
figure; histogram(phi_x(M_proc), histEdge1); title('hist(\theta)'); xlabel('\theta rad/px')
figure; histogram(phi_x(M_proc), histEdge1); title('hist(\phi_x)'); xlabel('\phi_x rad/px')
figure; histogram(phi_y(M_proc), histEdge1); title('hist(\phi_y)'); xlabel('\phi_y rad/px')

histEdge2=linspace(-pi/10, pi/10, 100);
figure; histogram(phi_x(M_proc)-pred_phi_x(M_proc), histEdge2); title('hist(error \phi_x)'); xlabel('\phi_x rad/px')
figure; histogram(phi_y(M_proc)-pred_phi_y(M_proc), histEdge2); title('hist(error \phi_y)'); xlabel('\phi_y rad/px')
figure; histogram(w_phi(M_proc)-pred_w_phi(M_proc), histEdge2); title('hist(error w_\phi)'); xlabel('w_\phi rad/px')
figure; histogram(theta(M_proc)-pred_theta(M_proc), histEdge2); title('hist(error \theta)'); xlabel('\theta rad/px')

