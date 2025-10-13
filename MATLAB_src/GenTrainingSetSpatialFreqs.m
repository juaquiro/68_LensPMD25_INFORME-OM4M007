%% Script para la generacion de conjutos de entrenamiento para la medida de la frecuencia espacial local
% mediante el uso de metodos de aprendizaje supervisado para regresion
% AQ V2 8AUG25

%% clear enviroment
clear all
close all

%% Params

SAVE_TRAINING_DATA_TO_DB=true;
RANDOM_FRINGE_MODULATION=true;

% tamaño de los parches
NR=15; %rows px
NC=15; %cols px
[x,y]=meshgrid(1:NC, 1:NR);
x0=floor(NC/2)+1; y0=floor(NR/2)+1; %image center
x=x-x0; y=y-y0;

FT_axes_ffx=x(1, :); %Fourier transform x axes in fringes/field
FT_axes_ffy=y(:, 1); %Fourier transform y axes in fringes/field



% rangos de variacion igrams
GV_max=250; % max GV
GV_min=5; % min GV
fringe_mod_min=10; % GV minimum fringe modulation

% rango de variacion de ruido aditivo
GV_noise_amplitude=0; %additive noise level amplitude GV

% Metodo de seleccion de las frecuencias para el conjunto de entrenamiento
metodoSeleccionFreqs='random';



%% genera la lista de parches 
%spatial freqs
switch metodoSeleccionFreqs
    case 'random'
        NS=1000; %tamaño conjunto entrenamiento metodo random de generacion de spatial freqs

        w=pi*rand(1, NS); %module of spatial frec [0, pi] rad/sample
        theta=pi*(rand(1, NS)-0.5); %fringe orientation, [-pi/2, pi/2]
        wx=w.*cos(theta); % spatial freq x only positive random spatial freqs [0, pi] rad/px
        wy=w.*sin(theta);  % spatial freq y random spatial freqs [-pi, pi] rad/px

        ffx=NC*wx/(2*pi); %spatial freq x fringes/field
        ffy=NR*wy/(2*pi); %spatial freq y fringes/field

    case 'equispaced'
        Dffx=0.1; %x freq spacing in fringes/field
        Dffy=0.1; %x freq spacing in fringes/field
        [ffx, ffy]=meshgrid(0:Dffx:floor(NC/2), -floor(NR/2):Dffy:floor(NR/2)); %spatial freqs in fringes/field

        wx=2*pi*ffx/NC; %spatial freq x rad/sample
        wy=2*pi*ffy/NR; %spatial freq y rad/sample

        w=abs(wx+1i*wy); %module of spatial frec [0, pi] rad/sample
        theta=atan2(-wy, wx);

        %filter feasible values for w and vectorize
        k=(w<=pi); % w can not be bigger that pi
        w=w(k)';
        theta=theta(k)';
        wx=wx(k)';
        wy=wy(k)';
        ffx=ffx(k)';
        ffy=ffy(k)';

        NS=length(w);
end

%% features
%featureNameList={"feature_GV", "feature_DFT", "feature_projected_DFT"};
featureNameList={"feature_projected_DFT"};

featureVectorList=cell(NS, length(featureNameList)); % here we will store all featureTypes for all samples



%% show spatial freqs sampling
% fringes field
figure; plot(ffx, ffy, '+'); axis equal;
xlabel('\omega_x (ff)'); ylabel('\omega_y (ff)'); title('Spatial freqs samples');
xlim([0, floor(NC/2)]); ylim([-floor(NC/2), floor(NC/2)])
grid on
axis equal;

%rad/px
figure; plot(wx, wy, '+');
axis equal; xlabel('\omega_x (rad/px)'); ylabel('\omega_y (rad/px)'); title('Spatial freqs samples')
xlim([0, pi]); ylim([-pi, pi])
grid on
axis equal;

%fringe period
px=2*pi./wx; % fringe period px
py=2*pi./wy; %fringe period px

% generar b y m aleatorios para que igram salga entre max_GV y min_GV
% es decir b+m<maxGV, b-m>minxGV

% Generate random m values within a feasible range
% if fringe_mod<min_fringe_mod
if RANDOM_FRINGE_MODULATION
    fringe_mod = rand(1, NS) * (GV_max - GV_min) / 2; %fringe modulation GV
    while any(fringe_mod<fringe_mod_min)
        k=find(fringe_mod<fringe_mod_min);
        fringe_mod(k)=rand(1, length(k)) * (GV_max - GV_min) / 2;
    end

else
    fringe_mod = ones(1, NS) * (GV_max - GV_min) / 2; %fringe modulation GV
end

% Define corresponding b values ensuring the constraints hold
fringe_bkgrd_min = fringe_mod + GV_min;  % Ensures fringe_bkgrd - fringe_mod >= min_GV
fringe_bkgrd_max = GV_max - fringe_mod;  % Ensures fringe_bkgrd + fringe_mod <= max_GV

% Generate fringe_bkgrd within the feasible range
fringe_bkgrd = fringe_bkgrd_min + (fringe_bkgrd_max - fringe_bkgrd_min) .* rand(1, NS);

% Verify constraints
assert(all(fringe_bkgrd + fringe_mod <= GV_max), 'Constraint b + m <= max_GV violated!');
assert(all(fringe_bkgrd - fringe_mod >= GV_min), 'Constraint b - m >= min_GV violated!');

%aditive noise level GV for each NRxNC patch
GV_noise = rand(NR, NC, NS) * GV_noise_amplitude; %uniform distribution [0, GV_noise_amplitude]

% [0-2pi] arbirtary phase shift 
phase_shift=2*pi*rand(1, NS);

hw=waitbar(0);
for ns=1:NS
    g=uint8(fringe_bkgrd(ns)+fringe_mod(ns)*cos(phase_shift(ns)+x*wx(ns)+y*wy(ns))+GV_noise(:, :, ns));
    

    % %fringe pattern
    % hg=figure('Position', [100, 100, 800, 800]); % Set window size;
    % imshow(g, 'InitialMagnification', 'fit'); % Keep the scaling
    % str_f = sprintf('period:p=(%0.2f,%0.2f) px, modulation m=%0.2f GV',px(ns), py(ns), fringe_mod(ns));
    % title(str_f);
    % set(gca, 'Position', [0.05 0.1 0.9 0.85]); %make room for the title
    % drawnow;
    % 
    % %fourier transform
    % G=fft2(g); %DFT coeffs NRxNC
    % G(1,1)=0;
    % hG=figure('Position', [100, 100, 800, 800]); % Set window size;
    % imagesc(FT_axes_ffx, FT_axes_ffy, abs(fftshift(G))); % Keep the scaling
    % str_f = sprintf('statial freq: w=(%0.2f,%0.2f) ff, modulation m=%0.2f GV',ffx(ns), ffy(ns), fringe_mod(ns));
    % title(str_f);
    % %set(gca, 'Position', [0.05 0.1 0.9 0.85]); %make room for the title
    % drawnow;


    % calculate features
    for ft=1:length(featureNameList)
        featureVectorList{ns, ft}=calcFeature(g, featureNameList{ft});
    end




    waitbar(ns/NS, hw, 'calculando features')

    %pause()
    % close(hg);
    % close(hG);
end
close(hw)

%% prepare DB representation as a table object

% all training data organized as a table object
training_data_tb=table(wx', wy', w', theta', fringe_bkgrd', fringe_mod', 'VariableNames', ["wx", "wy", "w", "theta", "fringe_bkgrd", "fringe_mod"]);
for ft=1:length(featureNameList)
    training_data_tb.(featureNameList{ft})=cell2mat(featureVectorList(:, ft));
end


% save all training data and params to a MAT file
%dataFilename=rootDataName + "_" + string(datetime('today')) + "_DB.mat";
%save(dataFilename);

%save features to excel files

 % ft == feature type
 hw=waitbar(0, 'creando DBs');
 training_data_feature_tb_List=cell(1, length(featureNameList));
 training_data_Params_tb=table(NR, NC, GV_noise_amplitude,...
         GV_max, GV_min, fringe_mod_min, string(metodoSeleccionFreqs), NS, ...
         'VariableNames', ["NR", "NC", "GV_noise_amplitude",...
         "GV_max", "GV_min", "fringe_mod_min", "metodoSeleccionFreqs", "NS"]);


 if SAVE_TRAINING_DATA_TO_DB
 %root dir for DB (the files) 
 rootFolder="..\local_data\ML_Models";

 for ft=1:length(featureNameList)

     training_data_feature_tb_List{ft}=table(wx', wy', w', theta', fringe_bkgrd', fringe_mod', 'VariableNames', ["wx", "wy", "w", "theta", "fringe_bkgrd", "fringe_mod"]);
     training_data_feature_tb_List{ft}.X=training_data_tb.(featureNameList{ft});
     %para evitar crear nombres como "X_ 1" con un espacio que genera
     %warnings vamos a expandir el campo X de forma que ya salda X_1, X_2,
     %antes de salvarlo al excel
     training_data_feature_tb_List{ft}=expandMatrixField(training_data_feature_tb_List{ft}, 'X');

     featureDBFileName=sprintf("DB_%s_%ix%i_GVN-%i_NS-%i_%s.xlsx", featureNameList{ft}, NR, NC, GV_noise_amplitude, NS, string(datetime('today')));

     featureDBFileName=fullfile(rootFolder, featureDBFileName);
     writetable(training_data_feature_tb_List{ft}, featureDBFileName, 'Sheet', 1);
     writetable(training_data_Params_tb, featureDBFileName, 'Sheet', 2);

     waitbar(ft/length(featureNameList),hw, 'creando DBs');
     disp(featureDBFileName)

 end

 end
 close(hw)


