function [w, wx, wy, MH]=calcSpatialFreqsHilbert2D(g, M, wTh, hilbertFilterDirection, filterFreqsFlag, mn)
% calcSpatialFreqsHilbert2D(g, M, wTh, hilbertFilterDirection) calculates
% the spatial freq module w and its components wx and wy all in rad/px for
% the input igram g with ROI M, wTH is the threshold to filter out low
% freqs and hilbertFilterDirection is the hilbert filter direction

arguments
    g (:,:) {mustBeNumeric}; % input igram
    M (:,:) {mustBeNumericOrLogical}=ones(size(g)); % input ROI
    wTh (1,1) {mustBeNumeric}=5; % statial freq module threshold in ff, filter spatial freqs bellow wTH
    hilbertFilterDirection string {mustBeMember(hilbertFilterDirection,["X", "Y"])}="X";  % output spatial freqs units
    filterFreqsFlag (1,1) {mustBeNumericOrLogical}=true;  % output spatial freqs filter flag 
    mn (2,1) {mustBeNumeric}=[15 15]; %filter size for spatial freqs
end

z=DemodHiltert2D(g, M, wTh, hilbertFilterDirection);

NS=round(mean(mn)); % 2*NS+1 is the neigbouhoord size for phasor filtering
Nmed=3; % median filter size for phase only cosine-sine filtering
LPCycles=3;
[wx, wy, MH]=phaseGradient(z, M, NS, Nmed, LPCycles); %rad/px
w=abs(wx+1i*wy);

