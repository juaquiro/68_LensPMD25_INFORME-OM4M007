% ======================================================================
% phasor filtering and direct gradient calculation
% this function calculates the gradient [phix, phiy] of a phasor's phase, z=b*exp(1i*phi). It uses the same sign convetion that MATLAB gradient().
%> Returns a the phase gradient and a Mask with valid differences. For this we use a median filter for outlier removal and gradient after phasor filtering
%>
%> Nmed median filter size for phase only cosine-sine filtering
%> NS 2*NS+1 is the neigbouhoord size for phasor filtering
%> M ROI with valid points
%> z input phasor z=b*exp(1i*phi)
%> LPCycles are the number of low pass cycles that we apply
%> to the calculated derivatives
%> Mxy ROI with valid differences
%> phix phase x-gradient in px^-1
%> phiy phase y-gradient in px^-1
% ======================================================================
function [phix, phiy, Mxy]=phaseGradient(z, M, NS, Nmed, LPCycles)

%set arguments type and defaults values
arguments
    z (:,:) {mustBeNumeric} % phasor z=b*exp(1i*phi)
    M (:,:) {mustBeNumericOrLogical} %M ROI with valid points
    NS (1,1) {mustBeNumeric} = 5 % 2*NS+1 is the neigbouhoord size for phasor filtering
    Nmed (1,1) {mustBeNumeric} = 2 % median filter size for phase only cosine-sine filtering
    LPCycles (1,1) {mustBeNumeric} = 2 % number of low pass cycles that we apply
end

%get phahor size
[NR, NC]=size(z);

%dx indexes
A=[2:NC NC];
B=[1 1:NC-1];

%dy indexes
C=[2:NR NR];
D=[1 1:NR-1];

%by definition set borders to zero for 1st diferences
M(:, 1:3)=0;
M(1:3, :)=0;
M(NR-2:NR, :)=0;
M(:, NC-2:NC)=0;

%dx
%calculate 1st difference
zd=z(:, A)./z(:, B);
zd(isnan(zd))=0;
phix=0.5.*angle(zd);

%dy,
%calculate 1st difference
zd=z(C, :)./z(D, :);
zd(isnan(zd))=0;
phiy=0.5.*angle(zd);

%filter derivatives, if the phasor is well sampled they shuold
%be continuous and the filtering does not depend strongly on the fringe
%period of the phasor
%medfilt for outliers
if (Nmed>0)
    phix=medfilt2(phix, [Nmed, Nmed]);
    phiy=medfilt2(phiy, [Nmed, Nmed]);
end
%mask for the diferences
Mxy=M(:, A).*M(:, B).*M.*M(C, :).*M(D, :);
%low pass filter
h=ones(2*NS+1)/(2*NS+1)^2;
for n=1:LPCycles
    phix=conv2(phix, h, 'same');
    phiy=conv2(phiy, h, 'same');
    Mxy=conv2(Mxy, h, 'same');
end

Mxy=(Mxy>0.999);

%trim results with ROI
phix=Mxy.*phix;
phiy=Mxy.*phiy;

end