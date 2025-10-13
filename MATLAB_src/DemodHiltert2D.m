function z = DemodHiltert2D(g, M, wTh, hilbertFilterDirection)
%DEMODHILTERT2D demodulated input ingram g using 2D hiltert transform
%   AQ 11APR24
%   Copyright 2009 OM4M

arguments
    g (:,:) {mustBeNumeric}; % input igram b+0.5*m*(exp(1i*phi)+exp(-1i*phi))
    M (:,:) {mustBeNumericOrLogical}=ones(size(g)); % input ROI
    wTh (1,1) {mustBeNumeric}=3; % statial freq module threshold in ff, filter spatial freqs bellow wTH
    hilbertFilterDirection string {mustBeMember(hilbertFilterDirection,["X", "Y"])}="X";  % output spatial freqs units
end


%% spatial freqs
%Image size in u and v axis
[NR, NC]=size(g);

%cartesian freq units
[u, v]=meshgrid(1:NC, 1:NR); %cartesian freq space in ff
u0=floor(NC/2)+1; u=u-u0;
v0=floor(NR/2)+1; v=v-v0;
q=abs(u+1i*v);

%% high pass filter all frewq bellow wTh are filtered
H1=1-exp(-0.5*(q/wTh).^2);

switch hilbertFilterDirection
    case "X"
        H=u>0;        
    case "Y"
        H=v>0;        
    otherwise 
        error("not valid filterDirection value ") 
end


%% g's Demod   filter

G=fft2(g); 
G=G.*ifftshift(H.*H1);
z=2*ifft2(G); % m*exp(1i*phi); 

end

