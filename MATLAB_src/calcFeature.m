function [featureVector, S] = calcFeature(g, featureName)
% calcFeature(g, featureName) calculates the feature "featureName" of the [NRxNC] patch
% *featureGV is the 1x[NR*NC] vectorized normalized version of the GV
% *feature_DFT is the 1x(NR*[floor(0.5*NC)+1]) DFT normalized abs coefiecients 
% *feature_projectedDFT is the normalized 1x[NR + floor(0.5*NC)+1] DFT abs coefs [Proj(wx) Proj(wy)]
% before calculating the features we filter out noise with a gaussian
% filter
% V1 15AUG25 three features 
% V2 28AUG25 add a spatial-domain filter to minimize border effects and
% make the patch more wavelet-like
arguments
    g (:,:) {mustBeNumeric}; % input igram
    featureName string {mustBeMember(featureName,["feature_GV", "feature_DFT", "feature_projected_DFT"])}="feature_projected_DFT";  % feature name
end

[NR,NC]=size(g);
%AQ NOTE 28AUG25 all these operations can be make before feature
%calculation
x0=floor(NC/2)+1; y0=floor(NR/2)+1; %image center
[x,y]=meshgrid(1:NC, 1:NR);
x=x-x0; y=y-y0;

%apodization mask, spatial window for minimizing border effects and filter noise in freq
%space
sigma_x=NC/3;
sigma_y=NR/3;
W=exp(-0.5*x.^2/sigma_x^2 -0.5*y.^2/sigma_y^2 );
g=double(g);
g=g-mean(g(:)); %reduce elliminate DC
g=g.*W; %apply window

% DFT
G=fft2(g); %DFT coeffs NRxNC
G(1,1)=0; %DC filtering
G=fftshift(G); %shift for facilitate interpretation

%filter hifreq noise
G=G.*W;

%feature calculation
GNorm=sum(abs(G(:))); %for a monochromatic signal GNorm==m
%G_sp semiplane wx>0
abs_G_sp=abs(G(:, x0:end)); %semiplane wx>=0 DFT NRx(floor(0.5*NC)+1)

% %check for the position of the peak delta(wy-wy0)
% %in any case row==y0 always pass just in case the deltas lye on it
% if theta_mu>0
%     %if the peak delta(wy-wy0) is located in the wy>0 section null the wy<0 semiplane
%     abs_G_sp(1:y0-1, :)=0;
% else
%     %if the peak delta(wy-wy0) is located in the wy<=0 plane, null the wy>0 semiplane
%     abs_G_sp(y0+1:end, :)=0;
% end

%check for location of the delta(wy-wy0)
if sum(sum(abs_G_sp(1:y0-1, :)))>sum(sum(abs_G_sp(y0+1:end, :)))
    %if the peak delta(wy-wy0) is located in the wy>0 section null the wy<0 semiplane
    abs_G_sp(y0+1:end, :)=0;
else
    %if the peak delta(wy-wy0) is located in the wy<=0 plane, null the wy>0 semiplane
    abs_G_sp(1:y0-1, :)=0;
end

%G_sp_XP G semiplane wx>0 X projected
G_sp_XP=sum(abs_G_sp, 1); %DFT coefs 1x(floor(0.5*NC)+1)
%G_sp_YP G semiplane wx>0 Y projected
G_sp_YP=sum(abs_G_sp, 2)'; %DFT coefs 1xNR


switch featureName
    case "feature_GV"
        muGV = mean(double(g(:)));
        sigmaGV = std(double(g(:)));
        featureVector=(double(g(:)')-muGV)/sigmaGV; %normalized GV [-1, 1] 1x(NRxNC)
    case "feature_DFT"
        featureVector=abs(abs_G_sp(:)')/GNorm;  %DFT coefs 1x(NR*[floor(0.5*NC)+1])
    case "feature_projected_DFT"
        featureVector=[G_sp_XP, G_sp_YP]/GNorm ; %DFT coefs 1x[NR + floor(0.5*NC)+1]
end
%figure; plot(features); figure(gcf)

S.G=G;
S.abs_G_sp=abs_G_sp;
S.G_sp_XP=G_sp_XP;
S.G_sp_YP=G_sp_YP;

end


