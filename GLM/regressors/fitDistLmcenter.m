function lmcenter_predictor = fitDistLmcenter(behav_ds, track_len)
% based on fit_distances_lmcenter in fig_response_GLM.py
% returns spatial predictors based on distance from landmark and updated
% roi_gcamp
TRACK_START = 0;
BINNR_SHORT = 80;
BINNR_LONG = 100;
TRACK_END_SHORT = 400;
TRACK_END_LONG = 400;
TRACK_SHORT = 3;
TRACK_LONG = 4;
NUM_PREDICTORS = 20;

bins_short = 18;
animal_location_short_idx = behav_ds(:,5) == track_len; % behav_ds[:,4] in python
animal_location = behav_ds(animal_location_short_idx,2); % behav_ds[animal_location_short_idx,1] in python

% matrix that holds predictors
lmcenter_predictor = zeros(size(animal_location, 1), bins_short);
lmcenter_loc = 220;

% calculate bin edges and centers
boxcar_short_edges = linspace(-180,180,bins_short+1);

% gaussian kernel
gauss_kernel = gausswin(120,4);
sz_bb = 1:120;
gauss_kernel = gausswin(120,4)/trapz(sz_bb, gauss_kernel);

%create boxcar vectors and convolve with gaussian
%{
single_pred = false;
if single_pred
    for i = 1:length(boxcar_short_centers)
        % calculate single predictor
        predictor_x = zeros(1,TRACK_END_SHORT);
        start = int16(boxcar_short_centers(i)-binsize_short/2 + 1);
        fin = int16(boxcar_short_centers(i)+binsize_short/2);
        predictor_x(start : fin) = 1;
        predictor_x = conv(predictor_x,gauss_kernel,'same');
        predictor_x = predictor_x/max(predictor_x);
    end
end
%}
for bs = 1: bins_short
    for i = 1: size(lmcenter_predictor,1)
        if (animal_location(i)-lmcenter_loc > boxcar_short_edges(bs)) && (animal_location(i)-lmcenter_loc < boxcar_short_edges(bs+1))
            lmcenter_predictor(i,bs) = 1;
        end
    end
    lmcenter_predictor(:,bs) = conv(lmcenter_predictor(:,bs), gauss_kernel, 'same');
end

end