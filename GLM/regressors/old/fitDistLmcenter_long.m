function lmcenter_predictor = fitDistLmcenter_long(behav_ds)
% based on fit_distances_lmcenter in fig_response_GLM.py
% returns spatial predictors based on distance from landmark and updated
% roi_gcamp
TRACK_END_LONG = 400;
TRACK_LONG = 4;

bins_long = 18;
animal_location_long_idx = behav_ds(:,5) == TRACK_LONG; % behav_ds[:,4] in python
animal_location = behav_ds(animal_location_long_idx,2); % behav_ds[animal_location_short_idx,1] in python

% determine size (cm) of individual spatial bins
binsize_long = TRACK_END_LONG/bins_long;

% matrix that holds predictors
lmcenter_predictor = zeros(size(animal_location, 1), bins_long);
lmcenter_loc = 220;

% calculate bin edges and centers
boxcar_long_edges = linspace(-180,180,binsize_long+1);

% gaussian kernel
gauss_kernel = gausswin(180,3);
sz_bb = 1:180;
gauss_kernel = gausswin(180,3)/trapz(sz_bb, gauss_kernel);

%create boxcar vectors and convolve with gaussian
for bs = 1: bins_long
    for i = 1: size(lmcenter_predictor,1)
        if (animal_location(i)-lmcenter_loc > boxcar_long_edges(bs)) && (animal_location(i)-lmcenter_loc < boxcar_long_edges(bs+1))
            lmcenter_predictor(i,bs) = 1;
        end
    end
    lmcenter_predictor(:,bs) = conv(lmcenter_predictor(:,bs), gauss_kernel, 'same');
end

end