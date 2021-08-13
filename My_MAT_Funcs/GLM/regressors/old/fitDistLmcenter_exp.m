function lmcenter_predictor = fitDistLmcenter_exp(behav_ds, track_len)
% based on fit_distances_lmcenter in fig_response_GLM.py
% returns spatial predictors based on distance from landmark and updated

bins_short = 18;
animal_location_short_idx = behav_ds(:,5) == track_len; % behav_ds[:,4] in python
animal_location = behav_ds(animal_location_short_idx,2); % behav_ds[animal_location_short_idx,1] in python

% matrix that holds predictors
lmcenter_predictor = zeros(size(animal_location, 1), bins_short);
lmcenter_loc = 220;

% calculate bin edges and centers
boxcar_short_edges = linspace(-180,180,bins_short+1);

% gaussian kernel
gauss_kernel = exp(linspace(-4,.75,60));
sz_bb = 1:60;
gauss_kernel = gauss_kernel/trapz(sz_bb, gauss_kernel);
gauss_kernel = flip(gauss_kernel);

for bs = 1: bins_short
    for i = 1: size(lmcenter_predictor,1)
        if (animal_location(i)-lmcenter_loc > boxcar_short_edges(bs)) && (animal_location(i)-lmcenter_loc < boxcar_short_edges(bs+1))
            lmcenter_predictor(i,bs) = 1;
        end
    end
    lmcenter_predictor(:,bs) = conv(lmcenter_predictor(:,bs), gauss_kernel, 'same');
end

end