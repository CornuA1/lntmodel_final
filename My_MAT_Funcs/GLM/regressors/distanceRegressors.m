function lmcenter_predictor = distanceRegressors(behav_ds)
% based on fit_distances_lmcenter in fig_response_GLM.py
% returns spatial predictors based on distance from landmark and updated
%  && (animal_speed(i) > 1)

bins = 18;
animal_track = behav_ds(:,5); % behav_ds[:,4] in python
animal_location = behav_ds(:,2); % behav_ds[animal_location_short_idx,1] in python
animal_speed = behav_ds(:,4);

% matrix that holds predictors
lmcenter_predictor = zeros(size(animal_location, 1), 2*bins);
lmcenter_loc = 220;

% calculate bin edges and centers
boxcar_short_edges = linspace(-180,180,bins+1);

% gaussian kernel
gauss_kernel = gausswin(120,4);
sz_bb = 1:120;
gauss_kernel = gausswin(120,4)/trapz(sz_bb, gauss_kernel);

%create boxcar vectors and convolve with gaussian
for bs = 1: bins
    for i = 1: size(lmcenter_predictor,1)
        if (animal_location(i)-lmcenter_loc > boxcar_short_edges(bs)) && (animal_location(i)-lmcenter_loc < boxcar_short_edges(bs+1)) && (animal_track(i) == 3)
            lmcenter_predictor(i,bs) = 1;
        end
    end
%    lmcenter_predictor(:,bs) = conv(lmcenter_predictor(:,bs), gauss_kernel, 'same')/(max(conv(lmcenter_predictor(:,bs), gauss_kernel, 'same')));
    lmcenter_predictor(:,bs) = conv(lmcenter_predictor(:,bs), gauss_kernel, 'same');
    for i = 1: size(lmcenter_predictor,1)
        if lmcenter_predictor(i,bs) >= 1
            lmcenter_predictor(i,bs) = 0;
        end
    end
end

for bs = 1: bins
    for i = 1: size(lmcenter_predictor,1)
        if (animal_location(i)-lmcenter_loc > boxcar_short_edges(bs)) && (animal_location(i)-lmcenter_loc < boxcar_short_edges(bs+1)) && (animal_track(i) == 4)
            lmcenter_predictor(i,bs+bins) = 1;
        end
    end
%    lmcenter_predictor(:,bs+bins) = conv(lmcenter_predictor(:,bs+bins), gauss_kernel, 'same')/(max(conv(lmcenter_predictor(:,bs+bins), gauss_kernel, 'same')));
    lmcenter_predictor(:,bs+bins) = conv(lmcenter_predictor(:,bs+bins), gauss_kernel, 'same');
    for i = 1: size(lmcenter_predictor,1)
        if lmcenter_predictor(i,bs+bins) >= 1
            lmcenter_predictor(i,bs+bins) = 0;
        end
    end
end
end