function final_res = o_shit(behav_ds,Track_size)
% based on fit_distances_lmcenter in fig_response_GLM.py
% returns spatial predictors based on distance from landmark and updated
% roi_gcamp
TRACK_END_LONG = 400;
behav_ds = behav_ds.';


bins_long = 18;
animal_location_long_idx = behav_ds(:,5) == Track_size; % behav_ds[:,4] in python
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

animal_speed = behav_ds(animal_location_long_idx,4); % behav_ds[animal_location_short_idx,3] in python
% split running speed into slow and fast
speed_bins = 3;
% below speed_threshold (cm/sec) = slow
speed_threshold = 0.5;
speed_long_x = zeros(size(animal_speed,1), speed_bins);
max_speed = nanmax(animal_speed);

for i= 1:size(speed_long_x,1)
    if animal_speed(i) < speed_threshold
        speed_long_x(i,1) = 1;
    else
        speed_long_x(i,2) = 1;
    end
    speed_long_x(i,3) = animal_speed(i)/max_speed;
end

lmcenter_predictor = cat(2,lmcenter_predictor,speed_long_x);

animal_location = behav_ds(animal_location_long_idx,2); % behav_ds[animal_location_short_idx,1] in python
lick_location = behav_ds(animal_location_long_idx,8); % behav_ds[animal_location_short_idx,1] in python
reward_location = behav_ds(animal_location_long_idx,6); % behav_ds[animal_location_short_idx,5] in python
% First is for the animal being in the land mark
% Second is for the onset of landmark
% Third is for the offset of landmark
% Fourth is the reward location
loc_long_x = zeros(size(animal_location,1), 3);

% Find the locations and rewards
for i= 1:size(loc_long_x,1)
    if lick_location(i) == 1
        loc_long_x(i,1) = 1;
    end
    if reward_location(i) == 1 || reward_location(i) == 2
        loc_long_x(i,2) = 1;
    end
    if i ~= 1
    if animal_location(i-1) > 200 && animal_location(i) < 100
        loc_long_x(i,3) = 1;
    end       
    end
end

% Convolve signals and append
loc_1 = conv(loc_long_x(:,1), gauss_kernel, 'same');
pred = cat(2,lmcenter_predictor,loc_1);
loc_4 = conv(loc_long_x(:,2), gauss_kernel, 'same');
loc_4 = loc_4/max(loc_4);
pred = cat(2,pred,circshift(loc_4, -120));
pred = cat(2,pred,circshift(loc_4, -60));
pred = cat(2,pred,loc_4);
loc_5 = conv(loc_long_x(:,3), gauss_kernel, 'same');
loc_5 = loc_5/max(loc_5);
pred = cat(2,pred,loc_5);
pred = cat(2,pred,circshift(loc_5, 60));
final_res = cat(2,pred,circshift(loc_5, 120));

for x = 1:size(final_res,2)
    varrr = var(final_res(:,x));
    if varrr < 1.0e-05
        final_res(:,x) = zeros(size(final_res(:,x),1),1);
    end
end

end

