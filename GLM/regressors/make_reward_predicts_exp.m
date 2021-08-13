function final_res = make_reward_predicts_exp(behave, pred, track_len)
% Function is used to make the predictors for landmark locations
% and for reward instances.
% Signals are convolvd with guassian pulses.

% gaussian kernel to convolve boxcar spatial predictors with

gauss_kernel = exp(linspace(-4,.75,60));
sz_bb = 1:60;
gauss_kernel = gauss_kernel/trapz(sz_bb, gauss_kernel);
gauss_kernel = flip(gauss_kernel);


% find landmark onsets
animal_location_short_idx = behave(:,5) == track_len; % behav_ds[:,4] in python
animal_location = behave(animal_location_short_idx,2); % behav_ds[animal_location_short_idx,1] in python
lick_location = behave(animal_location_short_idx,8); % behav_ds[animal_location_short_idx,1] in python
reward_location = behave(animal_location_short_idx,6); % behav_ds[animal_location_short_idx,5] in python
% First is for the animal being in the land mark
% Second is for the onset of landmark
% Third is for the offset of landmark
% Fourth is the reward location
loc_short_x = zeros(size(animal_location,1), 3);

% Find the locations and rewards
for i= 1:size(loc_short_x,1)
    if lick_location(i) == 1
        loc_short_x(i,1) = 1;
    end
    if reward_location(i) == 1 || reward_location(i) == 2
        loc_short_x(i,2) = 1;
    end
    if i ~= 1
    if animal_location(i-1) > 200 && animal_location(i) < 100
        loc_short_x(i,3) = 1;
    end        
    end
end

% Convolve signals and append
loc_1 = conv(loc_short_x(:,1), gauss_kernel, 'same');
pred = cat(2,pred,loc_1);
loc_4 = conv(loc_short_x(:,2), gauss_kernel, 'same');
loc_4 = loc_4/max(loc_4);
pred = cat(2,pred,circshift(loc_4, -120));
pred = cat(2,pred,circshift(loc_4, -60));
pred = cat(2,pred,loc_4);
loc_5 = conv(loc_short_x(:,3), gauss_kernel, 'same');
loc_5 = loc_5/max(loc_5);
pred = cat(2,pred,loc_5);
pred = cat(2,pred,circshift(loc_5, 60));
final_res = cat(2,pred,circshift(loc_5, 120));

for x = 1:size(final_res,2)
    varrr = var(final_res(:,x));
    if varrr < 1.0e-04
        final_res(:,x) = zeros(size(final_res(:,x),1),1);
    end
end

end