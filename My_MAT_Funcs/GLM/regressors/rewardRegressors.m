function rewardReg = rewardRegressors(behave, speedReg)
% Function is used to make the predictors for landmark locations
% and for reward instances.
% Signals are convolvd with guassian pulses.
% gaussian kernel to convolve boxcar spatial predictors with
gauss_kernel = gausswin(120,4);
sz_bb = 1:120;
gauss_kernel = gausswin(120,4)/trapz(sz_bb, gauss_kernel);
% find landmark onsets

animal_track = behave(:,5); % behav_ds[:,4] in python
animal_location = behave(:,2); % behav_ds[animal_location_short_idx,1] in python
lick_location = behave(:,8); % behav_ds[animal_location_short_idx,1] in python
reward_location = behave(:,6); % behav_ds[animal_location_short_idx,5] in python
% First is for the animal being in the land mark
% Second is for the onset of landmark
% Third is for the offset of landmark
% Fourth is the reward location
loc_short = zeros(size(animal_location,1), 3);

% Find the locations and rewards
for i= 1:size(loc_short,1)
    if lick_location(i) == 1
        loc_short(i,1) = 1;
    end
    if (reward_location(i) == 1 || reward_location(i) == 2) && animal_track(i) == 3
        loc_short(i,2) = 1;
    end
    if i ~= 1
    if animal_location(i-1) > 200 && animal_location(i) < 100 && animal_track(i) == 3
        loc_short(i,3) = 1;
    end        
    end
end

% Convolve signals and append
loc_1 = conv(loc_short(:,1), gauss_kernel, 'same');
pred = cat(2,speedReg,loc_1);
loc_4 = conv(loc_short(:,2), gauss_kernel, 'same');
loc_4 = loc_4/max(loc_4);
pred = cat(2,pred,circshift(loc_4, -120)); % CHANGE for frame shift in delay
pred = cat(2,pred,circshift(loc_4, -60)); % CHANGE for frame shift in delay
pred = cat(2,pred,loc_4);
loc_5 = conv(loc_short(:,3), gauss_kernel, 'same');
loc_5 = loc_5/max(loc_5);
pred = cat(2,pred,loc_5);
pred = cat(2,pred,circshift(loc_5, 60)); % CHANGE for frame shift in delay
pred = cat(2,pred,circshift(loc_5, 120)); % CHANGE for frame shift in delay

% Long trials
loc_long = zeros(size(animal_location,1), 2);

% Find the locations and rewards
for i= 1:size(loc_long,1)
    if (reward_location(i) == 1 || reward_location(i) == 2) && animal_track(i) == 4
        loc_long(i,1) = 1;
    end
    if i ~= 1
    if animal_location(i-1) > 200 && animal_location(i) < 100 && animal_track(i) == 4
        loc_long(i,2) = 1;
    end        
    end
end

% Convolve signals and append
loc_4 = conv(loc_long(:,1), gauss_kernel, 'same');
loc_4 = loc_4/max(loc_4);
pred = cat(2,pred,circshift(loc_4, -120)); % CHANGE for frame shift in delay
pred = cat(2,pred,circshift(loc_4, -60)); % CHANGE for frame shift in delay
pred = cat(2,pred,loc_4);
loc_5 = conv(loc_long(:,2), gauss_kernel, 'same');
loc_5 = loc_5/max(loc_5);
pred = cat(2,pred,loc_5);
pred = cat(2,pred,circshift(loc_5, 60)); % CHANGE for frame shift in delay
rewardReg = cat(2,pred,circshift(loc_5, 120)); % CHANGE for frame shift in delay

% The following for loop discards regressors with low variance by setting
% all of the values to zero. Without this the GLM can take advantage of low
% variance regressors. Although, this may need to be taken out if it
% interfers with the naive sessions too much.
for x = 1:size(rewardReg,2)
    varrr = var(rewardReg(:,x));
    if varrr < 1.0e-04
        rewardReg(:,x) = zeros(size(rewardReg(:,x),1),1);
    end
end

end