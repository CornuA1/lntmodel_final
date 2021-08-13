function predictor_short_x = makeSpeedPred(behav_ds, predictor_short_x, track_len)
% based on fit_location() in fig_response_GLM.py
animal_location_short_idx = behav_ds(:,5) == track_len; % behav_ds[:,4] in python
animal_speed = behav_ds(animal_location_short_idx,4); % behav_ds[animal_location_short_idx,3] in python
% split running speed into slow and fast
speed_bins = 3;

gauss_kernel = gausswin(50,5);
sz_bb = 1:50;
gauss_kernel = gausswin(50,5)/trapz(sz_bb, gauss_kernel);

% below speed_threshold (cm/sec) = stop
speed_threshold_1 = 0.5;
speed_threshold_2 = 2;
speed_short_x = zeros(size(animal_speed,1), speed_bins);
max_speed = nanmax(animal_speed);

for i= 1:size(speed_short_x,1)
    if animal_speed(i) < speed_threshold_1
        speed_short_x(i,1) = 1;
    elseif animal_speed(i) > speed_threshold_2
        speed_short_x(i,2) = 1;
    end
    speed_short_x(i,3) = animal_speed(i)/max_speed;
end

speed_short_x(:,1) = conv(speed_short_x(:,1), gauss_kernel, 'same');
speed_short_x(:,2) = conv(speed_short_x(:,2), gauss_kernel, 'same');


predictor_short_x = cat(2,predictor_short_x,speed_short_x);
end



