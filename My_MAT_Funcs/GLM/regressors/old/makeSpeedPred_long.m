function predictor_long_x = makeSpeedPred_long(behav_ds, predictor_long_x)
% based on fit_location() in fig_response_GLM.py
animal_location_long_idx = behav_ds(:,5) == 4; % behav_ds[:,4] in python
animal_speed = behav_ds(animal_location_long_idx,4); % behav_ds[animal_location_short_idx,3] in python
% split running speed into slow and fast
speed_bins = 3;
% below speed_threshold (cm/sec) = slow
speed_threshold_1 = 0.5;
speed_threshold_2 = 1;
speed_long_x = zeros(size(animal_speed,1), speed_bins);
max_speed = nanmax(animal_speed);

for i= 1:size(speed_long_x,1)
    if animal_speed(i) < speed_threshold_1
        speed_long_x(i,1) = 1;
    elseif animal_speed(i) > speed_threshold_2
        speed_long_x(i,2) = 1;
    end
    speed_long_x(i,3) = animal_speed(i)/max_speed;
end
predictor_long_x = cat(2,predictor_long_x,speed_long_x);
end