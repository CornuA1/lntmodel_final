function speedReg = speedRegressors(behav_ds, distanceReg)
% based on fit_location() in fig_response_GLM.py
animal_speed = behav_ds(:,4); % behav_ds[animal_location_short_idx,3] in python
% split running speed into slow and fast

gauss_kernel = gausswin(50,5);
sz_bb = 1:50;
gauss_kernel = gausswin(50,5)/trapz(sz_bb, gauss_kernel);

% below speed_threshold (cm/sec) = stop
speed_threshold_1 = 0.3333;
speed_threshold_2 = 3.5;
speed_regs = zeros(size(animal_speed,1), 3);
max_speed = nanmax(animal_speed);

for i= 1:size(speed_regs,1)
    if (abs(animal_speed(i)) < speed_threshold_1)
        speed_regs(i,1) = 1;
    elseif (abs(animal_speed(i)) > speed_threshold_2)
        speed_regs(i,2) = 1;
    end
    speed_regs(i,3) = animal_speed(i)/max_speed;
end

speed_regs(:,1) = conv(speed_regs(:,1), gauss_kernel, 'same');
speed_regs(:,2) = conv(speed_regs(:,2), gauss_kernel, 'same');

speedReg = cat(2,distanceReg,speed_regs);
end



