function create_glm_regressors(FILE, MOUSE)

disp(['Loading ',FILE,' and ',MOUSE]);

% load data
processed_data_path = strcat(FILE, MOUSE);
% processed_data_path = strcat(FILE); % LF DEBUG
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';

lmcenter_pred = distanceRegressors(behav_ds);
speed_pred = speedRegressors(behav_ds,lmcenter_pred);
regressors = rewardRegressors(behav_ds, speed_pred); 
regressors(isnan(regressors)) = 0;

save_loc = strcat(FILE,'GLMregs.mat');
save(save_loc,'regressors')
end