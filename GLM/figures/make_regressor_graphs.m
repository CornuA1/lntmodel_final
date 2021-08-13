clear;

SESS_NAME = 'M01_000_003';
MOUSE = 'LF191022_1';
SESSION = '20191121';

disp(['Loading ',SESSION,' and ',SESS_NAME]);

% load data
loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
processed_data_path = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\', SESS_NAME, '_results');
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';
dF_ds = loaded_data.S_dec;
ROI_amount = size(dF_ds,1);

for ROI=1:1
    
    disp(int2str(ROI));

    spike_rate_0 = dF_ds(ROI,:).';
    
    % make predictors
    [lmcenter_pred_short, spike_rate_short] = fitDistLmcenter(behav_ds,spike_rate_0);
    makeFigure3(lmcenter_pred_short(:,6), lmcenter_pred_short(:,7), lmcenter_pred_short(:,8), lmcenter_pred_short(:,9), lmcenter_pred_short(:,10))
    predictor_short_x_0 = makeSpeedPred(behav_ds,lmcenter_pred_short);
    predictor_short_x = make_reward_predicts(behav_ds, predictor_short_x_0);
    
    [lmcenter_pred_long, spike_rate_long] = fitDistLmcenter_long(behav_ds,spike_rate_0);
    predictor_long_x_0 = makeSpeedPred_long(behav_ds,lmcenter_pred_long);
    predictor_long_x = make_reward_predicts_long(behav_ds, predictor_long_x_0);
    
    % split the data into training and testing
    [predictor_short_x_fit,predictor_short_x_test] = splitmat(predictor_short_x);
    [spike_rate_fit_short,spike_rate_test_short] = splitmat(spike_rate_short);
    [predictor_long_x_fit,predictor_long_x_test] = splitmat(predictor_long_x);
    [spike_rate_fit_long,spike_rate_test_long] = splitmat(spike_rate_long);
    options = glmnetSet();
    options.alpha = 0.1;
    
    
end

disp(['Done with ',SESSION,' and ',SESS_NAME]);
