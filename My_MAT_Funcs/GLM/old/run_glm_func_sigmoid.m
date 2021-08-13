function run_glm_func_sigmoid(SESS_NAME, MOUSE, SESSION)

disp(['Loading ',SESSION,' and ',SESS_NAME]);

% load data
loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
processed_data_path = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\', SESS_NAME, '_results');
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';
dF_ds = loaded_data.C_dec;
ROI_amount = size(dF_ds,1);

lmcenter_pred_short = makeSpeedPred_sig(behav_ds, 3);
predictor_short_x_0 = fitDistLmcenter_sig(behav_ds, lmcenter_pred_short, 3);
predictor_short_x = make_reward_predicts(behav_ds, predictor_short_x_0, 3); 
[predictor_short_x_fit,predictor_short_x_test] = split_test_and_fit_behave(predictor_short_x,behav_ds,3);

lmcenter_pred_long = makeSpeedPred_sig(behav_ds, 4);
predictor_long_x_0 = fitDistLmcenter_sig(behav_ds, lmcenter_pred_long, 4);
predictor_long_x = make_reward_predicts(behav_ds, predictor_long_x_0, 4);
[predictor_long_x_fit,predictor_long_x_test] = split_test_and_fit_behave(predictor_long_x,behav_ds,4);

options = glmnetSet();
options.alpha = 0.9;

for ROI=1:1
    
    disp(int2str(ROI));

    spike_rate_0 = dF_ds(ROI,:).';
    min_rate = min(spike_rate_0)*ones(1,size(spike_rate_0,2));
    spike_rate_0 = spike_rate_0 - min_rate;
    
    % make predictors
    
    spike_rate_short = roi_return_short(behav_ds,spike_rate_0);
    
    spike_rate_long = roi_return_long(behav_ds,spike_rate_0);
    
    % split the data into training and testing
    [spike_rate_fit_short,spike_rate_test_short] = split_test_and_fit_behave(spike_rate_short,behav_ds,3);
    [spike_rate_fit_long,spike_rate_test_long] = split_test_and_fit_behave(spike_rate_long,behav_ds,4);
    
    % apply glmnet
    fit_short = glmnet(predictor_short_x_fit,spike_rate_fit_short, 'poisson', options);
    lam_short = fit_short.lambda(end); % lambda min
    dev_short = fit_short.dev(end); % deviance explained
    coef_short = glmnetCoef(fit_short,lam_short);
    spike_rate_pred_short = glmnetPredict(fit_short,predictor_short_x_test,lam_short,'response');
    spike_rate_fit_test_short = glmnetPredict(fit_short,predictor_short_x_fit,lam_short,'response');
    [r2_short] = r2_score(spike_rate_test_short, spike_rate_fit_short, spike_rate_pred_short);
%    makeFigure(spike_rate_fit_short,spike_rate_fit_test_short,coef_short,spike_rate_test_short,spike_rate_pred_short);
    
    fit_long = glmnet(predictor_long_x_fit,spike_rate_fit_long, 'poisson', options);
    lam_long = fit_long.lambda(end); % lambda min
    dev_long = fit_long.dev(end); % deviance explained
    coef_long = glmnetCoef(fit_long,lam_long);
    spike_rate_pred_long = glmnetPredict(fit_long,predictor_long_x_test,lam_long,'response');
    spike_rate_fit_test_long = glmnetPredict(fit_long,predictor_long_x_fit,lam_long,'response');
    [r2_long] = r2_score(spike_rate_test_long, spike_rate_fit_long, spike_rate_pred_long);
%    makeFigure(spike_rate_fit_long,spike_rate_fit_test_long,coef_long,spike_rate_test_long,spike_rate_pred_long);
    
    save_loc = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\data_glm_fit_sig\',int2str(ROI),'.mat');
    if ROI == 1
        status = mkdir(strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\data_glm_fit_sig'));
    end
    save(save_loc,'fit_short','spike_rate_fit_short','spike_rate_fit_test_short','coef_short','spike_rate_test_short','spike_rate_pred_short','r2_short',...
    'fit_long','spike_rate_fit_long','spike_rate_fit_test_long','coef_long','spike_rate_test_long','spike_rate_pred_long','r2_long')
end

disp(['Done with ',SESSION,' and ',SESS_NAME]);

end