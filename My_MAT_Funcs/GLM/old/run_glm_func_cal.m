function run_glm_func_cal(FILE, MOUSE)

disp(['Loading ',FILE,' and ',MOUSE]);

% load data
processed_data_path = strcat(FILE, MOUSE);
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';
dF_ds = loaded_data.dF_aligned;
ROI_amount = size(dF_ds,1);

load(strcat(FILE, 'glm_files.mat'));
pre_check_list = load(strcat(FILE, 'prunned_cell_bodies.mat'));
check_list = pre_check_list.res_log;

options = glmnetSet();
options.lambda_min = 0.15;
options.alpha = 0.9;
visit = false;
for ROI=1:ROI_amount
    if check_list(ROI)
    disp(int2str(ROI));

    spike_rate_0 = dF_ds(ROI,:).';
    min_rate = min(smooth(spike_rate_0,100));
    spike_rate_0 = smooth(spike_rate_0,100) - min_rate;
    
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
    
    save_loc = strcat(FILE,'data_glm_fit_cal\',int2str(ROI),'.mat');
    if ~visit
        status = mkdir(strcat(FILE,'data_glm_fit_cal'));
        visit = true;
    end
    save(save_loc,'fit_short','spike_rate_fit_short','spike_rate_fit_test_short','coef_short','spike_rate_test_short','spike_rate_pred_short','r2_short',...
    'fit_long','spike_rate_fit_long','spike_rate_fit_test_long','coef_long','spike_rate_test_long','spike_rate_pred_long','r2_long')

    disp(int2str(ROI));
    end
end

disp(['Done with ',FILE,' and ',MOUSE]);

end