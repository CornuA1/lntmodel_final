clear;
% load data
loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = 'LF191022_1';
SESSION = '20191207';
ROI = 1;
processed_data_path = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\', 'M01_000_000_results');
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';
dF_ds = loaded_data.S_dec;
spike_rate_0 = dF_ds(ROI,:).';

% make predictors
[lmcenter_pred_short, spike_rate_short] = fitDistLmcenter(behav_ds,spike_rate_0);
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

% apply glmnet
fit_short = glmnet(predictor_short_x_fit,spike_rate_fit_short, 'poisson', options);
lam_short = fit_short.lambda(end); % lambda min
dev_short = fit_short.dev(end); % deviance explained
coef_short = glmnetCoef(fit_short,lam_short);
glmnetPrint(fit_short)
spike_rate_pred_short = glmnetPredict(fit_short,predictor_short_x_test,lam_short,'response');
spike_rate_fit_test_short = glmnetPredict(fit_short,predictor_short_x_fit,lam_short,'response');
[r2_short] = r2_score(spike_rate_test_short, spike_rate_pred_short);
makeFigure2(spike_rate_fit_short,spike_rate_fit_test_short,coef_short,spike_rate_test_short,spike_rate_pred_short);

fit_long = glmnet(predictor_long_x_fit,spike_rate_fit_long, 'poisson', options);
lam_long = fit_long.lambda(end); % lambda min
dev_long = fit_long.dev(end); % deviance explained
coef_long = glmnetCoef(fit_long,lam_long);
glmnetPrint(fit_long)
spike_rate_pred_long = glmnetPredict(fit_long,predictor_long_x_test,lam_long,'response');
spike_rate_fit_test_long = glmnetPredict(fit_long,predictor_long_x_fit,lam_long,'response');
[r2_long] = r2_score(spike_rate_test_long, spike_rate_pred_long);
makeFigure2(spike_rate_fit_long,spike_rate_fit_test_long,coef_long,spike_rate_test_long,spike_rate_pred_long);

