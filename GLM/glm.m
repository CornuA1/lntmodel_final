% load data
loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = 'LF191022_1';
SESSION = '20191207';
ROI = 2;
processed_data_path = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\', 'aligned_data');
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned;
dF_ds = loaded_data.spikerate;
roi_gcamp_0 = dF_ds(:,ROI);
% make spatial predictors
%[predictor_short_x, location_vector] = makeSpatialPred();
[lmcenter_pred, roi_gcamp] = fitDistLmcenter(behav_ds,roi_gcamp_0);
predictor_short_x_0 = makeSpeedPred(behav_ds,lmcenter_pred);
predictor_short_x = make_reward_predicts(behav_ds, predictor_short_x_0);
% apply glmnet
%[r2] = fitTest(behav_ds, dF_ds(:,ROI));

cv = false;
split = true;
if cv
    % cross validation glmnet
    cvobj = cvglmnet(predictor_short_x, roi_gcamp, 'poisson');
    %cvglmnetPlot(cvobj)
    link_pred = cvglmnetPredict(cvobj, predictor_short_x,'lambda_min');
elseif split
    predictor_short_x_test = predictor_short_x(14001:end,:);
    predictor_short_x_fit = predictor_short_x(1:14000,:);
    roi_gcamp_test = roi_gcamp(14001:end);
    roi_gcamp_fit = roi_gcamp(1:14000);
    options = glmnetSet();
    options.alpha = 0.1;
    fit = glmnet(predictor_short_x_fit,roi_gcamp_fit, 'poisson', options);
    lam = fit.lambda(end); % lambda min
    dev = fit.dev(end); % deviance explained
    coef = glmnetCoef(fit,lam,false);
    glmnetPrint(fit)
    roi_gcamp_pred = glmnetPredict(fit,predictor_short_x_test,lam,'response');
    roi_gcamp_fit_test = glmnetPredict(fit,predictor_short_x_fit,lam,'response');
    [r2] = r2_score(roi_gcamp_test, roi_gcamp_pred);
    makeFigure(roi_gcamp_fit,roi_gcamp_fit_test,coef,roi_gcamp_test,roi_gcamp_pred);
else
    fit = glmnet(predictor_short_x,roi_gcamp, 'poisson');
    lam = fit.lambda(end); % lambda min
    dev = fit.dev(end); % deviance explained
    coef = glmnetCoef(fit,lam,false);
    glmnetPrint(fit)
    roi_gcamp_pred = glmnetPredict(fit,predictor_short_x,lam,'response');
    [r2] = r2_score(roi_gcamp, roi_gcamp_pred);
    makeFigure(predictor_short_x,coef,roi_gcamp,roi_gcamp,roi_gcamp_pred)
end