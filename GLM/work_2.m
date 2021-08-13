clear;
load('C:\Users\Lou\Documents\MATLAB\GLM\some_data.mat');
options = glmnetSet();
options.alpha = 0.9;
vals = zeros(28,1);
%%
for x = 1:28
tp = test_t(:,1:x);
fp = fit_t(:,1:x);
    
fit_long_c = glmnet(fp,spike_rate_fit_long, 'poisson', options);
lam_long_c = fit_long_c.lambda(end); % lambda min
   
spike_rate_pred_long_cc = glmnetPredict(fit_long_c,tp,lam_long_c,'response');

vals(x) = r2_score(spike_rate_test_long, spike_rate_fit_long, spike_rate_pred_long_cc);
end