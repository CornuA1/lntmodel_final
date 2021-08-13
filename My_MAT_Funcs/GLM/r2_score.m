function [r2] = r2_score(test, fit, pred)
%test is real neural data from the test set
%fit is the real neural data used for fitting (the predicted training data)
%pred is the prediction of test from the test set
% Driscoll R^2 implementation
ave = nanmean(fit);
sum_model = 0;
sum_null = 0;
for i=1:length(test)
    cur_model = test(i) * log(test(i)/pred(i)) - (test(i)-pred(i));
    cur_null = test(i) * log(test(i)/ave) - (test(i)-ave);
    if not(isnan(cur_model) || isnan(cur_null))
        sum_model = sum_model + cur_model;
        sum_null = sum_null + cur_null;
    end
end
sum_model = sum_model * 2;
sum_null = sum_null * 2;
r2 = real(1 - (sum_model/sum_null));
end