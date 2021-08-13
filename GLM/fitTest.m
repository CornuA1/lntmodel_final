function [r2] = fitTest(behav_ds, roi_gcamp)
NUM_PREDICTORS = 20;
[x_predictors, location_vector] = makeSpatialPred();
%n = 100;
%stdev = 10;
%alpha = (n-1)/(2*stdev);
%gauss_kernel = gausswin(n,alpha);
%boxcar_short_edges = linspace(TRACK_START, TRACK_END_SHORT,
%NUM_PREDICTORS+1);
animal_loc = behav_ds(1:40000,2);
animal_loc_test = behav_ds(40001:end,2);
roi_gcamp_test = roi_gcamp(40001:end);
roi_gcamp = roi_gcamp(1:40000);
predictor_short_x = zeros(size(animal_loc,1), NUM_PREDICTORS);
predictor_short_x_test = zeros(size(animal_loc_test,1), NUM_PREDICTORS);
x_predictors_T = transpose(x_predictors);


for i = 1:size(animal_loc,1)
     for j = 1:size(x_predictors_T,1)
           spx = x_predictors_T(j, :);
           [~,ind] = min(abs(location_vector - animal_loc(i)));
           predictor_short_x(i,j) = 1 * spx(ind);
     end
end

for i = 1:size(animal_loc_test,1)
     for j = 1:size(x_predictors_T,1)
           spx = x_predictors_T(j, :);
           [~,ind] = min(abs(location_vector - animal_loc_test(i)));
           predictor_short_x_test(i,j) = 1 * spx(ind);
     end
end

% fit glmnet
glmnet_fit = glmnet(predictor_short_x, roi_gcamp, 'poisson');
lam = glmnet_fit.lambda(end);
glmnet_coef = glmnetCoef(glmnet_fit,lam,false);
%roi_gcamp_model = glmnetPredict(glmnet_fit,predictor_short_x,100,'response');
roi_gcamp_pred = glmnetPredict(glmnet_fit,predictor_short_x_test,lam, 'response');
%glmnetPrint(glmnet_fit);
%calculate r2 score
%[r2_model] = r2_score(roi_gcamp, roi_gcamp_model);
[r2] = r2_score(roi_gcamp_test, roi_gcamp_pred);

%plot
subplot(5,1,1)
plot(predictor_short_x)
subplot(5,1,2)
plot(glmnet_coef)
subplot(5,1,3)
yyaxis left
plot(animal_loc, 'k')
hold on
yyaxis right
plot(roi_gcamp, 'g')
hold off
subplot(5,1,4)
yyaxis left
plot(animal_loc_test, 'k')
hold on
yyaxis right
plot(roi_gcamp_test,'g')
plot(roi_gcamp_pred,'r')
hold off
subplot(5,1,5)
plot(roi_gcamp_test-roi_gcamp_pred)

end