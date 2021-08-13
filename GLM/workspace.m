clear;

MOUSE = 'LF191022_1';
SESSION = '20191209';
SESS_NAME = 'M01_000_000';
load('D:\Lukas\data\animals_raw\LF191022_1\20191209\data_glm_fit_spike\2.mat')
% 2,3,6

loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
processed_data_path = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\', SESS_NAME, '_results');
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';

lmcenter_pred_long = fitDistLmcenter(behav_ds, 4);
predictor_long_x_0 = makeSpeedPred(behav_ds,lmcenter_pred_long, 4);
predictor_long_x = make_reward_predicts(behav_ds, predictor_long_x_0, 4);
[predictor_long_x_fit,predictor_long_x_test] = split_test_and_fit_behave(predictor_long_x,behav_ds,4);

%%
%{
predictor_selection = [1,2,3];

predictor_long_x_test_subset = predictor_long_x_test(:,predictor_selection);
fit_long_subset = fit_long;
fit_long_subset.beta = fit_long_subset.beta(predictor_selection,:);
fit_long_subset.dim(1) = length(predictor_selection);
spike_rate_pred_long = glmnetPredict(fit_long_subset,predictor_long_x_test_subset,lam_long,'response');
%}
%%
lam_long = fit_long.lambda(end);
%coef_cur = abs(coef_long(2:end));
coef_cur = coef_long(2:end);
[big,sssort] = sort(coef_cur,'descend');
test = predictor_long_x_test(:,sssort);
vals = zeros(28,1);

%%
for x = 1:28
    pred_selc = 1:x;
    fit_long_subset = fit_long;
    fit_long_subset.beta = fit_long_subset.beta(sssort,:);
    fit_long_subset.beta = fit_long_subset.beta(pred_selc,:);
    fit_long_subset.dim(1) = length(pred_selc);
    tp = test(:,pred_selc);
    spike_rate_pred_long_cc = glmnetPredict(fit_long_subset,tp,lam_long,'response');
    
    vals(x) = r2_score(spike_rate_test_long, spike_rate_fit_long, spike_rate_pred_long_cc);
end
%%




%{
loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
processed_data_path = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\', SESS_NAME, '_results');
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';
dF_ds = loaded_data.C_dec;
ROI_amount = size(dF_ds,1);

lmcenter_pred_short = fitDistLmcenter_exp(behav_ds, 3);
predictor_short_x_0 = makeSpeedPred_exp(behav_ds,lmcenter_pred_short, 3);
predictor_short_x = make_reward_predicts_exp(behav_ds, predictor_short_x_0, 3); 
[predictor_short_x_fit,predictor_short_x_test] = split_test_and_fit_behave(predictor_short_x,behav_ds,3);

    
lmcenter_pred_long = fitDistLmcenter_exp(behav_ds, 4);
predictor_long_x_0 = makeSpeedPred_exp(behav_ds,lmcenter_pred_long, 4);
predictor_long_x = make_reward_predicts_exp(behav_ds, predictor_long_x_0, 4);
[predictor_long_x_fit,predictor_long_x_test] = split_test_and_fit_behave(predictor_long_x,behav_ds,4);

%}


%{
load('D:\Lukas\data\animals_raw\LF191023_blue\20191113\M01_000_003_results.mat')
imim = A_save(:,:,99)*100;
%image(imim*100);
load('D:\Lukas\data\animals_raw\LF191023_blue\20191208\M01_000_002_results.mat')
imim2 = circshift(circshift(A_save(:,:,:),-35,1),45,2);
%figure;
save('D:\Lukas\data\animals_raw\LF191023_blue\20191208\new_a_save.mat', 'imim2');
%image(imim+imim2(:,:,144)*100);
%}
%{
load('D:\Lukas\data\animals_raw\LF191022_1\20191209\data_glm\20.mat')
load('D:\Lukas\data\animals_raw\LF191022_1\20191209\M01_000_000_results.mat')
cdc = roi_return_short(behaviour_aligned.', dF_aligned(20,:));
[rl, rw] = splitmat(cdc.');
plot(spike_rate_fit_short,'LineWidth',10);
hold on;
plot(spike_rate_fit_test_short,'LineWidth',10);
%}
%{
loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = ['LF191022_1___';'LF191022_3___';'LF191023_blue'];
MOUSE = MOUSE(3,1:end);
SESSION = ['20191114';'20191209';'20191113';'20191207';'20191113';'20191208'];
SESSION = SESSION(6,:);
SESS_NUM = ['M01_000_002';'M01_000_000';'M01_000_000';'M01_000_002';'M01_000_003';'M01_000_002';];
SESS_NUM = SESS_NUM(6,:);
file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION,'\',SESS_NUM,'_results.mat');
data = load(file);
num_roi = size(data.C_dec,1);


for i = 1:num_roi   
    file_roi = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\data_glm_poisson_c\',int2str(i),'.mat');
    data_roit = load(file_roi);
        if data_roit.coef_short(22) > 2
            figure;
            makeFigure2(data_roit.coef_short(2:end));
            disp('Short');
            disp(file_roi);
        elseif data_roit.coef_long(22) > 2
            figure;
            makeFigure2(data_roit.coef_long(2:end));
            disp('Long');
            disp(file_roi);
        end
end

%}






