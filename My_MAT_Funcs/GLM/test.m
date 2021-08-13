function test(FILE, MOUSE, data_type, fit_dist)

% data_type = {'calcium', 'detrended' , 'deconvolved'}
% fit_dist = {'gaussian', 'poisson'}

disp(['Loading ',FILE,' and ',MOUSE]);

% load data
processed_data_path = strcat(FILE, MOUSE);
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';
if strcmp(data_type,'calcium')
    dF_ds = loaded_data.new_dF.';
elseif strcmp(data_type,'detrended')
    dF_ds = loaded_data.deconv.';
elseif strcmp(data_type,'deconvolved')
    dF_ds = loaded_data.spikes.';
end
ROI_amount = size(dF_ds,1);

trial = (behav_ds.behaviour_aligned(7,end) - 1)/2;
trials = 1:trial;
trials = trials*2-1;

glmFiles = load(strcat(FILE, 'GLMregs.mat')).regressors;
pre_check_list = load(strcat(FILE, 'prunned_cell_bodies.mat'));
check_list = pre_check_list.res_log;

options = glmnetSet();
options.lambda_min = 0.2;
options.alpha = 0.9;

for ROI=1:ROI_amount
if check_list(ROI)
    disp(int2str(ROI));
    sigUse = dF_ds(:,ROI);
    execOrder = trials(randperm(length(trials))); 
    
    for val=1:length(execOrder)
        valueTrial = execOrder(val);
        trialIn = behav_ds(7,:) == valueTrial;
        trialEx = not(trialIn);
        out = glmRun(glmFiles(trialEx,:), glmFiles(trialIn,:), sig2fit, sig2test, fit_dist)
    end
    
%{
    save_loc = strcat(FILE,'data_glm_fit_news\',int2str(ROI),'.mat');
    if ~visit
        status = mkdir(strcat(FILE,'data_glm_fit_news'));
        visit = true;
    end
    save(save_loc,'fit_short','spike_rate_fit_short','spike_rate_fit_test_short','coef_short','spike_rate_test_short','spike_rate_pred_short','r2_short',...
    'fit_long','spike_rate_fit_long','spike_rate_fit_test_long','coef_long','spike_rate_test_long','spike_rate_pred_long','r2_long')
%}
end
end

disp(['Done with ',FILE,' and ',MOUSE]);

end